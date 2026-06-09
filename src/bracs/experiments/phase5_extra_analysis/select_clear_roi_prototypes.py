from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CLASS_NAMES = {
    0: "N",
    1: "PB",
    2: "UDH",
    3: "FEA",
    4: "ADH",
    5: "DCIS",
    6: "IC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selecciona ROIs prototípicas claras con estrategia jerárquica por clase."
    )
    parser.add_argument("--roi_predictions_csv", type=str, required=True)
    parser.add_argument("--review_cases_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)

    # Umbrales
    parser.add_argument("--strict_top1_prob", type=float, default=0.90)
    parser.add_argument("--relaxed_top1_prob", type=float, default=0.80)
    parser.add_argument("--min_margin", type=float, default=0.40)
    parser.add_argument("--min_patches", type=int, default=3)

    # Mínimo deseado por clase
    parser.add_argument("--min_per_class", type=int, default=5)

    return parser.parse_args()


def base_filter(
    df: pd.DataFrame,
    review_roi_ids: set[str],
    min_top1_prob: float,
    min_margin: float,
    min_patches: int,
) -> pd.DataFrame:
    out = df.copy()
    out = out[out["y_pred_mean_proba"] == out["y_true_roi"]]
    out = out[~out["roi_id"].astype(str).isin(review_roi_ids)]
    out = out[out["top1_prob"] >= min_top1_prob]
    out = out[out["margin_top1_top2"] >= min_margin]
    out = out[out["n_patches"] >= min_patches]
    return out


def main() -> None:
    args = parse_args()

    roi_path = Path(args.roi_predictions_csv)
    review_path = Path(args.review_cases_csv)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Cargando ROI predictions: {roi_path}")
    roi_df = pd.read_csv(roi_path)

    print(f"[INFO] Cargando review cases: {review_path}")
    review_df = pd.read_csv(review_path)

    review_roi_ids = set(review_df["roi_id"].astype(str).tolist())

    # Añadimos metadata útil
    roi_df = roi_df.copy()
    roi_df["class_name"] = roi_df["y_true_roi"].map(CLASS_NAMES)
    roi_df["model"] = args.model
    roi_df["method"] = args.method

    # Selección estricta
    strict_df = base_filter(
        roi_df,
        review_roi_ids=review_roi_ids,
        min_top1_prob=args.strict_top1_prob,
        min_margin=args.min_margin,
        min_patches=args.min_patches,
    ).copy()
    strict_df["selection_level"] = "strict"

    # Selección relajada candidata
    relaxed_candidates_df = base_filter(
        roi_df,
        review_roi_ids=review_roi_ids,
        min_top1_prob=args.relaxed_top1_prob,
        min_margin=args.min_margin,
        min_patches=args.min_patches,
    ).copy()

    # Orden común: primero mayor top1_prob, luego mayor margen
    sort_cols = ["y_true_roi", "top1_prob", "margin_top1_top2"]
    strict_df = strict_df.sort_values(by=sort_cols, ascending=[True, False, False])
    relaxed_candidates_df = relaxed_candidates_df.sort_values(
        by=sort_cols, ascending=[True, False, False]
    )

    selected_parts = []
    summary_rows = []

    for class_id, class_name in CLASS_NAMES.items():
        strict_cls = strict_df[strict_df["y_true_roi"] == class_id].copy()

        # Empezamos con los estrictos
        selected_cls = strict_cls.copy()

        n_strict = len(strict_cls)
        n_added_relaxed = 0

        if n_strict < args.min_per_class:
            needed = args.min_per_class - n_strict

            relaxed_cls = relaxed_candidates_df[
                (relaxed_candidates_df["y_true_roi"] == class_id)
                & (~relaxed_candidates_df["roi_id"].isin(strict_cls["roi_id"]))
            ].copy()

            relaxed_cls = relaxed_cls.head(needed).copy()
            relaxed_cls["selection_level"] = "relaxed"

            n_added_relaxed = len(relaxed_cls)
            selected_cls = pd.concat([selected_cls, relaxed_cls], ignore_index=True)

        selected_parts.append(selected_cls)

        summary_rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "n_strict": n_strict,
            "n_added_relaxed": n_added_relaxed,
            "n_final": len(selected_cls),
        })

    final_df = pd.concat(selected_parts, ignore_index=True)

    # Orden final
    final_df = final_df.sort_values(
        by=["y_true_roi", "selection_level", "top1_prob", "margin_top1_top2"],
        ascending=[True, True, False, False],
    )

    cols_to_keep = [
        "roi_id",
        "model",
        "method",
        "y_true_roi",
        "class_name",
        "selection_level",
        "n_patches",
        "y_pred_mean_proba",
        "top1_class",
        "top1_prob",
        "top2_class",
        "top2_prob",
        "margin_top1_top2",
        "entropy",
    ]
    final_df = final_df[cols_to_keep]

    final_df.to_csv(out_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_path.with_name(out_path.stem + "_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("[INFO] Selección jerárquica de prototipos completada.")
    print(f"[INFO] Nº ROIs claras finales: {len(final_df)}")
    print(f"[INFO] CSV guardado en: {out_path}")
    print(f"[INFO] Resumen guardado en: {summary_csv}")

    print("\n[INFO] Resumen por clase:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()