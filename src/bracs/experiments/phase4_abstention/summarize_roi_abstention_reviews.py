# ---------------------------------------------
# RESUMEN DE REVISIONES DE ABSTENCIÓN DE ROI
# ---------------------------------------------
# Entrada:
#   - Archivos CSV de casos para revisión (*_review_cases.csv)
#
# Salida:
#   - Archivos CSV resumen con:
#       * Casos de revisión concatenados para trazabilidad
#       * Conteos de abstención acumulados por clase real
#       * Conteos de abstención por par top1-top2
#       * Totales globales de abstención por modelo y método
# ---------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CLASS_NAMES_7 = {
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
        description="Resume ROIs dudosas por clase real y por par top1-top2."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Directorio base donde se encuentran almacenados los CSV de casos en revisión.",
    )
    parser.add_argument(
        "--tau_tag",
        type=str,
        default="tau010",
        help="Etiqueta correspondiente al valor de tau (ej. tau010).",
    )
    return parser.parse_args()


def parse_model_method(path: Path, tau_tag: str) -> tuple[str, str]:
    """
    Deduce el modelo y el método a partir de la ruta del archivo de casos de revisión.
    """
    model = path.parent.name
    stem = path.stem  # e.g. baseline_tau010_review_cases
    suffix = f"_{tau_tag}_review_cases"
    if not stem.endswith(suffix):
        raise ValueError(f"No se puede parsear el método desde {stem}")
    method = stem[: -len(suffix)]
    return model, method


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)

    # Buscamos todos los archivos correspondientes al tau especificado recursivamente
    review_case_files = sorted(base_dir.rglob(f"*_{args.tau_tag}_review_cases.csv"))

    if not review_case_files:
        raise FileNotFoundError(
            f"No se encontraron archivos *_{args.tau_tag}_review_cases.csv en {base_dir}"
        )

    all_review_rows = []
    class_summary_rows = []
    pair_summary_rows = []

    # Procesamos individualmente cada archivo encontrado
    for review_path in review_case_files:
        model, method = parse_model_method(review_path, args.tau_tag)
        df = pd.read_csv(review_path)

        # Copiamos y añadimos metadata para mantener la trazabilidad de los casos individuales
        df_all = df.copy()
        df_all["model"] = model
        df_all["method"] = method
        all_review_rows.append(df_all)

        # Generamos resumen de abstención acumulado por clase real
        cls_counts = (
            df["y_true_roi"]
            .value_counts()
            .sort_index()
            .rename_axis("y_true_roi")
            .reset_index(name="count")
        )
        cls_counts["class_name"] = cls_counts["y_true_roi"].map(CLASS_NAMES_7)
        cls_counts["model"] = model
        cls_counts["method"] = method
        cls_counts = cls_counts[["model", "method", "y_true_roi", "class_name", "count"]]
        class_summary_rows.append(cls_counts)

        # Generamos resumen de abstención por par de predicciones top1-top2 dudosas
        pair_counts = (
            df.groupby(["top1_class", "top2_class"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        pair_counts["top1_name"] = pair_counts["top1_class"].map(CLASS_NAMES_7)
        pair_counts["top2_name"] = pair_counts["top2_class"].map(CLASS_NAMES_7)
        pair_counts["model"] = model
        pair_counts["method"] = method
        pair_counts = pair_counts[
            ["model", "method", "top1_class", "top1_name", "top2_class", "top2_name", "count"]
        ]
        pair_summary_rows.append(pair_counts)

    # Concatenamos la información agregando todos los archivos de revisión
    all_reviews_df = pd.concat(all_review_rows, ignore_index=True)
    class_summary_df = pd.concat(class_summary_rows, ignore_index=True)
    pair_summary_df = pd.concat(pair_summary_rows, ignore_index=True)

    # Obtenemos conteos globales agrupados por modelo y método
    totals_df = (
        all_reviews_df.groupby(["model", "method"])
        .size()
        .reset_index(name="n_review_cases")
        .sort_values(["model", "method"])
    )

    out_dir = base_dir / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_reviews_path = out_dir / f"all_review_cases_{args.tau_tag}.csv"
    class_summary_path = out_dir / f"review_by_true_class_{args.tau_tag}.csv"
    pair_summary_path = out_dir / f"review_by_top1_top2_pair_{args.tau_tag}.csv"
    totals_path = out_dir / f"review_totals_{args.tau_tag}.csv"

    # Exportamos todos los CSVs resúmenes finales
    all_reviews_df.to_csv(all_reviews_path, index=False)
    class_summary_df.to_csv(class_summary_path, index=False)
    pair_summary_df.to_csv(pair_summary_path, index=False)
    totals_df.to_csv(totals_path, index=False)

    print("\n[INFO] Totales de ROIs dudosas por configuración:")
    print(totals_df.to_string(index=False))

    print("\n[INFO] Resumen por clase real:")
    print(class_summary_df.to_string(index=False))

    print("\n[INFO] Top pares top1-top2 por configuración:")
    for (model, method), g in pair_summary_df.groupby(["model", "method"]):
        print(f"\n=== {model} | {method} ===")
        print(g.head(10).to_string(index=False))

    print(f"\n[INFO] CSV guardado en: {all_reviews_path}")
    print(f"[INFO] CSV guardado en: {class_summary_path}")
    print(f"[INFO] CSV guardado en: {pair_summary_path}")
    print(f"[INFO] CSV guardado en: {totals_path}")


if __name__ == "__main__":
    main()