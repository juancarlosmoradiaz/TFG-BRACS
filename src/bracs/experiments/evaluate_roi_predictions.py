from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


CLASS_NAMES_7 = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa predicciones a nivel de ROI para una regla de votación concreta."
    )
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument(
        "--voting_method",
        type=str,
        required=True,
        choices=["mean_proba", "majority_vote", "most_malignant"],
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_clases", type=int, default=7)
    return parser.parse_args()


def get_pred_col(voting_method: str) -> str:
    mapping = {
        "mean_proba": "y_pred_mean_proba",
        "majority_vote": "y_pred_majority_vote",
        "most_malignant": "y_pred_most_malignant",
    }
    return mapping[voting_method]


def plot_confusion_matrix_with_percentages(
    cm_abs: np.ndarray,
    cm_pct: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_pct, interpolation="nearest", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm_abs.shape[0]):
        for j in range(cm_abs.shape[1]):
            txt = f"{cm_pct[i, j]:.1f}%\n({cm_abs[i, j]})"
            color = "white" if cm_pct[i, j] >= 50 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    pred_col = get_pred_col(args.voting_method)

    if pred_col not in df.columns:
        raise ValueError(f"No existe la columna de predicción '{pred_col}' en el CSV.")

    y_true = df["y_true_roi"].to_numpy(dtype=int)
    y_pred = df[pred_col].to_numpy(dtype=int)

    labels = list(range(args.n_clases))
    class_names = CLASS_NAMES_7 if args.n_clases == 7 else [str(i) for i in labels]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm_abs = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm_abs.sum(axis=1, keepdims=True)
    cm_pct = np.divide(
        cm_abs * 100.0,
        row_sums,
        out=np.zeros_like(cm_abs, dtype=float),
        where=row_sums != 0,
    )

    stem = input_path.stem.replace("_roi_predictions", "")
    metrics = {
        "input_csv": str(input_path),
        "voting_method": args.voting_method,
        "n_rois": int(len(df)),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "report": report,
    }

    metrics_path = output_dir / f"{stem}_{args.voting_method}_metrics.json"
    cm_abs_csv = output_dir / f"{stem}_{args.voting_method}_cm_abs.csv"
    cm_pct_csv = output_dir / f"{stem}_{args.voting_method}_cm_pct.csv"
    cm_png = output_dir / f"{stem}_{args.voting_method}_cm.png"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pd.DataFrame(cm_abs, index=class_names, columns=class_names).to_csv(cm_abs_csv)
    pd.DataFrame(cm_pct, index=class_names, columns=class_names).to_csv(cm_pct_csv)

    plot_confusion_matrix_with_percentages(
        cm_abs=cm_abs,
        cm_pct=cm_pct,
        class_names=class_names,
        title=f"Matriz de confusión ROI ({args.voting_method})",
        output_path=cm_png,
    )

    print("[INFO] Evaluación ROI completada.")
    print(f"[INFO] voting_method: {args.voting_method}")
    print(f"[INFO] n_rois: {len(df)}")
    print(f"[INFO] accuracy: {acc:.4f}")
    print(f"[INFO] f1_macro: {f1_macro:.4f}")
    print(f"[INFO] Metrics JSON: {metrics_path}")
    print(f"[INFO] CM abs CSV: {cm_abs_csv}")
    print(f"[INFO] CM pct CSV: {cm_pct_csv}")
    print(f"[INFO] CM PNG: {cm_png}")


if __name__ == "__main__":
    main()