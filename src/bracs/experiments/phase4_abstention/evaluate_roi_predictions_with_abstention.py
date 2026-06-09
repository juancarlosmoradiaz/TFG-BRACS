# ---------------------------------------------
# EVALUACIÓN DE PREDICCIONES DE ROI CON ABSTENCIÓN
# ---------------------------------------------
# Entrada:
#   - Archivo CSV con las predicciones a nivel de ROI
#
# Salida:
#   - Archivo JSON con las métricas del experimento con abstención (coverage, review_rate, etc.)
#   - Archivos CSV e imagen PNG con la matriz de confusión sobre los casos aceptados
#   - Archivos CSV con casos para revisión e informe de pares dudosas
# ---------------------------------------------

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


CLASS_NAMES_7 = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa predicciones ROI con abstención por margen top1-top2."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Ruta al archivo CSV con las predicciones de ROI.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        required=True,
        help="Umbral de tolerancia (tau) para la abstención de predicciones dudosas.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directorio de salida donde se guardarán las métricas y gráficos.",
    )
    parser.add_argument(
        "--n_clases",
        type=int,
        default=7,
        help="Número de clases del problema.",
    )
    return parser.parse_args()


def tau_to_tag(tau: float) -> str:
    """
    Genera un tag en formato de texto a partir del valor numérico de tau (ej. 0.10 -> tau010).
    """
    return f"tau{int(round(tau * 100)):03d}"


def plot_confusion_matrix_with_percentages(
    cm_abs: np.ndarray,
    cm_pct: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    """
    Genera un gráfico PNG con la matriz de confusión porcentual y conteos absolutos.
    """
    import matplotlib.pyplot as plt

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

    # Validamos las columnas requeridas
    required_cols = {
        "roi_id",
        "y_true_roi",
        "n_patches",
        "top1_class",
        "top1_prob",
        "top2_class",
        "top2_prob",
        "top3_class",
        "top3_prob",
        "margin_top1_top2",
        "entropy",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")

    tau_tag = tau_to_tag(args.tau)
    stem = input_path.stem.replace("_roi_predictions", "")
    class_names = CLASS_NAMES_7 if args.n_clases == 7 else [str(i) for i in range(args.n_clases)]

    # Aplicamos la regla de abstención en base al margen de probabilidad top1-top2
    df = df.copy()
    df["is_uncertain"] = df["margin_top1_top2"] < args.tau
    df["decision_with_abstention"] = np.where(
        df["is_uncertain"],
        "REVIEW",
        df["top1_class"].astype(int).astype(str),
    )

    df["accepted"] = ~df["is_uncertain"]

    # Separamos en casos aceptados por el sistema y casos enviados a revisión
    accepted_df = df[df["accepted"]].copy()
    review_df = df[df["is_uncertain"]].copy()

    n_total = int(len(df))
    n_accepted = int(len(accepted_df))
    n_review = int(len(review_df))
    coverage = float(n_accepted / n_total) if n_total > 0 else 0.0
    review_rate = float(n_review / n_total) if n_total > 0 else 0.0

    metrics = {
        "input_csv": str(input_path),
        "tau": args.tau,
        "tau_tag": tau_tag,
        "n_rois_total": n_total,
        "n_rois_accepted": n_accepted,
        "n_rois_review": n_review,
        "coverage": coverage,
        "review_rate": review_rate,
    }

    labels = list(range(args.n_clases))

    # Si hay casos aceptados, calculamos su rendimiento de clasificación
    if n_accepted > 0:
        y_true = accepted_df["y_true_roi"].to_numpy(dtype=int)
        y_pred = accepted_df["top1_class"].to_numpy(dtype=int)

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

        metrics["accuracy_accepted"] = float(acc)
        metrics["f1_macro_accepted"] = float(f1_macro)
        metrics["report_accepted"] = report

        cm_abs_path = output_dir / f"{stem}_{tau_tag}_cm_abs_accepted.csv"
        cm_pct_path = output_dir / f"{stem}_{tau_tag}_cm_pct_accepted.csv"
        cm_png_path = output_dir / f"{stem}_{tau_tag}_cm_accepted.png"

        pd.DataFrame(cm_abs, index=class_names, columns=class_names).to_csv(cm_abs_path)
        pd.DataFrame(cm_pct, index=class_names, columns=class_names).to_csv(cm_pct_path)

        plot_confusion_matrix_with_percentages(
            cm_abs=cm_abs,
            cm_pct=cm_pct,
            class_names=class_names,
            title=f"Matriz de confusión ROI aceptadas ({tau_tag})",
            output_path=cm_png_path,
        )
    else:
        metrics["accuracy_accepted"] = None
        metrics["f1_macro_accepted"] = None
        metrics["report_accepted"] = None

    # Distribución de casos en revisión por clase real y combinaciones top1-top2
    if n_review > 0:
        review_true_counts = (
            review_df["y_true_roi"].value_counts().sort_index().to_dict()
        )
        review_pairs = (
            review_df.groupby(["top1_class", "top2_class"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    else:
        review_true_counts = {}
        review_pairs = pd.DataFrame(columns=["top1_class", "top2_class", "count"])

    metrics["review_true_class_counts"] = {str(k): int(v) for k, v in review_true_counts.items()}

    # Rutas de guardado de los ficheros resultantes
    metrics_path = output_dir / f"{stem}_{tau_tag}_metrics.json"
    review_cases_path = output_dir / f"{stem}_{tau_tag}_review_cases.csv"
    review_pairs_path = output_dir / f"{stem}_{tau_tag}_review_pairs_summary.csv"
    all_decisions_path = output_dir / f"{stem}_{tau_tag}_all_decisions.csv"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    review_cols = [
        "roi_id",
        "y_true_roi",
        "n_patches",
        "top1_class",
        "top1_prob",
        "top2_class",
        "top2_prob",
        "top3_class",
        "top3_prob",
        "margin_top1_top2",
        "entropy",
    ]
    review_df[review_cols].to_csv(review_cases_path, index=False)
    review_pairs.to_csv(review_pairs_path, index=False)

    decision_cols = [
        "roi_id",
        "y_true_roi",
        "n_patches",
        "top1_class",
        "top1_prob",
        "top2_class",
        "top2_prob",
        "top3_class",
        "top3_prob",
        "margin_top1_top2",
        "entropy",
        "is_uncertain",
        "accepted",
        "decision_with_abstention",
    ]
    df[decision_cols].to_csv(all_decisions_path, index=False)

    print("[INFO] Evaluación ROI con abstención completada.")
    print(f"[INFO] tau: {args.tau:.3f} ({tau_tag})")
    print(f"[INFO] n_rois_total: {n_total}")
    print(f"[INFO] n_rois_accepted: {n_accepted}")
    print(f"[INFO] n_rois_review: {n_review}")
    print(f"[INFO] coverage: {coverage:.4f}")
    print(f"[INFO] review_rate: {review_rate:.4f}")
    print(f"[INFO] accuracy_accepted: {metrics['accuracy_accepted']}")
    print(f"[INFO] f1_macro_accepted: {metrics['f1_macro_accepted']}")
    print(f"[INFO] Metrics JSON: {metrics_path}")
    print(f"[INFO] Review cases CSV: {review_cases_path}")
    print(f"[INFO] Review pairs CSV: {review_pairs_path}")
    print(f"[INFO] All decisions CSV: {all_decisions_path}")

    if n_accepted > 0:
        print(f"[INFO] CM abs accepted CSV: {cm_abs_path}")
        print(f"[INFO] CM pct accepted CSV: {cm_pct_path}")
        print(f"[INFO] CM accepted PNG: {cm_png_path}")


if __name__ == "__main__":
    main()