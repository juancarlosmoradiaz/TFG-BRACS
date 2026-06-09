# ---------------------------------------------
# EVALUACIÓN DE PREDICCIONES DE ROI PARA 3 CLASES
# ---------------------------------------------
# Entrada:
#   - Archivo CSV con predicciones mapeadas a 3 clases diagnósticas
#
# Salida:
#   - Archivo JSON con las métricas acumuladas del experimento
#   - Archivos CSV e imágenes PNG de las matrices de confusión en absoluto y porcentaje
# ---------------------------------------------

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score



CLASS_NAMES = ["AT", "BT", "MT"]


# =========================================================
# ARGUMENTOS
# =========================================================
def parse_args() -> argparse.Namespace:
    """
    Parsea los argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description="Evalúa predicciones ROI en 3 clases."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Ruta al CSV de entrada con predicciones en 3 clases diagnósticas.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directorio donde se guardarán las métricas y gráficos resultantes.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Nombre del modelo evaluado (ej. virchow2, h_optimus_1).",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Nombre del método o configuración de votación.",
    )
    return parser.parse_args()



def save_confusion_matrix_figure(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str,
    as_percentage: bool = False,
) -> None:
    """
    Dibuja y guarda la matriz de confusión en formato PNG.
    Soporta visualización en valores numéricos o en porcentajes.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Etiqueta real",
        xlabel="Predicción",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if as_percentage:
                text_value = f"{cm[i, j]:.1f}%"
            else:
                text_value = f"{cm[i, j]:.0f}"

            ax.text(
                j,
                i,
                text_value,
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Cargando CSV: {input_path}")
    df = pd.read_csv(input_path)

    # Validamos las columnas requeridas en el CSV
    required_cols = {"roi_id", "y_true_roi_3cls", "y_pred_3cls"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")

    y_true = df["y_true_roi_3cls"].to_numpy(dtype=int)
    y_pred = df["y_pred_3cls"].to_numpy(dtype=int)

    # Cálculo de métricas agregadas en 3 clases
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
    )

    # Cálculo de las matrices de confusión (conteos absolutos y porcentajes normalizados)
    cm_counts = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_pct = cm_counts.astype(float) / cm_counts.sum(axis=1, keepdims=True) * 100.0

    # Diccionario de métricas finales
    metrics = {
        "model": args.model,
        "method": args.method,
        "n_rois": int(len(df)),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "report": report,
    }

    metrics_path = output_dir / f"{args.method}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Generación y almacenamiento de DataFrames para matrices de confusión en CSV
    cm_counts_df = pd.DataFrame(cm_counts, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_pct_df = pd.DataFrame(cm_pct, index=CLASS_NAMES, columns=CLASS_NAMES)

    cm_counts_csv = output_dir / f"{args.method}_cm_counts.csv"
    cm_pct_csv = output_dir / f"{args.method}_cm_pct.csv"
    cm_counts_df.to_csv(cm_counts_csv)
    cm_pct_df.to_csv(cm_pct_csv)

    cm_counts_png = output_dir / f"{args.method}_cm_counts.png"
    cm_pct_png = output_dir / f"{args.method}_cm_pct.png"

    # Generar y guardar las figuras gráficas de las matrices de confusión
    save_confusion_matrix_figure(
        cm=cm_counts,
        class_names=CLASS_NAMES,
        save_path=cm_counts_png,
        title=f"Matriz de confusión (conteos) - {args.model} - {args.method}",
        as_percentage=False,
    )

    save_confusion_matrix_figure(
        cm=cm_pct,
        class_names=CLASS_NAMES,
        save_path=cm_pct_png,
        title=f"Matriz de confusión (%) - {args.model} - {args.method}",
        as_percentage=True,
    )

    print("[INFO] Evaluación 3-clases completada correctamente.")
    print(f"[INFO] N ROIs: {len(df)}")
    print(f"[INFO] Accuracy: {acc:.6f}")
    print(f"[INFO] F1 macro: {f1_macro:.6f}")
    print(f"[INFO] Metrics JSON: {metrics_path}")
    print(f"[INFO] CM counts CSV: {cm_counts_csv}")
    print(f"[INFO] CM pct CSV: {cm_pct_csv}")
    print(f"[INFO] CM counts PNG: {cm_counts_png}")
    print(f"[INFO] CM pct PNG: {cm_pct_png}")


if __name__ == "__main__":
    main()