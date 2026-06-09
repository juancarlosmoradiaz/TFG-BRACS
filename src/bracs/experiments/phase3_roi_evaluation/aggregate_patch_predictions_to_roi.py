# ---------------------------------------------
# AGREGACIÓN DE PREDICCIONES DE PATCHES A NIVEL DE ROI
# ---------------------------------------------
# Objetivo:
#   - Agrupar predicciones generadas a nivel de patch para cada región de interés (ROI)
#   - Calcular predicciones agregadas mediante diferentes reglas de votación:
#       * Promedio de probabilidades (mean_proba)
#       * Voto por mayoría (majority_vote)
#       * Criterio de mayor malignidad (most_malignant)
#   - Extraer métricas adicionales como entropía y margen entre las clases top 1 y 2
#
# Entrada:
#   - Archivo CSV con predicciones a nivel de patch
#
# Salida:
#   - Archivo CSV con métricas agregadas por ROI
# ---------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# ARGUMENTOS
# =========================================================
def parse_args() -> argparse.Namespace:
    """
    Parsea los argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description="Agrega predicciones patch-level a nivel de ROI."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Ruta al archivo CSV de entrada con predicciones a nivel de patch.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Ruta donde se guardará el archivo CSV resultante con las agregaciones de ROI.",
    )
    parser.add_argument(
        "--n_clases",
        type=int,
        default=7,
        help="Número de clases del problema de clasificación.",
    )
    return parser.parse_args()


# =========================================================
# AUXILIARES / MÉTRICAS
# =========================================================
def majority_vote(arr: np.ndarray) -> int:
    """
    Calcula el voto por mayoría de un array de predicciones.
    En caso de empate, se selecciona la clase con menor índice numérico.
    """
    values, counts = np.unique(arr, return_counts=True)
    max_count = counts.max()
    winners = values[counts == max_count]
    return int(winners.min())  # desempate: menor índice


def safe_entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """
    Calcula la entropía de Shannon de un vector de probabilidades.
    Se añade un epsilon para evitar problemas de indeterminación matemática con log(0).
    """
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log(probs)))


def top_k_from_probs(probs: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    """
    Devuelve las k clases con mayor probabilidad, ordenadas de mayor a menor.
    Cada elemento es una tupla: (class_idx, prob)
    """
    probs = np.asarray(probs, dtype=float)
    sorted_idx = np.argsort(-probs)  # descendente
    top_idx = sorted_idx[:k]
    return [(int(i), float(probs[i])) for i in top_idx]



def main() -> None:
    args = parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Cargando CSV patch-level: {in_path}")
    df = pd.read_csv(in_path)

    # Validamos las columnas requeridas
    required_cols = {"roi_id", "y_true_patch", "y_pred_patch"}
    prob_cols = [f"prob_{i}" for i in range(args.n_clases)]
    required_cols.update(prob_cols)

    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias en el CSV de entrada: {sorted(missing)}"
        )

    roi_rows = []

    # Agrupamos por identificador único de ROI para realizar la agregación
    grouped = df.groupby("roi_id", sort=True)

    for roi_id, g in grouped:
        true_labels = g["y_true_patch"].unique()
        if len(true_labels) != 1:
            raise ValueError(
                f"La ROI {roi_id} tiene varias etiquetas reales patch-level: {true_labels.tolist()}"
            )

        y_true_roi = int(true_labels[0])
        n_patches = int(len(g))

        # Regla 1: Promedio de probabilidades (mean_proba)
        probs = g[prob_cols].to_numpy(dtype=float)
        mean_probs = probs.mean(axis=0)
        y_pred_mean_proba = int(np.argmax(mean_probs))

        # Regla 2: Voto por mayoría
        pred_patch = g["y_pred_patch"].to_numpy(dtype=int)
        y_pred_majority_vote = majority_vote(pred_patch)

        # Regla 3: Mayor malignidad
        y_pred_most_malignant = int(pred_patch.max())

        # Top-3 clases más probables a nivel ROI
        top3 = top_k_from_probs(mean_probs, k=min(3, args.n_clases))

        # Si hubiera menos de 3 clases por cualquier motivo, rellenamos con valores vacíos
        while len(top3) < 3:
            top3.append((-1, float("nan")))

        top1_class, top1_prob = top3[0]
        top2_class, top2_prob = top3[1]
        top3_class, top3_prob = top3[2]

        margin_top1_top2 = float(top1_prob - top2_prob)
        entropy = safe_entropy(mean_probs)

        # Estructura del registro para la ROI actual
        row = {
            "roi_id": roi_id,
            "y_true_roi": y_true_roi,
            "n_patches": n_patches,
            "y_pred_mean_proba": y_pred_mean_proba,
            "y_pred_majority_vote": y_pred_majority_vote,
            "y_pred_most_malignant": y_pred_most_malignant,
            "top1_class": top1_class,
            "top1_prob": top1_prob,
            "top2_class": top2_class,
            "top2_prob": top2_prob,
            "top3_class": top3_class,
            "top3_prob": top3_prob,
            "margin_top1_top2": margin_top1_top2,
            "entropy": entropy,
        }

        # Guardamos las probabilidades promedio para cada una de las clases
        for i in range(args.n_clases):
            row[f"mean_prob_{i}"] = float(mean_probs[i])

        roi_rows.append(row)

    out_df = pd.DataFrame(roi_rows)

    # Orden de columnas para dejarlo limpio y trazable
    ordered_cols = [
        "roi_id",
        "y_true_roi",
        "n_patches",
        "y_pred_mean_proba",
        "y_pred_majority_vote",
        "y_pred_most_malignant",
        "top1_class",
        "top1_prob",
        "top2_class",
        "top2_prob",
        "top3_class",
        "top3_prob",
        "margin_top1_top2",
        "entropy",
    ] + [f"mean_prob_{i}" for i in range(args.n_clases)]

    out_df = out_df[ordered_cols]

    # Guardamos los resultados agregados
    out_df.to_csv(out_path, index=False)

    print("[INFO] Agregación ROI completada correctamente.")
    print(f"[INFO] Nº ROIs: {len(out_df)}")
    print(f"[INFO] CSV guardado en: {out_path}")


if __name__ == "__main__":
    main()