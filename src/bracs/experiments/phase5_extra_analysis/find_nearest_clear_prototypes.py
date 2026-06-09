# ---------------------------------------------
# BÚSQUEDA DE PROTOTIPOS CLAROS MÁS CERCANOS
# ---------------------------------------------
# Entrada:
#   - Archivo NumPy (.npy) con los embeddings de las ROIs
#   - Archivo CSV con metadata de las ROIs
#   - Archivo CSV con los prototipos claros seleccionados
#   - Archivo CSV de casos dudosos (revisión)
#   - Archivo CSV con las predicciones de ROI
#
# Salida:
#   - Archivo CSV con los k prototipos más cercanos para cada ROI dudosa y sus métricas de similitud
# ---------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Busca los prototipos claros más cercanos para cada ROI dudosa."
    )
    parser.add_argument("--roi_embeddings_npy", type=str, required=True)
    parser.add_argument("--roi_embeddings_meta_csv", type=str, required=True)
    parser.add_argument("--clear_prototypes_csv", type=str, required=True)
    parser.add_argument("--review_cases_csv", type=str, required=True)
    parser.add_argument("--roi_predictions_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    emb_path = Path(args.roi_embeddings_npy)
    meta_path = Path(args.roi_embeddings_meta_csv)
    clear_path = Path(args.clear_prototypes_csv)
    review_path = Path(args.review_cases_csv)
    roi_pred_path = Path(args.roi_predictions_csv)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Cargando embeddings ROI: {emb_path}")
    X = np.load(emb_path)

    print(f"[INFO] Cargando metadata ROI: {meta_path}")
    roi_meta = pd.read_csv(meta_path)

    print(f"[INFO] Cargando prototipos claros: {clear_path}")
    clear_df = pd.read_csv(clear_path)

    print(f"[INFO] Cargando ROIs dudosas (review cases): {review_path}")
    review_df = pd.read_csv(review_path)

    print(f"[INFO] Cargando predicciones ROI: {roi_pred_path}")
    roi_pred_df = pd.read_csv(roi_pred_path)

    # Validamos que coincida la longitud de metadata y embeddings
    if len(roi_meta) != len(X):
        raise ValueError("La metadata ROI no coincide con la matriz de embeddings ROI.")

    # Generamos un mapeo rápido de roi_id a su índice en la matriz de embeddings
    roi_to_idx = {roi_id: i for i, roi_id in enumerate(roi_meta["roi_id"].astype(str).tolist())}

    # Conservamos únicamente los prototipos y casos de revisión válidos en nuestra metadata
    clear_df = clear_df[clear_df["roi_id"].astype(str).isin(roi_to_idx)].copy()
    review_df = review_df[review_df["roi_id"].astype(str).isin(roi_to_idx)].copy()

    print(f"[INFO] Nº prototipos utilizables: {len(clear_df)}")
    print(f"[INFO] Nº ROIs dudosas utilizables: {len(review_df)}")

    # Extraemos los embeddings de los prototipos
    clear_indices = [roi_to_idx[rid] for rid in clear_df["roi_id"].astype(str)]
    X_clear = X[clear_indices]

    rows = []

    # Para cada ROI dudosa, calculamos su similitud coseno con todos los prototipos claros
    for _, review_row in review_df.iterrows():
        roi_id = str(review_row["roi_id"])
        idx = roi_to_idx[roi_id]
        x_query = X[idx].reshape(1, -1)

        # Calculamos la similitud coseno
        sims = cosine_similarity(x_query, X_clear)[0]
        top_idx = np.argsort(sims)[::-1][: args.top_k]

        # Recuperamos la información predictiva de la ROI dudosa
        pred_row = roi_pred_df[roi_pred_df["roi_id"].astype(str) == roi_id]
        if len(pred_row) != 1:
            raise ValueError(f"No se encontró exactamente una fila en roi_predictions para {roi_id}")
        pred_row = pred_row.iloc[0]

        # Guardamos la información de los K prototipos más cercanos
        for rank, proto_pos in enumerate(top_idx, start=1):
            proto = clear_df.iloc[proto_pos]

            rows.append({
                "roi_id_dudosa": roi_id,
                "y_true_dudosa": int(pred_row["y_true_roi"]),
                "top1_class_dudosa": int(pred_row["top1_class"]),
                "top1_prob_dudosa": float(pred_row["top1_prob"]),
                "top2_class_dudosa": int(pred_row["top2_class"]),
                "top2_prob_dudosa": float(pred_row["top2_prob"]),
                "margin_top1_top2_dudosa": float(pred_row["margin_top1_top2"]),
                "entropy_dudosa": float(pred_row["entropy"]),
                "n_patches_dudosa": int(pred_row["n_patches"]),

                "prototype_rank": rank,
                "prototype_roi_id": str(proto["roi_id"]),
                "prototype_class": int(proto["y_true_roi"]),
                "prototype_class_name": proto["class_name"],
                "prototype_selection_level": proto["selection_level"],
                "prototype_top1_prob": float(proto["top1_prob"]),
                "prototype_margin": float(proto["margin_top1_top2"]),
                "prototype_n_patches": int(proto["n_patches"]),

                "cosine_similarity": float(sims[proto_pos]),
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    print("[INFO] Vecinos prototípicos calculados correctamente.")
    print(f"[INFO] CSV guardado en: {out_path}")
    print(f"[INFO] Filas generadas: {len(out_df)}")


if __name__ == "__main__":
    main()