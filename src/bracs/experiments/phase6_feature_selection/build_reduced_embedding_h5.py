# ---------------------------------------------
# CONSTRUCCIÓN DE H5 DE EMBEDDINGS REDUCIDOS
# ---------------------------------------------
# Entrada:
#   - Archivo H5 original de embeddings (features, labels, row_ids)
#   - Archivo NumPy (.npy) con los índices de variables preseleccionadas
#
# Salida:
#   - Archivo H5 con los embeddings reducidos (filtrando por las columnas indicadas)
# ---------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye un H5 reducido usando un subconjunto de variables seleccionadas."
    )
    parser.add_argument("--input_h5", type=str, required=True)
    parser.add_argument("--selected_idx_npy", type=str, required=True)
    parser.add_argument("--output_h5", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_h5 = Path(args.input_h5)
    selected_idx_npy = Path(args.selected_idx_npy)
    output_h5 = Path(args.output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    if not input_h5.exists():
        raise FileNotFoundError(f"No existe input_h5: {input_h5}")
    if not selected_idx_npy.exists():
        raise FileNotFoundError(f"No existe selected_idx_npy: {selected_idx_npy}")

    # Cargamos y limpiamos los índices de variables seleccionadas
    selected_idx = np.load(selected_idx_npy).astype(int)
    selected_idx = np.unique(selected_idx)

    print(f"[INFO] Cargando H5: {input_h5}")
    with h5py.File(input_h5, "r") as f:
        features = f["features"][:].astype(np.float32)
        labels = f["labels"][:].astype(np.int64)
        row_ids = f["row_ids"][:].astype(np.int64)

    print(f"[INFO] Shape original features: {features.shape}")
    print(f"[INFO] Nº variables seleccionadas: {len(selected_idx)}")

    # Validamos que los índices se correspondan con la dimensión del embedding
    if selected_idx.min() < 0 or selected_idx.max() >= features.shape[1]:
        raise ValueError("Los índices seleccionados están fuera de rango.")

    # Filtramos la matriz de características conservando solo las columnas seleccionadas
    reduced_features = features[:, selected_idx]

    print(f"[INFO] Shape reducida features: {reduced_features.shape}")

    # Guardamos los resultados comprimidos en formato H5
    with h5py.File(output_h5, "w") as f:
        f.create_dataset("features", data=reduced_features, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
        f.create_dataset("row_ids", data=row_ids, compression="gzip")

    print(f"[INFO] H5 reducido guardado en: {output_h5}")


if __name__ == "__main__":
    main()