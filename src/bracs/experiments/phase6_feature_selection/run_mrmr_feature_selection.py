# ---------------------------------------------
# SELECCIÓN DE VARIABLES CON MRMR (MAXIMUM RELEVANCE MINIMUM REDUNDANCY)
# ---------------------------------------------
# Entrada:
#   - Archivo H5 de embeddings de train
#
# Salida:
#   - Archivo NumPy (.npy) con los índices de las variables seleccionadas por mRMR
#   - Archivo JSON con el resumen del experimento
# ---------------------------------------------

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from mrmr import mrmr_classif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta selección de variables con mRMR sobre embeddings de train."
    )
    parser.add_argument("--train_h5", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--k_features", type=int, required=True)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_h5 = Path(args.train_h5)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_h5.exists():
        raise FileNotFoundError(f"No existe train_h5: {train_h5}")

    print(f"[INFO] Cargando H5: {train_h5}")
    with h5py.File(train_h5, "r") as f:
        X = f["features"][:].astype(np.float32)
        y = f["labels"][:].astype(np.int64)

    print(f"[INFO] Shape X_train: {X.shape}")
    print(f"[INFO] Shape y_train: {y.shape}")

    # Submuestreo opcional para mitigar problemas de memoria y cómputo
    if args.max_samples is not None:
        rng = np.random.default_rng(args.random_state)
        n_total = X.shape[0]
        n_keep = min(args.max_samples, n_total)
        idx = rng.choice(n_total, size=n_keep, replace=False)
        idx = np.sort(idx)
        X = X[idx]
        y = y[idx]
        print(f"[INFO] Submuestreo activado: {n_keep} muestras de {n_total}")
        print(f"[INFO] Nueva shape X_train: {X.shape}")
        print(f"[INFO] Nueva shape y_train: {y.shape}")

    n_features_total = X.shape[1]
    if args.k_features > n_features_total:
        raise ValueError(f"k_features={args.k_features} > n_features_total={n_features_total}")

    # mRMR de la biblioteca externa mrmr requiere DataFrames de Pandas
    col_names = [f"f{i}" for i in range(n_features_total)]
    X_df = pd.DataFrame(X, columns=col_names)
    y_sr = pd.Series(y)

    print(f"[INFO] Ejecutando mRMR con K={args.k_features}")
    selected_features = mrmr_classif(X=X_df, y=y_sr, K=args.k_features)

    # Convertimos los nombres de las columnas ('f123') de vuelta a índices numéricos enteros (123)
    selected_idx = np.array([int(name[1:]) for name in selected_features], dtype=int)
    selected_idx = np.unique(selected_idx)

    n_selected = len(selected_idx)
    pct_selected = 100.0 * n_selected / n_features_total

    print(f"[INFO] Variables totales: {n_features_total}")
    print(f"[INFO] Variables seleccionadas: {n_selected}")
    print(f"[INFO] Porcentaje retenido: {pct_selected:.2f}%")

    idx_path = output_dir / f"{args.model}_mrmr_selected_idx_k{args.k_features}.npy"
    np.save(idx_path, selected_idx)

    # Generamos el informe del experimento en JSON
    summary = {
        "model": args.model,
        "train_h5": str(train_h5),
        "n_samples": int(X.shape[0]),
        "n_features_total": int(n_features_total),
        "n_features_selected": int(n_selected),
        "pct_selected": float(pct_selected),
        "k_features": int(args.k_features),
        "random_state": int(args.random_state),
        "max_samples": None if args.max_samples is None else int(args.max_samples),
    }

    summary_path = output_dir / f"{args.model}_mrmr_summary_k{args.k_features}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Índices guardados en: {idx_path}")
    print(f"[INFO] Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()