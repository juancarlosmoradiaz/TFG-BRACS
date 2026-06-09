from __future__ import annotations

import argparse
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera predicciones patch-level sobre embeddings de test."
    )
    parser.add_argument("--model", type=str, required=True, choices=["h_optimus_1", "virchow2"])
    parser.add_argument("--test_h5", type=str, required=True)
    parser.add_argument("--test_metadata_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method_name", type=str, required=True)
    parser.add_argument("--n_clases", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=1024)
    return parser.parse_args()


def infer_input_dim_from_h5(test_h5: str) -> int:
    with h5py.File(test_h5, "r") as f:
        features = f["features"]
        if len(features.shape) != 2:
            raise ValueError(f"'features' debe ser 2D. Shape recibido: {features.shape}")
        return int(features.shape[1])


def build_linear_head(input_dim: int, n_clases: int) -> nn.Module:
    return nn.Linear(input_dim, n_clases)


def extract_roi_id(file_name: str) -> str:
    stem = Path(file_name).stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"No se puede derivar roi_id desde: {file_name}")
    return "_".join(parts[:-1])


def load_h5_embeddings(h5_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        features = f["features"][:]
        labels = f["labels"][:]
        row_ids = f["row_ids"][:]
    return features, labels, row_ids


def load_linear_weights(checkpoint_path: str, model: nn.Module) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Limpieza por si las claves llevan prefijos
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("classifier."):
            nk = nk.replace("classifier.", "")
        if nk.startswith("head."):
            nk = nk.replace("head.", "")
        if nk.startswith("linear."):
            nk = nk.replace("linear.", "")
        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=False)


def main() -> None:
    args = parse_args()

    input_dim = infer_input_dim_from_h5(args.test_h5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Modelo: {args.model}")
    print(f"[INFO] Método: {args.method_name}")
    print(f"[INFO] Dispositivo: {device}")

    print("[INFO] Cargando embeddings de test...")
    X, y_true, row_ids = load_h5_embeddings(args.test_h5)
    meta = pd.read_csv(args.test_metadata_csv)

    if len(meta) != len(X):
        raise ValueError(
            f"Desalineación entre metadata ({len(meta)}) y embeddings ({len(X)})."
        )

    if "file_name" not in meta.columns:
        raise ValueError("La metadata de test no contiene la columna 'file_name'.")

    print(f"[INFO] Nº patches test: {len(X)}")
    print(f"[INFO] Dimensión embedding: {X.shape[1]}")

    print("[INFO] Cargando cabeza lineal...")
    model = build_linear_head(input_dim=input_dim, n_clases=args.n_clases).to(device)
    load_linear_weights(args.checkpoint, model)
    model.eval()

    probs_all = []
    preds_all = []

    print("[INFO] Generando probabilidades patch-level...")
    with torch.no_grad():
        for start in range(0, len(X), args.batch_size):
            end = min(start + args.batch_size, len(X))
            xb = torch.tensor(X[start:end], dtype=torch.float32, device=device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            probs_all.append(probs)
            preds_all.append(preds)

    probs_all = np.concatenate(probs_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)

    out_df = meta.copy()
    out_df["y_true_patch"] = y_true.astype(int)
    out_df["y_pred_patch"] = preds_all.astype(int)
    out_df["roi_id"] = out_df["file_name"].apply(extract_roi_id)

    for c in range(args.n_clases):
        out_df[f"prob_{c}"] = probs_all[:, c]

    expected_cols = [
        "row_id", "path", "file_name", "roi_id",
        "label", "class_name", "split", "model_name",
        "y_true_patch", "y_pred_patch"
    ]
    prob_cols = [f"prob_{c}" for c in range(args.n_clases)]
    ordered_cols = [c for c in expected_cols if c in out_df.columns] + prob_cols
    out_df = out_df[ordered_cols]

    out_dir = Path("outputs/predictions/test_patches") / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.method_name}_patch_predictions.csv"
    out_df.to_csv(out_path, index=False)

    print("[INFO] Predicciones patch-level generadas correctamente.")
    print(f"[INFO] CSV guardado en: {out_path}")


if __name__ == "__main__":
    main()