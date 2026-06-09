from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye embeddings ROI medios a partir de embeddings patch-level de test."
    )
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    return parser.parse_args()


def extract_roi_id_from_filename(file_name: str) -> str:
    # Por ejemplo: BRACS_1855_N_2_3.jpeg -> BRACS_1855_N_2
    stem = Path(file_name).stem
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"No se puede extraer roi_id de: {file_name}")
    return "_".join(parts[:-1])


def main() -> None:
    args = parse_args()

    h5_path = Path(args.h5_path)
    metadata_path = Path(args.metadata_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Cargando H5: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        features = f["features"][:]
        labels = f["labels"][:]
        row_ids = f["row_ids"][:]

    print(f"[INFO] Cargando metadata CSV: {metadata_path}")
    meta = pd.read_csv(metadata_path)

    if len(meta) != len(features):
        raise ValueError("La metadata no coincide con el número de embeddings.")
    if not np.array_equal(meta["row_id"].to_numpy(), row_ids):
        raise ValueError("Los row_id del CSV no coinciden con los del H5.")
    if not np.array_equal(meta["label"].to_numpy(), labels):
        raise ValueError("Las labels del CSV no coinciden con las del H5.")

    # Construcción de roi_id
    if "file_name" in meta.columns:
        meta["roi_id"] = meta["file_name"].apply(extract_roi_id_from_filename)
    else:
        meta["file_name"] = meta["path"].apply(lambda p: Path(p).name)
        meta["roi_id"] = meta["file_name"].apply(extract_roi_id_from_filename)

    print(f"[INFO] Nº patches: {len(meta)}")
    print(f"[INFO] Nº ROIs únicas: {meta['roi_id'].nunique()}")

    # Agrupación por ROI
    roi_rows = []
    roi_embeddings = []

    for roi_id, idx in meta.groupby("roi_id", sort=True).groups.items():
        idx = np.array(list(idx), dtype=int)

        roi_feats = features[idx]
        roi_labels = meta.iloc[idx]["label"].unique()

        if len(roi_labels) != 1:
            raise ValueError(f"La ROI {roi_id} tiene varias etiquetas: {roi_labels.tolist()}")

        mean_embedding = roi_feats.mean(axis=0)

        roi_rows.append({
            "roi_id": roi_id,
            "y_true_roi": int(roi_labels[0]),
            "class_name": meta.iloc[idx]["class_name"].iloc[0],
            "n_patches": int(len(idx)),
            "model": args.model,
        })
        roi_embeddings.append(mean_embedding)

    roi_meta_df = pd.DataFrame(roi_rows)
    roi_embeddings_np = np.stack(roi_embeddings, axis=0)

    csv_out = output_dir / f"{args.model}_test_roi_embeddings_metadata.csv"
    npy_out = output_dir / f"{args.model}_test_roi_embeddings.npy"

    roi_meta_df.to_csv(csv_out, index=False)
    np.save(npy_out, roi_embeddings_np)

    print("[INFO] Embeddings ROI construidos correctamente.")
    print(f"[INFO] Metadata ROI guardada en: {csv_out}")
    print(f"[INFO] Matriz ROI guardada en: {npy_out}")
    print(f"[INFO] Shape ROI embeddings: {roi_embeddings_np.shape}")


if __name__ == "__main__":
    main()