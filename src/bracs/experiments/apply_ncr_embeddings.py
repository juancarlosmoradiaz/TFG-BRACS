"""
Aplica NeighbourhoodCleaningRule (NCR) sobre los embeddings de train de un
modelo fundacional:
    1. Carga embeddings, etiquetas y row_ids desde el H5 del split train.
    2. Carga la metadata asociada.
    3. Aplica NeighbourhoodCleaningRule sobre (X, y).
    4. Guarda:
        - embeddings filtrados en formato H5,
        - metadata filtrada,
        - trazabilidad completa keep/drop,
        - summary JSON con estadísticas del proceso.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import pandas as pd
from imblearn.under_sampling import NeighbourhoodCleaningRule


def parse_args() -> argparse.Namespace:
    """
    Define y parsea los argumentos de línea de comandos.

    Returns
    -------
    argparse.Namespace
        Argumentos ya parseados.
    """
    parser = argparse.ArgumentParser(
        description="Aplica NeighbourhoodCleaningRule sobre embeddings de train."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["h_optimus_1", "virchow2"],
        help="Modelo fundacional cuyos embeddings se van a limpiar.",
    )
    parser.add_argument(
        "--n_clases",
        type=int,
        default=7,
        help="Número de clases del problema. Por defecto: 7.",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=3,
        help="Número de vecinos para NCR. Por defecto: 3.",
    )
    parser.add_argument(
        "--threshold_cleaning",
        type=float,
        default=0.5,
        help="Threshold cleaning de NCR. Por defecto: 0.5.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Número de procesos paralelos para NCR. Por defecto: -1.",
    )
    return parser.parse_args()


def get_input_paths(model: str) -> Tuple[Path, Path]:
    """
    Construye las rutas de entrada del H5 y la metadata del split train
    para el modelo indicado.

    Parameters
    ----------
    model : str
        Nombre del modelo fundacional.

    Returns
    -------
    tuple[Path, Path]
        Ruta al H5 de embeddings y ruta al CSV de metadata.
    """
    base_dir = Path("outputs/embeddings")
    h5_path = base_dir / f"{model}_train.h5"
    metadata_path = base_dir / f"{model}_train_metadata.csv"
    return h5_path, metadata_path


def get_output_paths(model: str) -> Dict[str, Path]:
    """
    Construye todas las rutas de salida necesarias para NCR.

    Parameters
    ----------
    model : str
        Nombre del modelo fundacional.

    Returns
    -------
    dict[str, Path]
        Diccionario con las rutas de salida.
    """
    cleaned_dir = Path("outputs/cleaned_embeddings/ncr") / model
    cleaning_dir = Path("outputs/cleaning/ncr") / model

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    cleaning_dir.mkdir(parents=True, exist_ok=True)

    return {
        "h5_out": cleaned_dir / f"{model}_train_ncr.h5",
        "metadata_out": cleaned_dir / f"{model}_train_ncr_metadata.csv",
        "trace_out": cleaning_dir / f"{model}_train_ncr_trace.csv",
        "summary_out": cleaning_dir / f"{model}_train_ncr_summary.json",
    }


def load_h5_data(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga embeddings, etiquetas y row_ids desde un fichero H5.

    Este proyecto guarda los datos dentro del H5 con los nombres:
    - 'features': matriz de embeddings
    - 'labels': vector de etiquetas
    - 'row_ids': identificadores internos

    Parameters
    ----------
    h5_path : Path
        Ruta al fichero H5.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Embeddings, etiquetas y row_ids.
    """
    with h5py.File(h5_path, "r") as f:
        required_keys = ["features", "labels", "row_ids"]
        missing = [k for k in required_keys if k not in f]
        if missing:
            raise KeyError(
                f"Faltan datasets obligatorios en {h5_path}: {missing}. "
                f"Datasets disponibles: {list(f.keys())}"
            )

        features = f["features"][:]
        labels = f["labels"][:]
        row_ids = f["row_ids"][:]

    return features, labels, row_ids


def compute_class_distribution(y: np.ndarray, n_clases: int) -> Dict[str, int]:
    """
    Calcula la distribución por clases y la devuelve como diccionario con
    claves string, fácilmente serializable a JSON.

    Parameters
    ----------
    y : np.ndarray
        Vector de etiquetas.
    n_clases : int
        Número total de clases esperadas.

    Returns
    -------
    dict[str, int]
        Distribución por clases.
    """
    distribution = {}
    for c in range(n_clases):
        distribution[str(c)] = int(np.sum(y == c))
    return distribution


def build_trace_dataframe(
    metadata: pd.DataFrame,
    y_original: np.ndarray,
    row_ids_original: np.ndarray,
    selected_indices: np.ndarray,
    method_name: str,
) -> pd.DataFrame:
    """
    Construye un CSV de trazabilidad patch a patch, marcando si cada muestra
    ha sido conservada o eliminada por el método aplicado.

    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata original completa.
    y_original : np.ndarray
        Etiquetas originales.
    row_ids_original : np.ndarray
        row_ids originales procedentes del H5.
    selected_indices : np.ndarray
        Índices conservados por el método.
    method_name : str
        Nombre del método aplicado.

    Returns
    -------
    pd.DataFrame
        DataFrame de trazabilidad.
    """
    keep_mask = np.zeros(len(metadata), dtype=bool)
    keep_mask[selected_indices] = True

    trace_df = metadata.copy()

    if "row_id" not in trace_df.columns:
        trace_df.insert(0, "row_id", row_ids_original)

    trace_df["label"] = y_original.astype(int)
    trace_df["decision"] = np.where(keep_mask, "keep", "drop")
    trace_df["method"] = method_name

    return trace_df


def save_resampled_h5(
    h5_path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    row_ids: np.ndarray,
) -> None:
    """
    Guarda el subconjunto filtrado en formato H5 manteniendo la misma estructura
    que los ficheros originales del proyecto:
    - 'features'
    - 'labels'
    - 'row_ids'

    Parameters
    ----------
    h5_path : Path
        Ruta de salida.
    features : np.ndarray
        Matriz de embeddings conservados.
    labels : np.ndarray
        Etiquetas conservadas.
    row_ids : np.ndarray
        Identificadores internos conservados.
    """
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("features", data=features)
        f.create_dataset("labels", data=labels)
        f.create_dataset("row_ids", data=row_ids)


def main() -> None:
    """
    Función principal del script.
    """
    args = parse_args()

    model = args.model
    n_clases = args.n_clases
    n_neighbors = args.n_neighbors
    threshold_cleaning = args.threshold_cleaning
    n_jobs = args.n_jobs

    print(f"[INFO] Modelo: {model}")
    print("[INFO] Método: NeighbourhoodCleaningRule")
    print(f"[INFO] n_neighbors: {n_neighbors}")
    print(f"[INFO] threshold_cleaning: {threshold_cleaning}")
    print(f"[INFO] n_jobs: {n_jobs}")

    # ---------------------------------------------------------------------
    # Rutas de entrada y salida.
    # ---------------------------------------------------------------------
    h5_path, metadata_path = get_input_paths(model)
    out_paths = get_output_paths(model)

    if not h5_path.exists():
        raise FileNotFoundError(f"No existe el fichero H5 de entrada: {h5_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"No existe el CSV de metadata: {metadata_path}")

    # ---------------------------------------------------------------------
    # Carga de embeddings y metadata.
    # ---------------------------------------------------------------------
    print("[INFO] Cargando embeddings y metadata...")
    X, y, row_ids = load_h5_data(h5_path)
    metadata = pd.read_csv(metadata_path)

    if len(metadata) != len(X):
        raise ValueError(
            "El número de filas de la metadata no coincide con el número de "
            f"embeddings. Metadata: {len(metadata)}, embeddings: {len(X)}"
        )

    try:
        y = y.astype(int)
    except ValueError as exc:
        raise ValueError(
            "Las etiquetas del dataset 'labels' no se pueden convertir a int."
        ) from exc

    print(f"[INFO] Número de patches originales: {len(X)}")
    print(f"[INFO] Dimensión del embedding: {X.shape[1]}")
    print("[INFO] Etiquetas cargadas desde el dataset 'labels' del H5")

    class_distribution_before = compute_class_distribution(y, n_clases)
    print(f"[INFO] Distribución antes: {class_distribution_before}")

    # ---------------------------------------------------------------------
    # Aplicación de NeighbourhoodCleaningRule.
    #
    # sample_indices_ nos da los índices conservados del conjunto original,
    # fundamentales para:
    # - filtrar metadata
    # - reconstruir row_ids
    # - generar trazabilidad completa
    # ---------------------------------------------------------------------
    ncr = NeighbourhoodCleaningRule(
        sampling_strategy="auto",
        n_neighbors=n_neighbors,
        threshold_cleaning=threshold_cleaning,
        n_jobs=n_jobs,
    )

    print("[INFO] Aplicando NeighbourhoodCleaningRule...")
    X_resampled, y_resampled = ncr.fit_resample(X, y)

    if hasattr(ncr, "classes_to_clean_"):
        print(
            "[INFO] Clases consideradas en la segunda fase de cleaning: "
            f"{ncr.classes_to_clean_}"
        )

    if not hasattr(ncr, "sample_indices_"):
        raise AttributeError(
            "NeighbourhoodCleaningRule no expuso el atributo sample_indices_. "
            "No es posible reconstruir la trazabilidad."
        )

    selected_indices = np.asarray(ncr.sample_indices_, dtype=int)
    row_ids_resampled = row_ids[selected_indices]

    # ---------------------------------------------------------------------
    # Filtrado de metadata con los mismos índices conservados.
    # ---------------------------------------------------------------------
    metadata_resampled = metadata.iloc[selected_indices].reset_index(drop=True)

    if "row_id" not in metadata_resampled.columns:
        metadata_resampled.insert(0, "row_id", row_ids_resampled)

    # ---------------------------------------------------------------------
    # Construcción de trazabilidad completa keep/drop.
    # ---------------------------------------------------------------------
    trace_df = build_trace_dataframe(
        metadata=metadata,
        y_original=y,
        row_ids_original=row_ids,
        selected_indices=selected_indices,
        method_name="ncr",
    )

    # ---------------------------------------------------------------------
    # Estadísticas finales.
    # ---------------------------------------------------------------------
    n_original = len(X)
    n_kept = len(X_resampled)
    n_removed = n_original - n_kept
    pct_removed = 100.0 * n_removed / n_original

    class_distribution_after = compute_class_distribution(y_resampled, n_clases)

    print(f"[INFO] Número de patches conservados: {n_kept}")
    print(f"[INFO] Número de patches eliminados: {n_removed}")
    print(f"[INFO] Porcentaje eliminado: {pct_removed:.2f}%")
    print(f"[INFO] Distribución después: {class_distribution_after}")

    # ---------------------------------------------------------------------
    # Guardado de artefactos.
    # ---------------------------------------------------------------------
    print("[INFO] Guardando artefactos...")

    save_resampled_h5(
        out_paths["h5_out"],
        features=X_resampled,
        labels=y_resampled,
        row_ids=row_ids_resampled,
    )
    metadata_resampled.to_csv(out_paths["metadata_out"], index=False)
    trace_df.to_csv(out_paths["trace_out"], index=False)

    summary = {
        "method": "ncr",
        "model": model,
        "split": "train",
        "n_original": int(n_original),
        "n_kept": int(n_kept),
        "n_removed": int(n_removed),
        "pct_removed": float(round(pct_removed, 4)),
        "embedding_dim": int(X.shape[1]),
        "label_column": "labels",
        "class_distribution_before": class_distribution_before,
        "class_distribution_after": class_distribution_after,
        "classes_to_clean": [int(c) for c in getattr(ncr, "classes_to_clean_", [])],
        "parameters": {
            "sampling_strategy": "auto",
            "n_neighbors": int(n_neighbors),
            "threshold_cleaning": float(threshold_cleaning),
            "n_jobs": int(n_jobs),
        },
    }

    with open(out_paths["summary_out"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"[INFO] H5 guardado en: {out_paths['h5_out']}")
    print(f"[INFO] Metadata guardada en: {out_paths['metadata_out']}")
    print(f"[INFO] Trace CSV guardado en: {out_paths['trace_out']}")
    print(f"[INFO] Summary JSON guardado en: {out_paths['summary_out']}")
    print("[INFO] NeighbourhoodCleaningRule aplicado correctamente.")


if __name__ == "__main__":
    main()