# ---------------------------------------------
# LIMPIEZA DE EMBEDDINGS MEDIANTE INFORMACIÓN MUTUA
# ---------------------------------------------
#   - Este script SIEMPRE trabaja sobre train.
#   - Se usa PCA como paso previo para hacer el cálculo más tractable.

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

# =========================================================
# ARGUMENTOS
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Limpieza de embeddings de train mediante información mutua y comparación local."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["h_optimus_1", "virchow2"],
        help="Modelo fundacional cuyos embeddings train se van a limpiar.",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Si se indica, se limita el número de muestras cargadas del train.",
    )

    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="Semilla para el muestreo aleatorio cuando se usa --max_samples. Permite que el subconjunto sea aleatorio.",
    )

    parser.add_argument(
        "--n_clases",
        type=int,
        default=7,
        help="Número de clases del problema.",
    )

    parser.add_argument(
        "--pca_components",
        type=int,
        default=64,
        help="Número de componentes principales para reducir embeddings antes del cálculo de MI.",
    )

    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=6,
        help="Número K de vecinos para la comparación local.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Umbral de discrepancia entre score de un patch y el de sus vecinos.",
    )

    parser.add_argument(
        "--count_threshold",
        type=int,
        default=None,
        help=(
            "Umbral de conteo de discrepancias para eliminar un patch. "
            "Si no se indica, se usa ceil(0.5 * K)."
        ),
    )

    parser.add_argument(
        "--distance_metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Métrica para calcular vecinos en el espacio PCA.",
    )

    parser.add_argument(
        "--save_neighbors",
        action="store_true",
        help="Si se activa, se guarda la matriz de vecinos en disco.",
    )

    return parser.parse_args()


# =========================================================
# CARGA Y VALIDACIÓN DE DATOS
# =========================================================
def load_embeddings_and_metadata(model_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Carga los embeddings del split train y su metadata asociada.

    Entrada:
    - model_name: nombre del modelo ("h_optimus_1" o "virchow2")

    Salidas:
    - features: matriz N x D de embeddings
    - labels: vector N de etiquetas enteras
    - row_ids: vector N de identificadores internos
    - metadata_df: DataFrame con trazabilidad patch a patch
    """

    h5_path = Path("outputs") / "embeddings" / f"{model_name}_train.h5"
    csv_path = Path("outputs") / "embeddings" / f"{model_name}_train_metadata.csv"

    if not h5_path.exists():
        raise FileNotFoundError(f"No existe el fichero H5 de embeddings: {h5_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el fichero CSV de metadata: {csv_path}")

    with h5py.File(h5_path, "r") as f:
        features = f["features"][:]
        labels = f["labels"][:]
        row_ids = f["row_ids"][:]

    metadata_df = pd.read_csv(csv_path)

    return features, labels, row_ids, metadata_df


def validate_alignment(
    features: np.ndarray,
    labels: np.ndarray,
    row_ids: np.ndarray,
    metadata_df: pd.DataFrame,
) -> None:
    """
    Comprueba que embeddings, labels, row_ids y metadata estén perfectamente alineados.
    """
    n = features.shape[0]

    if features.ndim != 2:
        raise ValueError(f"features debe ser una matriz 2D. Shape recibido: {features.shape}")

    if len(labels) != n:
        raise ValueError(f"labels tiene {len(labels)} elementos y features tiene {n} filas.")

    if len(row_ids) != n:
        raise ValueError(f"row_ids tiene {len(row_ids)} elementos y features tiene {n} filas.")

    if len(metadata_df) != n:
        raise ValueError(f"metadata_df tiene {len(metadata_df)} filas y features tiene {n} filas.")

    if "row_id" not in metadata_df.columns:
        raise ValueError("El CSV de metadata no contiene la columna 'row_id'.")

    # Comprobamos que los row_ids del H5 sean únicos.
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("Se han detectado row_ids duplicados en el H5.")

    # Comprobamos que sean consecutivos 0..N-1
    expected_row_ids = np.arange(n, dtype=np.int64)
    if not np.array_equal(row_ids, expected_row_ids):
        raise ValueError("Los row_ids del H5 no son consecutivos desde 0 hasta N-1.")

    # Comprobamos coincidencia exacta con el CSV
    csv_row_ids = metadata_df["row_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(row_ids, csv_row_ids):
        raise ValueError("Los row_ids del H5 y del CSV no coinciden exactamente.")


# =========================================================
# REDUCCIÓN DE DIMENSIONALIDAD
# =========================================================
def reduce_embeddings_with_pca(features: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """
    Reduce la dimensionalidad de los embeddings mediante PCA.
    """

    n_samples, n_dims = features.shape
    
    # Si n_components es mayor que min(N, D), sklearn fallará.
    # Por eso aquí lo ajustamos al máximo valor permitido.
    max_allowed = min(n_samples, n_dims)

    if n_components > max_allowed:
        raise ValueError(
            f"pca_components={n_components} es demasiado grande para una matriz de shape {features.shape}. "
            f"Debe ser <= {max_allowed}."
        )

    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(features)
    return reduced, pca


# =========================================================
# CÁLCULO DE VECINOS
# =========================================================
def compute_neighbors(
    features_reduced: np.ndarray,
    k_neighbors: int,
    distance_metric: str,
) -> np.ndarray:
    """
    Calcula los K vecinos más cercanos de cada patch en el espacio PCA.

    Entrada:
    - features_reduced: matriz N x d'
    - k_neighbors: número K de vecinos
    - distance_metric: "euclidean" o "cosine"

    Salida:
    - neighbors: matriz N x K con índices de vecinos
    """

    n = features_reduced.shape[0]

    if k_neighbors <= 0:
        raise ValueError("k_neighbors debe ser mayor que 0.")

    if k_neighbors >= n:
        raise ValueError(
            f"k_neighbors={k_neighbors} es inválido para N={n}. Debe ser menor que el número de muestras."
        )

    # Pedimos K+1 vecinos porque el primero será el propio punto.
    # Después eliminamos esa primera columna y nos quedamos con K vecinos reales.
    nn_model = NearestNeighbors(
        n_neighbors=k_neighbors + 1,
        metric=distance_metric,
    )
    nn_model.fit(features_reduced)

    # indices tendrá shape [N, K+1]
    # La primera columna corresponde al propio punto con distancia 0.
    _, indices = nn_model.kneighbors(features_reduced)

    neighbors = indices[:, 1:]

    # Comprobació para asegurar que ningún punto aparece como vecino de sí mismo.
    self_idx = np.arange(n).reshape(-1, 1)
    if np.any(neighbors == self_idx):
        raise ValueError("Se ha detectado un punto que aparece como vecino de sí mismo.")

    return neighbors


# =========================================================
# CÁLCULO DE LA INFORMACIÓN MUTUA
# =========================================================
def estimate_mutual_information(
    features_reduced: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Estima la información mutua total entre embeddings reducidos y clase.

    Entrada:
    - features_reduced: matriz N x d'
    - labels: vector N

    Salida:
    - mi_total: información mutua total
    """
    
    mi_per_feature = mutual_info_classif(
        X=features_reduced,
        y=labels,
        discrete_features=False,
        random_state=42,
    )

    mi_total = float(np.sum(mi_per_feature))
    return mi_total


def compute_leave_one_out_mi_scores(
    features_reduced: np.ndarray,
    labels: np.ndarray,
    mi_full: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Para cada patch i:
    - elimina ese patch
    - recalcula la MI del conjunto sin él
    - calcula delta_information = mi_full - mi_without_patch

    Entradas:
    - features_reduced: matriz N x d'
    - labels: vector N
    - mi_full: MI global del conjunto completo

    Salidas:
    - mi_without_patch: vector N
    - delta_information: vector N
    """
    n = features_reduced.shape[0]

    mi_without_patch = np.zeros(n, dtype=np.float64)
    delta_information = np.zeros(n, dtype=np.float64)

    iterator = tqdm(
        range(n),
        desc="Calculando MI leave-one-out",
        leave=False,
    )

    for i in iterator:
        # Creamos una máscara booleana que deja fuera el patch i
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        X_minus_i = features_reduced[mask]
        y_minus_i = labels[mask]

        mi_minus_i = estimate_mutual_information(X_minus_i, y_minus_i)

        mi_without_patch[i] = mi_minus_i
        delta_information[i] = mi_full - mi_minus_i

    return mi_without_patch, delta_information


# =========================================================
# COMPARACIÓN LOCAL
# =========================================================
def compute_neighbor_disagreement_counts(
    delta_information: np.ndarray,
    neighbors: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Para cada patch i:
    - compara su delta_information con la de cada uno de sus K vecinos
    - cuenta cuántas diferencias absolutas superan alpha

    Entradas:
    - delta_information: vector N con los scores informativos
    - neighbors: matriz N x K con índices de vecinos
    - alpha: umbral de discrepancia

    Salida:
    - disagreement_counts: vector N
    """
    n = len(delta_information)
    disagreement_counts = np.zeros(n, dtype=np.int64)

    for i in range(n):
        patch_score = delta_information[i]
        neighbor_indices = neighbors[i]

        count = 0
        for j in neighbor_indices:
            diff = abs(patch_score - delta_information[j])
            if diff > alpha:
                count += 1

        disagreement_counts[i] = count

    return disagreement_counts


# =========================================================
# DECISIÓN MANTENER / ELIMINAR
# =========================================================
def decide_keep_drop(
    disagreement_counts: np.ndarray,
    count_threshold: int,
) -> np.ndarray:
    """
    Decide qué patches se conservan y cuáles se eliminan.

    - keep = 1 si disagreement_count < count_threshold
    - keep = 0 si disagreement_count >= count_threshold

    Salida:
    - keep_mask: vector booleano de tamaño N
    """
    keep_mask = disagreement_counts < count_threshold
    return keep_mask


# =========================================================
# CONSTRUCCIÓN DEL DATASET LIMPIO
# =========================================================
def build_clean_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    row_ids: np.ndarray,
    metadata_df: pd.DataFrame,
    keep_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Aplica la máscara keep/drop del método anterior y construye el dataset limpio.

    Entradas:
    - features: embeddings originales
    - labels: etiquetas originales
    - row_ids: ids originales
    - metadata_df: metadata original
    - keep_mask: máscara booleana de patches a conservar

    Salidas:
    - features_clean
    - labels_clean
    - row_ids_clean
    - metadata_clean_df
    """
    features_clean = features[keep_mask]
    labels_clean = labels[keep_mask]
    row_ids_clean = row_ids[keep_mask]

    metadata_clean_df = metadata_df.loc[keep_mask].reset_index(drop=True)

    return features_clean, labels_clean, row_ids_clean, metadata_clean_df


# =========================================================
# RESUMEN DE LIMPIEZA
# =========================================================
def build_cleaning_summary(
    labels_original: np.ndarray,
    labels_clean: np.ndarray,
) -> Dict:
    """
    Construye un resumen del proceso de limpieza.

    Se incluyen:
    - número original de patches
    - número conservado
    - número eliminado
    - porcentaje eliminado
    - distribución por clase antes/después
    """
    n_original = int(len(labels_original))
    n_clean = int(len(labels_clean))
    n_removed = n_original - n_clean
    pct_removed = 100.0 * n_removed / n_original if n_original > 0 else 0.0

    before_counts = pd.Series(labels_original).value_counts().sort_index().to_dict()
    after_counts = pd.Series(labels_clean).value_counts().sort_index().to_dict()

    summary = {
        "n_original": n_original,
        "n_clean": n_clean,
        "n_removed": n_removed,
        "pct_removed": pct_removed,
        "class_distribution_before": {str(k): int(v) for k, v in before_counts.items()},
        "class_distribution_after": {str(k): int(v) for k, v in after_counts.items()},
    }
    return summary


# =========================================================
# GUARDADO DE RESULTADOS
# =========================================================
def save_embeddings_h5(
    output_path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    row_ids: np.ndarray,
) -> None:
    """
    Guarda el dataset limpio en HDF5.

    Datasets guardados:
    - features
    - labels
    - row_ids
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("features", data=features, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
        f.create_dataset("row_ids", data=row_ids, compression="gzip")


def save_cleaning_outputs(
    model_name: str,
    metadata_df: pd.DataFrame,
    mi_full: float,
    mi_without_patch: np.ndarray,
    delta_information: np.ndarray,
    disagreement_counts: np.ndarray,
    keep_mask: np.ndarray,
    args: argparse.Namespace,
    features_clean: np.ndarray,
    labels_clean: np.ndarray,
    row_ids_clean: np.ndarray,
    metadata_clean_df: pd.DataFrame,
    summary: Dict,
    neighbors: np.ndarray | None = None,
) -> None:
    """
    Guarda todos los artefactos producidos por el limpiador.

    Artefactos:
    - CSV completo de cleaning
    - H5 limpio
    - metadata limpia
    - summary JSON
    - vecinos opcionales
    """
    cleaning_dir = Path("outputs") / "cleaning"
    cleaned_embeddings_dir = Path("outputs") / "cleaned_embeddings"
    intermediate_dir = cleaning_dir / "intermediate"

    cleaning_dir.mkdir(parents=True, exist_ok=True)
    cleaned_embeddings_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------
    # 1) CSV completo de cleaning
    # -----------------------------------------
    cleaning_df = metadata_df.copy()

    # Añadimos columnas numéricas del proceso.
    cleaning_df["mi_full"] = mi_full
    cleaning_df["mi_without_patch"] = mi_without_patch
    cleaning_df["delta_information"] = delta_information
    cleaning_df["neighbor_disagreement_count"] = disagreement_counts
    cleaning_df["k_neighbors"] = args.k_neighbors
    cleaning_df["alpha"] = args.alpha
    cleaning_df["count_threshold"] = args.count_threshold
    cleaning_df["distance_metric"] = args.distance_metric
    cleaning_df["keep"] = keep_mask.astype(int)

    cleaning_csv_path = cleaning_dir / f"{model_name}_train_cleaning.csv"
    cleaning_df.to_csv(cleaning_csv_path, index=False)

    # -----------------------------------------
    # 2) Dataset limpio en H5
    # -----------------------------------------
    clean_h5_path = cleaned_embeddings_dir / f"{model_name}_train_clean.h5"
    save_embeddings_h5(
        output_path=clean_h5_path,
        features=features_clean,
        labels=labels_clean,
        row_ids=row_ids_clean,
    )

    # -----------------------------------------
    # 3) Metadata limpia
    # -----------------------------------------
    clean_metadata_csv_path = cleaned_embeddings_dir / f"{model_name}_train_clean_metadata.csv"
    metadata_clean_df.to_csv(clean_metadata_csv_path, index=False)

    # -----------------------------------------
    # 4) Summary JSON
    # -----------------------------------------
    summary_path = cleaning_dir / f"{model_name}_train_cleaning_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # -----------------------------------------
    # 5) Vecinos opcionalmente
    # -----------------------------------------
    if neighbors is not None:
        neighbors_path = intermediate_dir / f"{model_name}_train_neighbors.npy"
        np.save(neighbors_path, neighbors)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    
    args = parse_args()

    # Si el usuario no da count_threshold, usamos por defecto ceil(0.5 * K)
    if args.count_threshold is None:
        args.count_threshold = math.ceil(0.5 * args.k_neighbors)

    print(f"[INFO] Modelo: {args.model}")
    print("[INFO] Split objetivo: train")
    print(f"[INFO] PCA components: {args.pca_components}")
    print(f"[INFO] K vecinos: {args.k_neighbors}")
    print(f"[INFO] alpha: {args.alpha}")
    print(f"[INFO] count_threshold: {args.count_threshold}")
    print(f"[INFO] distance_metric: {args.distance_metric}")

    # -----------------------------------------
    # 1) Carga de embeddings y metadata
    # -----------------------------------------
    features, labels, row_ids, metadata_df = load_embeddings_and_metadata(args.model)

    # -------------------------------------------------
    # MODO TEST: muestreo aleatorio reproducible
    # -------------------------------------------------
    # Si max_samples no es None, en vez de coger las primeras N muestras,
    # cogemos N patches aleatorios del train, para que el subconjunto sea representativo y no solo de una misma clase
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max_samples debe ser mayor que 0.")

        max_samples = min(args.max_samples, features.shape[0])

        rng = np.random.default_rng(args.sample_seed)
        sampled_indices = rng.choice(features.shape[0], size=max_samples, replace=False)

        # Ordenamos los índices seleccionados para que el resultado final
        # quede estable y fácil de inspeccionar.
        sampled_indices = np.sort(sampled_indices)

        features = features[sampled_indices]
        labels = labels[sampled_indices]
        row_ids = row_ids[sampled_indices]
        metadata_df = metadata_df.iloc[sampled_indices].copy().reset_index(drop=True)

        print(
            f"[INFO] Modo test activado: usando {max_samples} muestras "
            f"seleccionadas aleatoriamente con sample_seed={args.sample_seed}."
        )

    # Reajustamos row_ids tras el posible muestreo para asegurar que:
    # - sean consecutivos desde 0 hasta N-1
    # - coincidan exactamente con el CSV de metadata
    row_ids = np.arange(len(row_ids), dtype=np.int64)
    metadata_df["row_id"] = row_ids
        
    # -----------------------------------------
    # 2) Validación de alineación
    # -----------------------------------------
    validate_alignment(features, labels, row_ids, metadata_df)

    print(f"[INFO] Número de patches originales: {features.shape[0]}")
    print(f"[INFO] Dimensión original del embedding: {features.shape[1]}")

    # -----------------------------------------
    # 3) Reducción de dimensionalidad con PCA
    # -----------------------------------------
    features_reduced, _pca = reduce_embeddings_with_pca(
        features=features,
        n_components=args.pca_components,
    )

    print(f"[INFO] Dimensión tras PCA: {features_reduced.shape[1]}")

    # -----------------------------------------
    # 4) Cálculo de vecinos
    # -----------------------------------------
    neighbors = compute_neighbors(
        features_reduced=features_reduced,
        k_neighbors=args.k_neighbors,
        distance_metric=args.distance_metric,
    )

    # -----------------------------------------
    # 5) Información mutua global
    # -----------------------------------------
    mi_full = estimate_mutual_information(
        features_reduced=features_reduced,
        labels=labels,
    )
    print(f"[INFO] MI global del conjunto completo: {mi_full:.6f}")

    # -----------------------------------------
    # 6) Cálculo leave-one-out
    # -----------------------------------------
    mi_without_patch, delta_information = compute_leave_one_out_mi_scores(
        features_reduced=features_reduced,
        labels=labels,
        mi_full=mi_full,
    )

    # -----------------------------------------
    # 7) Conteo de discrepancias locales
    # -----------------------------------------
    disagreement_counts = compute_neighbor_disagreement_counts(
        delta_information=delta_information,
        neighbors=neighbors,
        alpha=args.alpha,
    )

    # -----------------------------------------
    # 8) Decisión keep/drop
    # -----------------------------------------
    keep_mask = decide_keep_drop(
        disagreement_counts=disagreement_counts,
        count_threshold=args.count_threshold,
    )

    n_original = len(keep_mask)
    n_clean = int(np.sum(keep_mask))
    n_removed = n_original - n_clean

    if n_clean == 0:
        raise ValueError(
            "La limpieza ha eliminado todos los patches. "
            "Revisa K, alpha o count_threshold."
        )

    # -----------------------------------------
    # 9) Construcción del dataset limpio
    # -----------------------------------------
    features_clean, labels_clean, row_ids_clean, metadata_clean_df = build_clean_dataset(
        features=features,
        labels=labels,
        row_ids=row_ids,
        metadata_df=metadata_df,
        keep_mask=keep_mask,
    )

    # -----------------------------------------
    # 10) Resumen
    # -----------------------------------------
    summary = build_cleaning_summary(
        labels_original=labels,
        labels_clean=labels_clean,
    )

    # -----------------------------------------
    # 11) Guardado
    # -----------------------------------------
    save_cleaning_outputs(
        model_name=args.model,
        metadata_df=metadata_df,
        mi_full=mi_full,
        mi_without_patch=mi_without_patch,
        delta_information=delta_information,
        disagreement_counts=disagreement_counts,
        keep_mask=keep_mask,
        args=args,
        features_clean=features_clean,
        labels_clean=labels_clean,
        row_ids_clean=row_ids_clean,
        metadata_clean_df=metadata_clean_df,
        summary=summary,
        neighbors=neighbors if args.save_neighbors else None,
    )

    # -----------------------------------------
    # 12) Resumen final en consola
    # -----------------------------------------
    pct_removed = 100.0 * n_removed / n_original if n_original > 0 else 0.0

    print("[INFO] Limpieza finalizada correctamente.")
    print(f"[INFO] Patches originales: {n_original}")
    print(f"[INFO] Patches conservados: {n_clean}")
    print(f"[INFO] Patches eliminados: {n_removed}")
    print(f"[INFO] Porcentaje eliminado: {pct_removed:.2f}%")
    print(f"[INFO] CSV cleaning guardado en: outputs/cleaning/{args.model}_train_cleaning.csv")
    print(f"[INFO] H5 limpio guardado en: outputs/cleaned_embeddings/{args.model}_train_clean.h5")
    print(
        "[INFO] Metadata limpia guardada en: "
        f"outputs/cleaned_embeddings/{args.model}_train_clean_metadata.csv"
    )
    print(
        "[INFO] Summary guardado en: "
        f"outputs/cleaning/{args.model}_train_cleaning_summary.json"
    )
    if args.save_neighbors:
        print(
            "[INFO] Vecinos guardados en: "
            f"outputs/cleaning/intermediate/{args.model}_train_neighbors.npy"
        )


if __name__ == "__main__":
    main()