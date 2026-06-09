# ---------------------------------------------
# ANÁLISIS NO SUPERVISADO CON K-MEANS
# ---------------------------------------------
# Entrada:
#   - Archivo H5 con embeddings de parches de train
#   - Archivo CSV con metadata de parches de train
#
# Salida:
#   - Archivos CSV con proyecciones PCA y varianza explicada
#   - Archivos CSV con asignaciones de cluster e informes de cruce con la clase real
#   - Gráficos PNG con la varianza explicada de PCA y proyección coloreada por clase real y por cluster
#   - Mapa de calor en PNG que cruza la clase real con el cluster
# ---------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análisis no supervisado con PCA + KMeans sobre embeddings de train."
    )
    parser.add_argument("--model", type=str, required=True, choices=["virchow2", "h_optimus_1"])
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--k_values", type=int, nargs="+", default=[3, 5, 7, 10])
    parser.add_argument("--pca_components", type=int, default=50)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def ensure_dirs(base_out: Path) -> dict[str, Path]:
    """
    Asegura la creación de los subdirectorios de salida para PCA y KMeans.
    """
    pca_dir = base_out / "pca"
    kmeans_dir = base_out / "kmeans"
    pca_dir.mkdir(parents=True, exist_ok=True)
    kmeans_dir.mkdir(parents=True, exist_ok=True)
    return {"base": base_out, "pca": pca_dir, "kmeans": kmeans_dir}


def load_embeddings(h5_path: Path, metadata_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Carga los embeddings del archivo H5 y los alinea con su archivo de metadata.
    """
    with h5py.File(h5_path, "r") as f:
        features = f["features"][:]
        labels = f["labels"][:]
        row_ids = f["row_ids"][:]

    meta = pd.read_csv(metadata_csv)

    # Validamos consistencia formal
    if len(meta) != features.shape[0]:
        raise ValueError("La metadata no coincide en número de filas con los embeddings.")
    if len(labels) != features.shape[0]:
        raise ValueError("labels no coincide con features.")
    if len(row_ids) != features.shape[0]:
        raise ValueError("row_ids no coincide con features.")
    if not np.array_equal(meta["row_id"].to_numpy(), row_ids):
        raise ValueError("Los row_id del CSV no coinciden con los row_ids del H5.")
    if not np.array_equal(meta["label"].to_numpy(), labels):
        raise ValueError("Las labels del CSV no coinciden con las labels del H5.")

    return features, labels, row_ids, meta


def plot_pca_variance(explained_ratio: np.ndarray, model_name : str, save_path: Path) -> None:
    """
    Dibuja y guarda la curva de varianza explicada acumulada para los componentes de PCA.
    """
    cumulative = np.cumsum(explained_ratio)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o")
    plt.xlabel("Número de componentes PCA")
    plt.ylabel("Varianza explicada acumulada")
    plt.title(f"PCA: varianza explicada acumulada - {model_name}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pca_true_labels(pca_2d: np.ndarray, meta: pd.DataFrame, model_name: str , save_path: Path) -> None:
    """
    Dibuja la proyección PCA 2D coloreada en base a las etiquetas reales del dataset.
    """
    class_names = sorted(meta["class_name"].unique().tolist())

    plt.figure(figsize=(8.5, 6.5))
    for class_name in class_names:
        mask = meta["class_name"] == class_name
        plt.scatter(
            pca_2d[mask, 0],
            pca_2d[mask, 1],
            s=8,
            alpha=0.55,
            label=class_name,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA 2D coloreado por clase real - {model_name}")    
    plt.legend(markerscale=2, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pca_clusters(pca_2d: np.ndarray, clusters: np.ndarray, k: int, model_name: str, save_path: Path) -> None:    
    """
    Dibuja la proyección PCA 2D coloreada por la asignación resultante de clusters de K-Means.
    """
    plt.figure(figsize=(8.5, 6.5))
    for cluster_id in sorted(np.unique(clusters)):
        mask = clusters == cluster_id
        plt.scatter(
            pca_2d[mask, 0],
            pca_2d[mask, 1],
            s=8,
            alpha=0.55,
            label=f"Cluster {cluster_id}",
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA 2D coloreado por clusters de K-Means (k={k}) - {model_name}")
    plt.legend(markerscale=2, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(cluster_vs_class_pct: pd.DataFrame, k: int, model_name: str, save_path: Path) -> None:
    """
    Genera un mapa de calor que cruza la clase real y el cluster asignado.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(cluster_vs_class_pct.to_numpy(), aspect="auto")

    ax.set_xticks(np.arange(cluster_vs_class_pct.shape[1]))
    ax.set_yticks(np.arange(cluster_vs_class_pct.shape[0]))
    ax.set_xticklabels(cluster_vs_class_pct.columns.tolist(), rotation=45, ha="right")
    ax.set_yticklabels(cluster_vs_class_pct.index.tolist())
    ax.set_xlabel("Clase real")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Distribución porcentual clase real vs cluster (k={k}) - {model_name}")

    for i in range(cluster_vs_class_pct.shape[0]):
        for j in range(cluster_vs_class_pct.shape[1]):
            val = cluster_vs_class_pct.iloc[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()

    h5_path = Path(args.h5_path)
    metadata_csv = Path(args.metadata_csv)

    pretty_model_name = "Virchow2" if args.model == "virchow2" else "H-Optimus1"

    base_out = Path("outputs/unsupervised") / args.model
    dirs = ensure_dirs(base_out)

    print(f"[INFO] Modelo: {args.model}")
    print(f"[INFO] H5: {h5_path}")
    print(f"[INFO] Metadata: {metadata_csv}")

    features, labels, row_ids, meta = load_embeddings(h5_path, metadata_csv)

    print(f"[INFO] Nº patches: {features.shape[0]}")
    print(f"[INFO] Dimensión embedding: {features.shape[1]}")

    # Estandarizamos los datos antes del análisis dimensional
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Reducción de dimensiones con PCA
    pca = PCA(n_components=args.pca_components, random_state=args.random_state)
    X_pca = pca.fit_transform(X_scaled)
    explained_ratio = pca.explained_variance_ratio_
    cumulative_ratio = np.cumsum(explained_ratio)

    # Exportamos el resumen de varianza de PCA
    pca_variance_df = pd.DataFrame({
        "component": np.arange(1, len(explained_ratio) + 1),
        "explained_variance_ratio": explained_ratio,
        "cumulative_explained_variance_ratio": cumulative_ratio,
    })
    pca_variance_df.to_csv(dirs["pca"] / "pca_variance.csv", index=False)

    # Exportamos las proyecciones en las dos primeras componentes
    pca_2d_df = meta.copy()
    pca_2d_df["pc1"] = X_pca[:, 0]
    pca_2d_df["pc2"] = X_pca[:, 1]
    pca_2d_df.to_csv(dirs["pca"] / "pca_projection_2d.csv", index=False)

    # Graficamos la varianza explicada y la proyección coloreada por clase real
    plot_pca_variance(explained_ratio, pretty_model_name, dirs["pca"] / "pca_explained_variance.png")
    plot_pca_true_labels(X_pca[:, :2], meta, pretty_model_name, dirs["pca"] / "pca_true_labels_2d.png")

    print("[INFO] PCA calculado y guardado correctamente.")

    summary_rows = []

    # Iteramos ejecutando K-Means para los distintos valores de K configurados
    for k in args.k_values:
        print(f"[INFO] Ejecutando K-Means con k={k}...")

        kmeans = KMeans(
            n_clusters=k,
            random_state=args.random_state,
            n_init=20,
        )
        clusters = kmeans.fit_predict(X_pca)

        # Calculamos las métricas de calidad de clustering
        sil = silhouette_score(X_pca, clusters)
        ari = adjusted_rand_score(labels, clusters)
        nmi = normalized_mutual_info_score(labels, clusters)

        summary_rows.append({
            "model": args.model,
            "k": k,
            "silhouette": sil,
            "ari": ari,
            "nmi": nmi,
        })

        # Guardamos la asignación de clusters a parches
        assign_df = meta.copy()
        assign_df["cluster"] = clusters
        assign_df["pc1"] = X_pca[:, 0]
        assign_df["pc2"] = X_pca[:, 1]
        assign_df.to_csv(dirs["kmeans"] / f"k{k}_assignments.csv", index=False)

        # Matriz de cruzamiento: conteos absolutos y porcentajes normalizados
        cluster_vs_class = pd.crosstab(assign_df["cluster"], assign_df["class_name"])
        cluster_vs_class.to_csv(dirs["kmeans"] / f"k{k}_cluster_vs_class_counts.csv")

        cluster_vs_class_pct = cluster_vs_class.div(cluster_vs_class.sum(axis=1), axis=0) * 100.0
        cluster_vs_class_pct.to_csv(dirs["kmeans"] / f"k{k}_cluster_vs_class_pct.csv")

        # Dibujamos las proyecciones coloreadas por cluster y su mapa de calor cruzado
        plot_pca_clusters(X_pca[:, :2], clusters, k, pretty_model_name, dirs["kmeans"] / f"k{k}_pca_clusters.png")
        plot_heatmap(cluster_vs_class_pct, k, pretty_model_name, dirs["kmeans"] / f"k{k}_cluster_vs_class_heatmap.png")

        print(
            f"[INFO] k={k} | silhouette={sil:.4f} | ARI={ari:.4f} | NMI={nmi:.4f}"
        )

    # Exportamos el resumen global de KMeans
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(dirs["kmeans"] / "kmeans_summary.csv", index=False)

    print("[INFO] Análisis K-Means finalizado correctamente.")
    print(f"[INFO] Resultados guardados en: {base_out}")


if __name__ == "__main__":
    main()