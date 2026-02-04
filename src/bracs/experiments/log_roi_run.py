import argparse
import pickle
from pathlib import Path

import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd

from bracs.utils import paths
from bracs.utils.mlflow_utils import start_run, log_common_tags


def _to_1d_labels(arr):
    """Convierte lista/array de etiquetas a vector 1D de ints."""
    arr = np.asarray(arr)
    if arr.ndim > 1:
        return arr.argmax(axis=1)
    return arr.astype(int)


def plot_confusion_matrix(cm, class_names):
    """Devuelve una figura de matplotlib con la matriz de confusión normalizada."""
    fig, ax = plt.subplots(figsize=(5, 5))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix (normalized)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Números dentro de cada celda
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Log de resultados ROI en MLflow (matriz de confusión, métricas, etc.)"
    )

    parser.add_argument(
        "--results_folder_name",
        type=str,
        required=True,
        help="Nombre de la carpeta dentro de outputs/results donde está results_Epoch_*.pkl",
    )
    parser.add_argument(
        "--data_RoI",
        type=str,
        required=True,
        help="Nombre del .pkl de datos ROI (solo para dejarlo registrado como param)",
    )
    parser.add_argument("--n_clases", type=int, required=True)
    parser.add_argument("--patch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--max_patches", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--data_augmentation", type=int, required=True)
    parser.add_argument("--weightsbyclass", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--lr_min", type=float, required=True)
    parser.add_argument("--warmup", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)

    args = parser.parse_args()

    # 1) Localizar carpeta de resultados (dentro de outputs)
        # 1) Localizar carpeta de resultados (probamos varias rutas posibles)
    candidates = [
        paths.outputs_root() / "results" / args.results_folder_name,
        paths.project_root() / "results" / args.results_folder_name,
        paths.outputs_root() / args.results_folder_name,
    ]

    results_dir = None
    for c in candidates:
        if c.exists():
            results_dir = c
            break

    if results_dir is None:
        raise FileNotFoundError(
            "La carpeta de resultados no existe en ninguna de las rutas probadas:\n"
            + "\n".join(str(c) for c in candidates)
        )

    # 2) Cargar el último results_Epoch*.pkl
    pkls = sorted(results_dir.glob("results_Epoch*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No se ha encontrado ningún results_Epoch*.pkl en {results_dir}")

    pkl_path = pkls[-1]
    print(f"Usando fichero de resultados: {pkl_path}")

    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    # 3) Extraer cosas importantes
    best_acc = float(results.get("best_acc", np.nan))
    best_auc = float(results.get("best_auc", np.nan))
    best_epoch = int(results.get("best_epoch", -1))

    val_labels_raw = results["val_labels"]
    val_preds_raw = results["val_preds"]
    val_probs_raw = results.get("val_probs", None)

    y_true = _to_1d_labels(val_labels_raw)
    y_pred = _to_1d_labels(val_preds_raw)

    # Probabilidades para AUC (si existen)
    y_proba = None
    if val_probs_raw is not None:
        y_proba = np.asarray(val_probs_raw)
        if y_proba.ndim == 1:
            y_proba = None

    # 4) Métricas de sklearn
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    target_names = [f"Class_{i}" for i in range(args.n_clases)]
    cls_report = classification_report(
        y_true, y_pred, target_names=target_names, digits=4
    )
    print("\nClassification report:\n", cls_report)

    # AUC multi-clase si tenemos probs
    auc_macro = None
    if y_proba is not None and args.n_clases > 1:
        try:
            y_true_ohe = np.eye(args.n_clases)[y_true]
            auc_macro = roc_auc_score(
                y_true_ohe, y_proba, multi_class="ovr", average="macro"
            )
            print(f"\nROC AUC macro (val): {auc_macro:.4f}")
        except Exception as e:
            print("No se pudo calcular AUC multi-clase:", e)

    # 5) Log en MLflow
    run_name = f"roi_{args.patch_size}_{args.n_clases}cls_{args.results_folder_name}"
    with start_run(run_name=run_name):
        # Tags comunes (proyecto, host, etc.)
        log_common_tags()

        # Hiperparámetros
        mlflow.log_param("results_folder_name", args.results_folder_name)
        mlflow.log_param("data_RoI", args.data_RoI)
        mlflow.log_param("n_clases", args.n_clases)
        mlflow.log_param("patch_size", args.patch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("max_patches", args.max_patches)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("data_augmentation", args.data_augmentation)
        mlflow.log_param("weightsbyclass", args.weightsbyclass)
        mlflow.log_param("dropout", args.dropout)
        mlflow.log_param("optimizer", args.optimizer)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("lr_min", args.lr_min)
        mlflow.log_param("warmup", args.warmup)
        mlflow.log_param("model", args.model)

        # Métricas "globales" del pickle
        mlflow.log_metric("best_val_acc", best_acc)
        mlflow.log_metric("best_val_auc_saved", best_auc)
        mlflow.log_metric("best_epoch", best_epoch)

        if auc_macro is not None:
            mlflow.log_metric("val_auc_macro_recomputed", auc_macro)

        # Curvas por época si están disponibles
        acc_array = results.get("acc_array", {})
        loss_array = results.get("loss_array", {})
        auc_array = results.get("auc_array", {})

        # acc_array, loss_array, auc_array suelen ser dicts con claves "train", "val"
        for split, values in acc_array.items():
            values = np.asarray(values).flatten()
            for epoch_idx, v in enumerate(values):
                mlflow.log_metric(f"acc_{split}", float(v), step=epoch_idx)

        for split, values in loss_array.items():
            values = np.asarray(values).flatten()
            for epoch_idx, v in enumerate(values):
                mlflow.log_metric(f"loss_{split}", float(v), step=epoch_idx)

        for split, values in auc_array.items():
            values = np.asarray(values).flatten()
            for epoch_idx, v in enumerate(values):
                mlflow.log_metric(f"auc_{split}", float(v), step=epoch_idx)

        # Matriz de confusión como imagen
        fig = plot_confusion_matrix(cm, target_names)
        mlflow.log_figure(fig, "confusion_matrix_val.png")
        plt.close(fig)

        # Matriz de confusión como tabla CSV
        cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
        cm_csv_path = results_dir / "confusion_matrix_val.csv"
        cm_df.to_csv(cm_csv_path)
        mlflow.log_artifact(str(cm_csv_path), artifact_path="artifacts")

        # Classification report como txt
        report_path = results_dir / "classification_report_val.txt"
        with open(report_path, "w") as f:
            f.write(cls_report)
        mlflow.log_artifact(str(report_path), artifact_path="artifacts")

        # Opcional: guardar también el pickle original como artifact
        mlflow.log_artifact(str(pkl_path), artifact_path="artifacts")

    print("\n✅ Resultados logueados en MLflow.")


if __name__ == "__main__":
    main()