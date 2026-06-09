# ---------------------------------------------
# ENTRENAMIENTO DE CABEZA LINEAL SOBRE EMBEDDINGS
#   - Carga embeddings ya extraídos
#   - Usar train limpio y val original
#   - Entrenar únicamente una capa lineal
#   - Evaluar igual que en el baseline fundacional
# ---------------------------------------------

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from bracs.data.roi_dataset import get_class_names
from bracs.utils import paths
from bracs.utils.seed import set_seed


# =========================================================
# DATASET DE EMBEDDINGS
# =========================================================
class EmbeddingDataset(Dataset):
    """
    Dataset sencillo para trabajar directamente con embeddings ya extraídos.

    Cada muestra devuelve:
        - embedding: tensor float32 de dimensión D
        - label: tensor long con la clase
        - row_id: identificador interno de fila

    Este dataset NO carga imágenes, sino vectores de características.
    """

    def __init__(self, h5_path: Path) -> None:
        super().__init__()

        if not h5_path.exists():
            raise FileNotFoundError(f"No existe el fichero H5: {h5_path}")

        # Abrimos el H5 una única vez y cargamos todo en memoria.
        # Esto simplifica mucho el entrenamiento y, para los tamaños con los que
        # estamos trabajando, es perfectamente razonable.
        with h5py.File(h5_path, "r") as f:
            self.features = f["features"][:].astype(np.float32)
            self.labels = f["labels"][:].astype(np.int64)
            self.row_ids = f["row_ids"][:].astype(np.int64)

        if self.features.ndim != 2:
            raise ValueError(f"features debe ser una matriz 2D. Shape recibido: {self.features.shape}")

        if len(self.labels) != len(self.features):
            raise ValueError("El número de labels no coincide con el número de embeddings.")

        if len(self.row_ids) != len(self.features):
            raise ValueError("El número de row_ids no coincide con el número de embeddings.")

        self.in_dim = self.features.shape[1]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = torch.from_numpy(self.features[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        row_id = torch.tensor(self.row_ids[idx], dtype=torch.long)
        return embedding, label, row_id


# =========================================================
# MODELO
# =========================================================
class LinearEmbeddingClassifier(nn.Module):
    """
    Clasificador lineal sobre embeddings.

    Entrada:
        - vector embedding de dimensión in_dim

    Salida:
        - logits de dimensión n_clases
    """

    def __init__(self, in_dim: int, n_clases: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_dim, n_clases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# =========================================================
# OPTIMIZADOR Y SCHEDULER
# =========================================================
def build_optimizer(params, args: argparse.Namespace):
    opt_name = args.optimizer.lower()

    if opt_name == "adamw":
        return AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name == "adam":
        return Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name == "sgd":
        return SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizador no soportado: {args.optimizer}")


def build_scheduler(optimizer, args: argparse.Namespace):
    sched_name = args.scheduler.lower()

    if sched_name == "none":
        return None
    elif sched_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif sched_name == "step":
        return StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    else:
        raise ValueError(f"Scheduler no soportado: {args.scheduler}")


# =========================================================
# MÉTRICAS
# =========================================================
def compute_epoch_metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


@torch.no_grad()
def collect_preds_and_labels(model: nn.Module, dataloader, device: torch.device, desc: str | None = None):
    """
    Recorre un DataLoader completo y devuelve:
        - y_true
        - y_pred
    """
    model.eval()

    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    iterator = tqdm(dataloader, desc=desc, leave=False) if desc else dataloader

    for embeddings, labels, _row_ids in iterator:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        logits = model(embeddings)
        preds = torch.argmax(logits, dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    return y_true, y_pred


# =========================================================
# ENTRENAMIENTO Y VALIDACIÓN POR ÉPOCA
# =========================================================
def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_idx: int,
    num_epochs: int,
):
    """
    Entrena una época y devuelve:
        - loss
        - acc
        - f1_macro
        - f1_weighted
    """
    model.train()

    running_loss = 0.0
    total_samples = 0

    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    train_iter = tqdm(dataloader, desc=f"Train epoch {epoch_idx}/{num_epochs}", leave=False)

    for embeddings, labels, _row_ids in train_iter:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(embeddings)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * embeddings.size(0)
        total_samples += embeddings.size(0)

        all_labels.append(labels.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())

        y_true_partial = np.concatenate(all_labels, axis=0)
        y_pred_partial = np.concatenate(all_preds, axis=0)
        partial_metrics = compute_epoch_metrics_from_arrays(y_true_partial, y_pred_partial)

        train_iter.set_postfix(
            loss=f"{running_loss / total_samples:.4f}",
            acc=f"{partial_metrics['acc']:.3f}",
            f1m=f"{partial_metrics['f1_macro']:.3f}",
        )

    epoch_loss = running_loss / total_samples
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    epoch_metrics = compute_epoch_metrics_from_arrays(y_true, y_pred)

    return {
        "loss": epoch_loss,
        "acc": epoch_metrics["acc"],
        "f1_macro": epoch_metrics["f1_macro"],
        "f1_weighted": epoch_metrics["f1_weighted"],
    }


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    epoch_idx: int,
    num_epochs: int,
):
    """
    Evalúa una época sobre validación.
    """
    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    val_iter = tqdm(dataloader, desc=f"Val   epoch {epoch_idx}/{num_epochs}", leave=False)

    for embeddings, labels, _row_ids in val_iter:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        logits = model(embeddings)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * embeddings.size(0)
        total_samples += embeddings.size(0)

        all_labels.append(labels.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())

        y_true_partial = np.concatenate(all_labels, axis=0)
        y_pred_partial = np.concatenate(all_preds, axis=0)
        partial_metrics = compute_epoch_metrics_from_arrays(y_true_partial, y_pred_partial)

        val_iter.set_postfix(
            loss=f"{running_loss / total_samples:.4f}",
            acc=f"{partial_metrics['acc']:.3f}",
            f1m=f"{partial_metrics['f1_macro']:.3f}",
        )

    epoch_loss = running_loss / total_samples
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    epoch_metrics = compute_epoch_metrics_from_arrays(y_true, y_pred)

    return {
        "loss": epoch_loss,
        "acc": epoch_metrics["acc"],
        "f1_macro": epoch_metrics["f1_macro"],
        "f1_weighted": epoch_metrics["f1_weighted"],
    }


# =========================================================
# MATRICES DE CONFUSIÓN
# =========================================================
def save_confusion_matrix_figure(cm: np.ndarray, class_names: List[str], save_path: Path, title: str) -> None:
    """
    Guarda una figura PNG de la matriz de confusión.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


# =========================================================
# ARGUMENTOS
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento de cabeza lineal sobre embeddings limpios/originales."
    )

    # Datos
    parser.add_argument("--model", type=str, required=True, choices=["h_optimus_1", "virchow2"])
    parser.add_argument("--n_clases", type=int, default=7)

    parser.add_argument("--train_h5", type=str, required=True)
    parser.add_argument("--val_h5", type=str, required=True)

    # Entrenamiento
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    # Hiperparámetros
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Optimizador
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "step"])
    parser.add_argument("--scheduler_step_size", type=int, default=10)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)

    # MLflow
    parser.add_argument("--experiment_name", type=str, default="embedding-linear-clean-7cls")
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    args = parse_args()
    paths.ensure_dirs()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")
    print(f"[INFO] Seed del experimento: {args.seed}")

    train_h5 = Path(args.train_h5)
    val_h5 = Path(args.val_h5)

    print("[INFO] Cargando datasets de embeddings ...")
    train_dataset = EmbeddingDataset(train_h5)
    val_dataset = EmbeddingDataset(val_h5)

    if train_dataset.in_dim != val_dataset.in_dim:
        raise ValueError(
            f"La dimensión del embedding no coincide entre train y val: "
            f"{train_dataset.in_dim} vs {val_dataset.in_dim}"
        )

    print(f"[INFO] Nº muestras train: {len(train_dataset)}")
    print(f"[INFO] Nº muestras val:   {len(val_dataset)}")
    print(f"[INFO] Dimensión embedding: {train_dataset.in_dim}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("[INFO] Construyendo clasificador lineal ...")
    model = LinearEmbeddingClassifier(
        in_dim=train_dataset.in_dim,
        n_clases=args.n_clases,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model.parameters(), args)
    scheduler = build_scheduler(optimizer, args)

    tracking_uri = f"file:{paths.mlflow_root()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    print(f"[INFO] MLflow tracking URI: {tracking_uri}")
    print(f"[INFO] MLflow experiment: {args.experiment_name}")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tag("family", "foundation_embeddings")
        mlflow.set_tag("phase", "clean_retrain")
        mlflow.set_tag("task", f"{args.n_clases}cls")
        mlflow.set_tag("seed", args.seed)

        mlflow.log_params(
            {
                "family": "foundation_embeddings",
                "model": args.model,
                "n_clases": args.n_clases,
                "train_h5": str(train_h5),
                "val_h5": str(val_h5),
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "optimizer": args.optimizer,
                "momentum": args.momentum,
                "scheduler": args.scheduler,
                "scheduler_step_size": args.scheduler_step_size,
                "scheduler_gamma": args.scheduler_gamma,
                "num_workers": args.num_workers,
                "embedding_dim": train_dataset.in_dim,
            }
        )

        best_val_acc = -1.0
        best_val_f1_macro = -1.0
        best_epoch = -1
        best_model_state = None

        global_start = time.time()

        for epoch in range(args.epochs):
            epoch_idx = epoch + 1
            print(f"\n===== Época {epoch_idx}/{args.epochs} =====")

            epoch_start = time.time()

            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch_idx=epoch_idx,
                num_epochs=args.epochs,
            )

            val_metrics = evaluate_one_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                epoch_idx=epoch_idx,
                num_epochs=args.epochs,
            )

            if scheduler is not None:
                scheduler.step()

            epoch_time = time.time() - epoch_start

            print(
                f"[TRAIN] loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['acc']:.4f} "
                f"f1_macro={train_metrics['f1_macro']:.4f} | "
                f"[VAL] loss={val_metrics['loss']:.4f} "
                f"acc={val_metrics['acc']:.4f} "
                f"f1_macro={val_metrics['f1_macro']:.4f} | "
                f"time={epoch_time:.1f}s"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["acc"],
                    "train_f1_macro": train_metrics["f1_macro"],
                    "train_f1_weighted": train_metrics["f1_weighted"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["acc"],
                    "val_f1_macro": val_metrics["f1_macro"],
                    "val_f1_weighted": val_metrics["f1_weighted"],
                    "epoch_time_sec": epoch_time,
                },
                step=epoch,
            )

            current_val_f1_macro = val_metrics["f1_macro"]
            current_val_acc = val_metrics["acc"]

            is_better = (
                (current_val_f1_macro > best_val_f1_macro)
                or (
                    np.isclose(current_val_f1_macro, best_val_f1_macro)
                    and current_val_acc > best_val_acc
                )
            )

            if is_better:
                best_val_f1_macro = current_val_f1_macro
                best_val_acc = current_val_acc
                best_epoch = epoch_idx
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

        total_training_time = time.time() - global_start

        print(f"\n[INFO] Mejor época: {best_epoch}")
        print(f"[INFO] best_val_f1_macro = {best_val_f1_macro:.4f}")
        print(f"[INFO] best_val_acc      = {best_val_acc:.4f}")
        print(f"[INFO] Tiempo total      = {total_training_time:.1f}s")

        mlflow.log_metric("best_val_f1_macro", best_val_f1_macro)
        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_metric("total_training_time_sec", total_training_time)
        mlflow.log_param("best_epoch", best_epoch)

        if best_model_state is not None:
            models_root = paths.models_root() / "linear_on_embeddings"
            models_root.mkdir(parents=True, exist_ok=True)

            model_filename = f"{args.model}_{args.n_clases}cls_seed{args.seed}_best_epoch{best_epoch}_linear.pt"
            model_path = models_root / model_filename

            torch.save(best_model_state, model_path)
            print(f"[INFO] Mejor modelo guardado en: {model_path}")
            mlflow.log_artifact(str(model_path))

        if best_model_state is not None:
            print("[INFO] Generando matrices de confusión y classification reports ...")

            model.load_state_dict(best_model_state)
            model.to(device)
            model.eval()

            class_names = get_class_names(args.n_clases)

            y_true_train, y_pred_train = collect_preds_and_labels(
                model=model,
                dataloader=train_loader,
                device=device,
                desc="Collect train preds",
            )
            train_report = classification_report(
                y_true_train,
                y_pred_train,
                target_names=class_names,
                output_dict=True,
            )
            train_cm = confusion_matrix(
                y_true_train,
                y_pred_train,
                labels=list(range(args.n_clases)),
            )

            y_true_val, y_pred_val = collect_preds_and_labels(
                model=model,
                dataloader=val_loader,
                device=device,
                desc="Collect val preds",
            )
            val_report = classification_report(
                y_true_val,
                y_pred_val,
                target_names=class_names,
                output_dict=True,
            )
            val_cm = confusion_matrix(
                y_true_val,
                y_pred_val,
                labels=list(range(args.n_clases)),
            )

            figures_dir = paths.figures_root() / "linear_on_embeddings"
            results_dir = paths.results_root() / "linear_on_embeddings"
            figures_dir.mkdir(parents=True, exist_ok=True)
            results_dir.mkdir(parents=True, exist_ok=True)

            cm_train_png = figures_dir / f"cm_train_{args.model}_{args.n_clases}cls_seed{args.seed}_linear.png"
            cm_val_png = figures_dir / f"cm_val_{args.model}_{args.n_clases}cls_seed{args.seed}_linear.png"

            save_confusion_matrix_figure(
                cm=train_cm,
                class_names=class_names,
                save_path=cm_train_png,
                title=f"Train confusion matrix - {args.model} - seed {args.seed} - linear",
            )
            save_confusion_matrix_figure(
                cm=val_cm,
                class_names=class_names,
                save_path=cm_val_png,
                title=f"Val confusion matrix - {args.model} - seed {args.seed} - linear",
            )

            cm_train_csv = results_dir / f"cm_train_{args.model}_{args.n_clases}cls_seed{args.seed}_linear.csv"
            cm_val_csv = results_dir / f"cm_val_{args.model}_{args.n_clases}cls_seed{args.seed}_linear.csv"
            np.savetxt(cm_train_csv, train_cm, fmt="%d", delimiter=",")
            np.savetxt(cm_val_csv, val_cm, fmt="%d", delimiter=",")

            train_report_json = results_dir / f"report_train_{args.model}_{args.n_clases}cls_seed{args.seed}_linear.json"
            val_report_json = results_dir / f"report_val_{args.model}_{args.n_clases}cls_seed{args.seed}_linear.json"

            with open(train_report_json, "w") as f:
                json.dump(train_report, f, indent=2)
            with open(val_report_json, "w") as f:
                json.dump(val_report, f, indent=2)

            for artifact_path in [
                cm_train_png,
                cm_val_png,
                cm_train_csv,
                cm_val_csv,
                train_report_json,
                val_report_json,
            ]:
                mlflow.log_artifact(str(artifact_path))

            mlflow.log_metric("train_f1_macro_final", float(train_report["macro avg"]["f1-score"]))
            mlflow.log_metric("train_f1_weighted_final", float(train_report["weighted avg"]["f1-score"]))
            mlflow.log_metric("val_f1_macro_final", float(val_report["macro avg"]["f1-score"]))
            mlflow.log_metric("val_f1_weighted_final", float(val_report["weighted avg"]["f1-score"]))

            print("[INFO] Matrices de confusión y reports subidos a MLflow.")

    print("\n[INFO] Entrenamiento finalizado correctamente.")


if __name__ == "__main__":
    main()