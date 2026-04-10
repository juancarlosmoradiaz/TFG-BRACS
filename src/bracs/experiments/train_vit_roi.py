# ---------------------------------------------
# ENTRENAMIENTO DE VISION TRANSFORMERS SOBRE PARCHES ROI (BRACS)
# ---------------------------------------------

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import timm
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm.auto import tqdm

from bracs.data.dataloaders import get_roi_train_val_dataloaders
from bracs.data.roi_dataset import get_class_names
from bracs.utils import paths
from bracs.utils.seed import set_seed


# =========================================================
# CONSTRUCCIÓN DEL MODELO
# =========================================================
def build_vit(model_name: str, n_clases: int, pretrained: bool = True) -> nn.Module:
    """
    Construimos un Vision Transformer con timm y adaptamos
    la cabeza de clasificación al número de clases.

    Args:
        model_name:
            Nombre del modelo en timm.
        n_clases:
            Número de clases del problema.
        pretrained:
            Si es True, cargamos pesos preentrenados.

    Returns:
        Modelo listo para entrenar.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=n_clases,
    )
    return model


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
    """
    Construimos el scheduler del learning rate.
    """
    sched_name = args.scheduler.lower()

    if sched_name == "none":
        return None

    if sched_name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )

    if sched_name == "step":
        return StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma,
        )

    raise ValueError(f"Scheduler no soportado: {args.scheduler}")


# =========================================================
# MÉTRICAS
# =========================================================
def compute_epoch_metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculamos métricas globales a partir de arrays de etiquetas reales y predichas.
    """
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }
    return metrics


@torch.no_grad()
def collect_preds_and_labels(
    model: nn.Module,
    dataloader,
    device: torch.device,
    desc: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recorremos un DataLoader completo y devolvemos:
        - y_true
        - y_pred
    """
    model.eval()

    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    iterator = tqdm(dataloader, desc=desc, leave=False) if desc else dataloader

    for imgs, labels, _paths in iterator:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
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
) -> Dict[str, float]:
    """
    Entrenamos una época y devolvemos:
        - train_loss
        - train_acc
        - train_f1_macro
        - train_f1_weighted
    """
    model.train()

    running_loss = 0.0
    total_samples = 0

    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    train_iter = tqdm(dataloader, desc=f"Train epoch {epoch_idx}/{num_epochs}", leave=False)

    for imgs, labels, _paths in train_iter:
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Pone a 0 los gradientes de todos los parámetros del modelo
        optimizer.zero_grad()

        # Hacemos una pasada hacia adelante:
        #   El modelo mira la imagen, da una respuesta (logits) y luego se le dice lo equivocado que estaba
        logits = model(imgs)
        loss = criterion(logits, labels)

        # Retropropagación: calcula cuánto contribuyó cada parámetro al error
        loss.backward()
        # Ajusta los parámetros del modelo para reducir el error
        optimizer.step()

        # Obtenemos las predicciones (la clase con mayor probabilidad para cada imagen)
        preds = torch.argmax(logits, dim=1)

        # Acumulamos la pérdida y el número de muestras
        running_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)

        # Guardamos las etiquetas reales y las predicciones en arrays de numpy en la CPU
        all_labels.append(labels.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())

        # Calculamos las métricas parciales (para ir viendo dinámicamente cómo va el entrenamiento)
        y_true_partial = np.concatenate(all_labels, axis=0)
        y_pred_partial = np.concatenate(all_preds, axis=0)
        partial_metrics = compute_epoch_metrics_from_arrays(y_true_partial, y_pred_partial)

        # Actualizamos la barra de tqdm 
        train_iter.set_postfix(
            loss=f"{running_loss / total_samples:.4f}",
            acc=f"{partial_metrics['acc']:.3f}",
            f1m=f"{partial_metrics['f1_macro']:.3f}",
        )

    # Calculamos la pérdida y las métricas finales de la época
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
) -> Dict[str, float]:
    """
    Evaluamos una época completa sobre validación. Mismo funcionamiento que train_one_epoch pero sin retropropagación.
    """
    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    val_iter = tqdm(
        dataloader,
        desc=f"Val   epoch {epoch_idx}/{num_epochs}",
        leave=False,
    )

    for imgs, labels, _paths in val_iter:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)

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
def save_confusion_matrix_figure(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str,
) -> None:
    """
    Guardamos una figura PNG de la matriz de confusión.
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
        description="Entrenamiento de Vision Transformers sobre patches RoI de BRACS."
    )

    # Dataset / tarea
    parser.add_argument("--n_clases", type=int, default=7)
    parser.add_argument("--dataset_name", type=str, default=None)

    # Modelo
    parser.add_argument(
        "--model",
        type=str,
        default="vit_base_patch16_224",
        choices=[
            "vit_base_patch16_224",
            "deit_small_patch16_224",
            "swin_tiny_patch4_window7_224",
        ],
    )
    parser.add_argument("--pretrained", type=int, default=1)

    # Seed
    parser.add_argument("--seed", type=int, required=True)

    # Entrenamiento
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Optimizador
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd"],
    )
    parser.add_argument("--momentum", type=float, default=0.9)

    # Scheduler
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "step"],
    )
    parser.add_argument("--scheduler_step_size", type=int, default=10)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)

    # Datos / transforms
    parser.add_argument("--tam_imagen", type=int, default=224)
    parser.add_argument(
        "--nivel_augmentation",
        type=str,
        default="none",
        choices=["none", "light", "heavy"],
    )
    parser.add_argument(
        "--tipo_normalizacion",
        type=str,
        default="none",
        choices=["none", "imagenet"],
    )
    parser.add_argument("--num_workers", type=int, default=4)

    # MLflow
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="roi-vit-7cls-baseline",
    )
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()



def main() -> None:
    args = parse_args()
    paths.ensure_dirs()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")
    print(f"[INFO] Seed del experimento: {args.seed}")

    print("[INFO] Construyendo DataLoaders de train/val ...")
    train_loader, val_loader = get_roi_train_val_dataloaders(
        n_clases=args.n_clases,
        batch_size=args.batch_size,
        tam_imagen=args.tam_imagen,
        nivel_augmentation=args.nivel_augmentation,
        tipo_normalizacion=args.tipo_normalizacion,
        num_workers=args.num_workers,
        dataset_name=args.dataset_name,
    )

    print(f"[INFO] Construyendo modelo ViT {args.model} ...")
    model = build_vit(
        model_name=args.model,
        n_clases=args.n_clases,
        pretrained=bool(args.pretrained),
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model.parameters(), args)
    scheduler = build_scheduler(optimizer, args)

    tracking_uri = f"file:{paths.mlflow_root()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    print(f"[INFO] MLflow tracking URI: {tracking_uri}")
    print(f"[INFO] MLflow experiment: {args.experiment_name}")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tag("family", "vit")
        mlflow.set_tag("phase", "baseline")
        mlflow.set_tag("task", f"{args.n_clases}cls")
        mlflow.set_tag("seed", args.seed)

        mlflow.log_params(
            {
                "family": "vit",
                "model": args.model,
                "n_clases": args.n_clases,
                "dataset_name": args.dataset_name or "default",
                "pretrained": bool(args.pretrained),
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
                "tam_imagen": args.tam_imagen,
                "nivel_augmentation": args.nivel_augmentation,
                "tipo_normalizacion": args.tipo_normalizacion,
                "num_workers": args.num_workers,
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
            models_root = paths.models_root() / "vit_roi"
            models_root.mkdir(parents=True, exist_ok=True)

            model_filename = (
                f"{args.model}_{args.n_clases}cls_seed{args.seed}_best_epoch{best_epoch}.pt"
            )
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

            figures_dir = paths.figures_root() / "vit_roi"
            results_dir = paths.results_root() / "vit_roi"
            figures_dir.mkdir(parents=True, exist_ok=True)
            results_dir.mkdir(parents=True, exist_ok=True)

            cm_train_png = figures_dir / f"cm_train_{args.model}_{args.n_clases}cls_seed{args.seed}.png"
            cm_val_png = figures_dir / f"cm_val_{args.model}_{args.n_clases}cls_seed{args.seed}.png"

            save_confusion_matrix_figure(
                cm=train_cm,
                class_names=class_names,
                save_path=cm_train_png,
                title=f"Train confusion matrix - {args.model} - seed {args.seed}",
            )
            save_confusion_matrix_figure(
                cm=val_cm,
                class_names=class_names,
                save_path=cm_val_png,
                title=f"Val confusion matrix - {args.model} - seed {args.seed}",
            )

            cm_train_csv = results_dir / f"cm_train_{args.model}_{args.n_clases}cls_seed{args.seed}.csv"
            cm_val_csv = results_dir / f"cm_val_{args.model}_{args.n_clases}cls_seed{args.seed}.csv"
            np.savetxt(cm_train_csv, train_cm, fmt="%d", delimiter=",")
            np.savetxt(cm_val_csv, val_cm, fmt="%d", delimiter=",")

            train_report_json = results_dir / f"report_train_{args.model}_{args.n_clases}cls_seed{args.seed}.json"
            val_report_json = results_dir / f"report_val_{args.model}_{args.n_clases}cls_seed{args.seed}.json"

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