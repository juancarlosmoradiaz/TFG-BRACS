# ---------------------------------------------
# ENTRENAMIENTO DE CNNs SOBRE PARCHES ROI 
# ---------------------------------------------
# Este script define el pipeline oficial de entrenamiento para
# modelos convolucionales (CNNs) sobre los parches.
#
# Objetivo:
#   - construir un benchmark limpio y reproducible
#   - trabajar solo con train/val
#   - NO usar test
#   - NO usar data augmentation
#   - NO usar normalización adicional
#   - comparar los 3 modelos bajo 5 semillas distintas
#
# Métricas principales:
#   - val_accuracy
#   - val_f1_macro
#
# Todo quedará registrado en MLflow

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchvision import models
from tqdm.auto import tqdm

from bracs.data.dataloaders import get_roi_train_val_dataloaders
from bracs.data.roi_dataset import get_class_names
from bracs.utils import paths
from bracs.utils.seed import set_seed


# =========================================================
# CONSTRUCCIÓN DEL MODELO
# =========================================================
def build_cnn(model_name: str, n_clases: int, pretrained: bool = True) -> nn.Module:
    """
    Construimos una CNN y adaptamos su última capa al número de clases.

    Modelos soportados:
        - resnet18
        - resnet50
        - densenet121

    Args:
        model_name:
            Nombre del backbone a utilizar.
        n_clases:
            Número de clases del problema (3 o 7).
        pretrained:
            Si es True, cargamos pesos preentrenados en ImageNet.

    Returns:
        model:
            Modelo listo para entrenar.
    """
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_clases)

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_clases)

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, n_clases)

    else:
        raise ValueError(f"Modelo CNN no soportado: {model_name}")

    return model


# =========================================================
# OPTIMIZADOR Y SCHEDULER
# =========================================================
def build_optimizer(params, args: argparse.Namespace):
    """
    Construimos el optimizador a partir de los argumentos del experimento.
    """
    opt_name = args.optimizer.lower()

    if opt_name == "adamw":
        optimizer = AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif opt_name == "adam":
        optimizer = Adam(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif opt_name == "sgd":
        optimizer = SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Optimizador no soportado: {args.optimizer}")

    return optimizer


def build_scheduler(optimizer, args: argparse.Namespace):
    """
    Construimos el scheduler de learning rate.

    Opciones:
        - none
        - cosine
        - step
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
# FUNCIONES ADICIONALES PARA LAS MÉTRICAS
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

    Esto nos permite calcular:
        - accuracy
        - F1 macro
        - matrices de confusión
        - classification reports
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

    train_iter = tqdm(
        dataloader,
        desc=f"Train epoch {epoch_idx}/{num_epochs}",
        leave=False,
    )

    for imgs, labels, _paths in train_iter:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(imgs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)

        all_labels.append(labels.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())

        # Métricas parciales mostradas durante la época en tiempo real
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
) -> Dict[str, float]:
    """
    Evaluamos una época completa sobre validación y devolvemos:
        - val_loss
        - val_acc
        - val_f1_macro
        - val_f1_weighted
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
# VISUALIZACIÓN Y GUARDADO DE MATRICES DE CONFUSIÓN
# =========================================================
def save_confusion_matrix_figure(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str,
) -> None:
    """
    Guardamos una imagen PNG de la matriz de confusión.
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
# ARGUMENTOS EN LA LÍNEA DE COMANDOS
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento de CNNs sobre patches RoI de BRACS."
    )

    # Dataset / tarea
    parser.add_argument("--n_clases", type=int, default=7, help="Número de clases (3 o 7).")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Nombre del .pkl dentro de data/datasets/roi.",
    )

    # Modelo
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50", "densenet121"],
        help="Modelo CNN a entrenar.",
    )
    parser.add_argument(
        "--pretrained",
        type=int,
        default=1,
        help="Usar pesos preentrenados en ImageNet (1) o no (0).",
    )

    # Semilla
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Semilla aleatoria del experimento.",
    )

    # Entrenamiento
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas.")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de batch.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay del optimizador.",
    )

    # Optimizador
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd"],
        help="Optimizador.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum para SGD.",
    )

    # Scheduler
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "step"],
        help="Scheduler del learning rate.",
    )
    parser.add_argument(
        "--scheduler_step_size",
        type=int,
        default=10,
        help="Step size de StepLR.",
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.1,
        help="Gamma de StepLR.",
    )

    # Datos / transformaciones
    parser.add_argument("--tam_imagen", type=int, default=512, help="Tamaño de entrada.")
    parser.add_argument(
        "--nivel_augmentation",
        type=str,
        default="none",
        choices=["none", "light", "heavy"],
        help="Nivel de augmentation. En baseline será none.",
    )
    parser.add_argument(
        "--tipo_normalizacion",
        type=str,
        default="none",
        choices=["none", "imagenet"],
        help="Tipo de normalización. En baseline será none.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Workers del DataLoader.",
    )

    # MLflow
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="roi-cnn-7cls-baseline",
        help="Nombre del experimento en MLflow.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Nombre del run en MLflow.",
    )

    return parser.parse_args()


# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================
def main() -> None:
    # Parseo y preparación inicial
    args = parse_args()
    paths.ensure_dirs()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")
    print(f"[INFO] Seed del experimento: {args.seed}")

    # DataLoaders
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

    # Modelo, loss, optimizer y scheduler
    print(f"[INFO] Construyendo modelo {args.model} ...")
    model = build_cnn(
        model_name=args.model,
        n_clases=args.n_clases,
        pretrained=bool(args.pretrained),
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model.parameters(), args)
    scheduler = build_scheduler(optimizer, args)

    # Configuración de MLflow
    tracking_uri = f"file:{paths.mlflow_root()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    print(f"[INFO] MLflow tracking URI: {tracking_uri}")
    print(f"[INFO] MLflow experiment: {args.experiment_name}")

    # Run de MLflow
    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tag("family", "cnn")
        mlflow.set_tag("phase", "baseline")
        mlflow.set_tag("task", f"{args.n_clases}cls")
        mlflow.set_tag("seed", args.seed)

        # Hiperparámetros / configuración
        mlflow.log_params(
            {
                "family": "cnn",
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

        # Bucle principal de entrenamiento
        best_val_acc = -1.0
        best_val_f1_macro = -1.0
        best_epoch = -1
        best_model_state = None

        train_history = []
        val_history = []

        global_start = time.time()

        for epoch in range(args.epochs):
            epoch_idx = epoch + 1
            print(f"\n===== Época {epoch_idx}/{args.epochs} =====")

            epoch_start = time.time()

            # Train
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch_idx=epoch_idx,
                num_epochs=args.epochs,
            )

            # Val 
            val_metrics = evaluate_one_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                epoch_idx=epoch_idx,
                num_epochs=args.epochs,
            )

            # Scheduler 
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

            # Log por época en MLflow
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

            train_history.append(train_metrics)
            val_history.append(val_metrics)

            # Guardado del mejor modelo 
            # Priorizamos F1 macro. Si empata, usamos accuracy
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

        # Guardado del mejor checkpoint
        if best_model_state is not None:
            models_root = paths.models_root() / "cnn_roi"
            models_root.mkdir(parents=True, exist_ok=True)

            model_filename = (
                f"{args.model}_{args.n_clases}cls_seed{args.seed}_best_epoch{best_epoch}.pt"
            )
            model_path = models_root / model_filename

            torch.save(best_model_state, model_path)
            print(f"[INFO] Mejor modelo guardado en: {model_path}")
            mlflow.log_artifact(str(model_path))

        # Evaluación final con best model
        if best_model_state is not None:
            print("[INFO] Generando matrices de confusión y classification reports ...")

            model.load_state_dict(best_model_state)
            model.to(device)
            model.eval()

            class_names = get_class_names(args.n_clases)

            # TRAIN 
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

            # VAL 
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

            # Guardado en disco
            figures_dir = paths.figures_root() / "cnn_roi"
            results_dir = paths.results_root() / "cnn_roi"
            figures_dir.mkdir(parents=True, exist_ok=True)
            results_dir.mkdir(parents=True, exist_ok=True)

            # Matrices de confusión train y val
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

            # CSVs de matrices
            cm_train_csv = results_dir / f"cm_train_{args.model}_{args.n_clases}cls_seed{args.seed}.csv"
            cm_val_csv = results_dir / f"cm_val_{args.model}_{args.n_clases}cls_seed{args.seed}.csv"
            np.savetxt(cm_train_csv, train_cm, fmt="%d", delimiter=",")
            np.savetxt(cm_val_csv, val_cm, fmt="%d", delimiter=",")

            # Reports JSON
            train_report_json = results_dir / f"report_train_{args.model}_{args.n_clases}cls_seed{args.seed}.json"
            val_report_json = results_dir / f"report_val_{args.model}_{args.n_clases}cls_seed{args.seed}.json"

            with open(train_report_json, "w") as f:
                json.dump(train_report, f, indent=2)
            with open(val_report_json, "w") as f:
                json.dump(val_report, f, indent=2)

            # Log de artifacts en MLflow
            for artifact_path in [
                cm_train_png,
                cm_val_png,
                cm_train_csv,
                cm_val_csv,
                train_report_json,
                val_report_json,
            ]:
                mlflow.log_artifact(str(artifact_path))

            # Log extra de métricas finales
            mlflow.log_metric("train_f1_macro_final", float(train_report["macro avg"]["f1-score"]))
            mlflow.log_metric("train_f1_weighted_final", float(train_report["weighted avg"]["f1-score"]))
            mlflow.log_metric("val_f1_macro_final", float(val_report["macro avg"]["f1-score"]))
            mlflow.log_metric("val_f1_weighted_final", float(val_report["weighted avg"]["f1-score"]))

            print("[INFO] Matrices de confusión y reports subidos a MLflow.")

    print("\n[INFO] Entrenamiento finalizado correctamente.")


if __name__ == "__main__":
    main()