# ---------------------------------------------
# ENTRENAMIENTO DE RESNET SOBRE PARCHES ROI 
# ---------------------------------------------
# Todo lo importante se configura por línea de comandos:
#   - nº de clases
#   - modelo (resnet18 / resnet50)
#   - optimizador (AdamW / Adam / SGD)
#   - scheduler (cosine / step / none)
#   - lr, weight_decay, etc.
#   - data augmentation y normalización
#
# Además, incluimos un modo opcional de LR finder:
#   --lr_find_only 1   -> ejecuta un LR range test y termina
# ---------------------------------------------

from __future__ import annotations

import argparse
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

import torch
from torch import nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from torchvision import models

import mlflow

from bracs.utils import paths
from bracs.data.dataloaders import get_roi_train_val_dataloaders

from tqdm.auto import tqdm

from pathlib import Path

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import json

from bracs.data.roi_dataset import get_class_names


# ------------------------------
# Construcción del modelo
# ------------------------------
def build_resnet(model_name: str, n_clases: int, pretrained: bool = True) -> nn.Module:
    """
    Construimos un modelo ResNet (18 o 50) y adaptamos la última capa (fc) para
    que tenga n_clases salidas.
    """
    model_name = model_name.lower()

    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Modelo no soportado todavía: {model_name}")

    # La capa final de ResNet es un Linear llamado "fc"
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_clases)

    return model


# ------------------------------
# Optimizador y scheduler
# ------------------------------
def build_optimizer(params, args: argparse.Namespace):
    """
    Construimos el optimizador a partir de los argumentos.
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
    Construimos el scheduler de LR según los argumentos.
        - none   -> sin scheduler
        - cosine -> CosineAnnealingLR con T_max = epochs
        - step   -> StepLR con step_size y gamma configurables
    """
    sched_name = args.scheduler.lower()

    if sched_name == "none":
        return None
    elif sched_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif sched_name == "step":
        return StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma,
        )
    else:
        raise ValueError(f"Scheduler no soportado: {args.scheduler}")


# ------------------------------
# Bucle de entrenamiento / evaluación
# ------------------------------
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
    Entrenamos una época completa sobre el dataloader de entrenamiento.

    Añadimos una barra de progreso con tqdm para ver el avance dentro de la época.
    """
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    # Barra de progreso sobre el dataloader de train
    train_iter = tqdm(
        dataloader,
        desc=f"Train epoch {epoch_idx}/{num_epochs}",
        leave=False,
    )

    for imgs, labels, _paths in train_iter:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)               # [B, n_clases]
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Métricas
        _, preds = torch.max(outputs, dim=1)
        running_loss += loss.item() * imgs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += imgs.size(0)

        # Actualizamos el texto de la barra con loss y acc parciales
        current_loss = running_loss / total_samples
        current_acc = running_corrects / total_samples
        train_iter.set_postfix(
            loss=f"{current_loss:.4f}",
            acc=f"{current_acc:.3f}",
        )

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return {"loss": epoch_loss, "acc": epoch_acc}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    epoch_idx: int,
    num_epochs: int,
) -> Dict[str, float]:
    """
    Evaluamos el modelo sobre el dataloader de validación.

    Devolvemos:
        - "loss"
        - "acc"

    Usamos también una barra de progreso con tqdm.
    """
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    val_iter = tqdm(
        dataloader,
        desc=f"Val   epoch {epoch_idx}/{num_epochs}",
        leave=False,
    )

    for imgs, labels, _paths in val_iter:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, dim=1)
        running_loss += loss.item() * imgs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += imgs.size(0)

        current_loss = running_loss / total_samples
        current_acc = running_corrects / total_samples
        val_iter.set_postfix(
            loss=f"{current_loss:.4f}",
            acc=f"{current_acc:.3f}",
        )

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return {"loss": epoch_loss, "acc": epoch_acc}


@torch.no_grad()
def collect_preds_and_labels(
    model: nn.Module,
    dataloader,
    device: torch.device,
):
    """
    Recorremos un dataloader completo y devolvemos:
        - y_true: np.ndarray con etiquetas reales (int)
        - y_pred: np.ndarray con predicciones (int)
    """
    model.eval()

    all_labels = []
    all_preds = []

    for imgs, labels, _paths in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)             # [B, n_clases]
        _, preds = torch.max(logits, 1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    return y_true, y_pred

def save_confusion_matrix_figure(
    cm: np.ndarray,
    class_names,
    save_path: Path,
    title: str,
) -> None:
    """
    Guardamos una figura PNG con la matriz de confusión.
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

    # Rotamos etiquetas del eje X para que quepan
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Escribimos los números dentro de cada celda
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
    
# ------------------------------
# LR Finder (LR range test)
# ------------------------------
def lr_range_test(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    """
    Hacemos un LR range test sencillo:
    - Recorremos unos cuantos batches de train (lr_find_steps).
    - Empezamos en lr_find_min_lr y terminamos en lr_find_max_lr
    """
    model.train()

    num_steps = min(args.lr_find_steps, len(train_loader))
    if num_steps < 2:
        raise ValueError("lr_find_steps demasiado pequeño o dataloader muy corto.")

    # Definimos un optimizador específico para el LR finder
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_find_min_lr,
        weight_decay=args.weight_decay,
    )

    min_lr = args.lr_find_min_lr
    max_lr = args.lr_find_max_lr

    lrs: List[float] = []
    losses: List[float] = []

    print(
        f"[LR FINDER] Empezamos range test: "
        f"min_lr={min_lr:.1e}, max_lr={max_lr:.1e}, steps={num_steps}"
    )

    step_idx = 0
    for imgs, labels, _paths in train_loader:
        if step_idx >= num_steps:
            break

        # LR exponencial entre min_lr y max_lr
        lr = min_lr * (max_lr / min_lr) ** (step_idx / (num_steps - 1))
        for g in optimizer.param_groups:
            g["lr"] = lr

        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        lrs.append(lr)
        losses.append(loss_val)

        # Log en MLflow: podremos ver la curva loss vs step,
        # y también el valor de lr en cada step.
        mlflow.log_metrics(
            {
                "lr_find_loss": loss_val,
                "lr_find_lr": lr,
            },
            step=step_idx,
        )

        if step_idx % 10 == 0:
            print(f"[LR FINDER] step={step_idx:03d}  lr={lr:.3e}  loss={loss_val:.4f}")

        step_idx += 1

    # Elegimos una lr sugerida:
    #   tomamos el índice del mínimo de la pérdida
    #   y retrocedemos unos pasos (por seguridad)
    idx_min = int(np.argmin(losses))
    idx_suggested = max(0, idx_min - 5)
    suggested_lr = lrs[idx_suggested]

    print("\n[LR FINDER] Resultados:")
    print(f"  - loss mínima: {losses[idx_min]:.4f} en step={idx_min}, lr={lrs[idx_min]:.3e}")
    print(f"  - lr sugerido (5 steps antes): {suggested_lr:.3e}")

    mlflow.log_param("lr_find_min_lr", min_lr)
    mlflow.log_param("lr_find_max_lr", max_lr)
    mlflow.log_param("lr_find_steps", num_steps)
    mlflow.log_param("lr_find_suggested_lr", suggested_lr)

    return suggested_lr


# ------------------------------
# Argumentos
# ------------------------------
def parse_args() -> argparse.Namespace:
    """
    Definimos los argumentos de línea de comandos para este script.
    """
    parser = argparse.ArgumentParser(
        description="Entrenamiento de ResNet sobre parches RoI (BRACS)."
    )

    # Datos / dataset
    parser.add_argument("--n_clases", type=int, default=7, help="Número de clases (3 o 7).")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Nombre del .pkl en data/datasets/roi (por defecto según n_clases).",
    )

    # Modelo
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Nombre del modelo backbone (resnet18, resnet50, ...).",
    )
    parser.add_argument(
        "--pretrained",
        type=int,
        default=1,
        help="Usar pesos preentrenados en ImageNet (1) o no (0).",
    )

    # Optimizador
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd"],
        help="Optimizador a usar.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum (solo se usa con SGD).",
    )

    # Entrenamiento
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas.")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de batch.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate inicial.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay para el optimizador."
    )

    # Scheduler
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "step"],
        help="Tipo de scheduler de LR.",
    )
    parser.add_argument(
        "--scheduler_step_size",
        type=int,
        default=10,
        help="Número de épocas entre pasos de StepLR (si scheduler='step').",
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.1,
        help="Factor de decaimiento de StepLR (si scheduler='step').",
    )

    # Transforms / augmentación
    parser.add_argument(
        "--tam_imagen",
        type=int,
        default=512,
        help="Tamaño de los parches (ya son 512, pero lo dejamos parametrizado).",
    )
    parser.add_argument(
        "--nivel_augmentation",
        type=str,
        default="light",
        choices=["none", "light", "heavy"],
        help="Nivel de data augmentation para train.",
    )
    parser.add_argument(
        "--tipo_normalizacion",
        type=str,
        default="imagenet",
        choices=["none", "imagenet"],
        help="Tipo de normalización para las imágenes.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Número de workers para los DataLoaders.",
    )

    # MLflow
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="roi-patches-resnet",
        help="Nombre del experimento de MLflow.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Nombre del run en MLflow (si None, MLflow pone uno por defecto).",
    )

    # LR finder
    parser.add_argument(
        "--lr_find_only",
        type=int,
        default=0,
        help="Si vale 1, ejecuta solo un LR range test y termina.",
    )
    parser.add_argument(
        "--lr_find_min_lr",
        type=float,
        default=1e-6,
        help="LR mínima para el LR range test.",
    )
    parser.add_argument(
        "--lr_find_max_lr",
        type=float,
        default=1e-2,
        help="LR máxima para el LR range test.",
    )
    parser.add_argument(
        "--lr_find_steps",
        type=int,
        default=100,
        help="Número de batches a usar en el LR range test.",
    )

    args = parser.parse_args()
    return args



def main() -> None:
    # Parseamos argumentos
    args = parse_args()

    # Aseguramos que existen las carpetas base (data, outputs, mlruns, ...)
    paths.ensure_dirs()

    # CPU o GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")

    # DataLoaders
    print("[INFO] Construyendo DataLoaders ...")
    train_loader, val_loader = get_roi_train_val_dataloaders(
        n_clases=args.n_clases,
        batch_size=args.batch_size,
        tam_imagen=args.tam_imagen,
        nivel_augmentation=args.nivel_augmentation,
        tipo_normalizacion=args.tipo_normalizacion,
        num_workers=args.num_workers,
        dataset_name=args.dataset_name,
    )

    # Modelo
    print(f"[INFO] Construyendo modelo {args.model} (n_clases={args.n_clases}) ...")
    model = build_resnet(
        model_name=args.model,
        n_clases=args.n_clases,
        pretrained=bool(args.pretrained),
    )
    model.to(device)

    # Criterio
    criterion = nn.CrossEntropyLoss()

    # MLflow
    tracking_uri = f"file:{paths.mlflow_root()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    print(f"[INFO] MLflow tracking URI: {tracking_uri}")
    print(f"[INFO] MLflow experiment: {args.experiment_name}")

    # Run de MLflow
    with mlflow.start_run(run_name=args.run_name):
        # Log de hiperparámetros generales
        mlflow.log_params(
            {
                "n_clases": args.n_clases,
                "dataset_name": args.dataset_name or "default",
                "model": args.model,
                "pretrained": bool(args.pretrained),
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
                "lr_find_only": args.lr_find_only,
                "lr_find_min_lr": args.lr_find_min_lr,
                "lr_find_max_lr": args.lr_find_max_lr,
                "lr_find_steps": args.lr_find_steps,
            }
        )

        # MODO LR FINDER SOLO
        if args.lr_find_only:
            print("[INFO] Modo LR finder activado (no se entrenará el modelo completo).")

            # Hacemos el range test con el modelo actual
            _ = lr_range_test(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                device=device,
                args=args,
            )

            print("[INFO] LR finder completado. Consulta MLflow para ver la curva.")
            return

        # Entrenamiento normal

        # Optimizador y scheduler
        optimizer = build_optimizer(model.parameters(), args)
        scheduler = build_scheduler(optimizer, args)

        best_val_acc = 0.0
        best_epoch = -1
        best_model_state = None

        for epoch in range(args.epochs):
            epoch_idx = epoch + 1
            print(f"\n===== Época {epoch_idx}/{args.epochs} =====")

            # Entrenamiento
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch_idx=epoch_idx,
                num_epochs=args.epochs,
            )

            # Validación
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                epoch_idx=epoch_idx,
                num_epochs=args.epochs,
            )

            # Scheduler (si existe)
            if scheduler is not None:
                scheduler.step()

            print(
                f"[TRAIN] loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} | "
                f"[VAL] loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f}"
            )

            # Log métricas
            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["acc"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["acc"],
                },
                step=epoch,
            )

            # Guardamos el mejor modelo según val_acc
            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                best_epoch = epoch
                best_model_state = model.state_dict()

        print(f"\n[INFO] Mejor época: {best_epoch}  |  best_val_acc={best_val_acc:.4f}")

        # Guardamos el mejor modelo y lo subimos a MLflow
        if best_model_state is not None:
            models_root = paths.models_root()
            models_root.mkdir(parents=True, exist_ok=True)

            model_filename = (
                f"resnet_roi_{args.n_clases}cls_{args.model}_"
                f"{args.optimizer}_best_epoch{best_epoch}.pt"
            )
            model_path = models_root / model_filename

            torch.save(best_model_state, model_path)
            print(f"[INFO] Mejor modelo guardado en: {model_path}")

            mlflow.log_artifact(str(model_path))

        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_param("best_epoch", best_epoch)
        
        # ======================================================
        #  EVALUACIÓN CON EL MEJOR MODELO (TRAIN + VAL)
        #  - Matrices de confusión
        #  - Classification report por clase
        # ======================================================
        if best_model_state is not None:
            print("[INFO] Evaluando best model en train y val para generar matrices de confusión ...")

            # Cargamos los pesos del mejor modelo
            model.load_state_dict(best_model_state)
            model.to(device)
            model.eval()

            class_names = get_class_names(args.n_clases)

            # ---------- TRAIN ----------
            y_true_train, y_pred_train = collect_preds_and_labels(
                model=model,
                dataloader=train_loader,
                device=device,
            )

            cm_train = confusion_matrix(
                y_true_train,
                y_pred_train,
                labels=list(range(args.n_clases)),
            )

            # Directorios para guardar resultados/figuras
            figs_dir = paths.figures_root() / "roi_resnet"
            results_dir = paths.results_root() / "roi_resnet"
            figs_dir.mkdir(parents=True, exist_ok=True)
            results_dir.mkdir(parents=True, exist_ok=True)

            cm_train_png = figs_dir / f"cm_train_{args.model}_{args.n_clases}cls.png"
            cm_train_csv = results_dir / f"cm_train_{args.model}_{args.n_clases}cls.csv"

            save_confusion_matrix_figure(
                cm=cm_train,
                class_names=class_names,
                save_path=cm_train_png,
                title=f"Confusion matrix - TRAIN ({args.model}, {args.n_clases} clases)",
            )
            np.savetxt(cm_train_csv, cm_train, fmt="%d", delimiter=",")

            mlflow.log_artifact(str(cm_train_png))
            mlflow.log_artifact(str(cm_train_csv))

            report_train = classification_report(
                y_true_train,
                y_pred_train,
                target_names=class_names,
                output_dict=True,
            )
            report_train_path = results_dir / f"classification_report_train_{args.model}_{args.n_clases}cls.json"
            with open(report_train_path, "w") as f:
                json.dump(report_train, f, indent=2)
            mlflow.log_artifact(str(report_train_path))

            # ---------- VAL ----------
            y_true_val, y_pred_val = collect_preds_and_labels(
                model=model,
                dataloader=val_loader,
                device=device,
            )

            cm_val = confusion_matrix(
                y_true_val,
                y_pred_val,
                labels=list(range(args.n_clases)),
            )

            cm_val_png = figs_dir / f"cm_val_{args.model}_{args.n_clases}cls.png"
            cm_val_csv = results_dir / f"cm_val_{args.model}_{args.n_clases}cls.csv"

            save_confusion_matrix_figure(
                cm=cm_val,
                class_names=class_names,
                save_path=cm_val_png,
                title=f"Confusion matrix - VAL ({args.model}, {args.n_clases} clases)",
            )
            np.savetxt(cm_val_csv, cm_val, fmt="%d", delimiter=",")

            mlflow.log_artifact(str(cm_val_png))
            mlflow.log_artifact(str(cm_val_csv))

            report_val = classification_report(
                y_true_val,
                y_pred_val,
                target_names=class_names,
                output_dict=True,
            )
            report_val_path = results_dir / f"classification_report_val_{args.model}_{args.n_clases}cls.json"
            with open(report_val_path, "w") as f:
                json.dump(report_val, f, indent=2)
            mlflow.log_artifact(str(report_val_path))

            # También logueamos F1 globales (macro y weighted)
            mlflow.log_metric("train_f1_macro", float(report_train["macro avg"]["f1-score"]))
            mlflow.log_metric("train_f1_weighted", float(report_train["weighted avg"]["f1-score"]))
            mlflow.log_metric("val_f1_macro_eval", float(report_val["macro avg"]["f1-score"]))
            mlflow.log_metric("val_f1_weighted_eval", float(report_val["weighted avg"]["f1-score"]))

            print("[INFO] Matrices de confusión y classification reports guardados y subidos a MLflow.")


if __name__ == "__main__":
    main()