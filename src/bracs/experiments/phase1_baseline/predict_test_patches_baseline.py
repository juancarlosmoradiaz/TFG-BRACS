from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import timm
import torch
from timm.layers import SwiGLUPacked
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import ViTModel

from bracs.data.roi_dataset import ROIPatchesDataset, get_class_names
from bracs.data.transforms import transformaciones_roi


# =========================================================
# BACKBONES FUNDACIONALES
# =========================================================
class PhikonBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = ViTModel.from_pretrained("owkin/phikon")
        self.out_dim = self.model.config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        return outputs.last_hidden_state[:, 0]


class TimmHFBackbone(nn.Module):
    def __init__(self, hf_hub_name: str, foundation_name: str) -> None:
        super().__init__()
        self.foundation_name = foundation_name

        if foundation_name == "virchow2":
            self.model = timm.create_model(
                f"hf-hub:{hf_hub_name}",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
            self.out_dim = 2560
        elif foundation_name == "h_optimus_1":
            self.model = timm.create_model(
                f"hf-hub:{hf_hub_name}",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=False,
                num_classes=0,
            )
            self.out_dim = 1536
        else:
            self.model = timm.create_model(
                f"hf-hub:{hf_hub_name}",
                pretrained=True,
                num_classes=0,
            )
            self.out_dim = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.foundation_name == "virchow2":
            output = self.model(x)
            class_token = output[:, 0]
            patch_tokens = output[:, 5:]
            return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)

        return self.model(x)


class FoundationClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, out_dim: int, n_clases: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(out_dim, n_clases)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.classifier(feats)


def build_foundation_model(model_name: str, n_clases: int) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "phikon":
        backbone = PhikonBackbone()
        return FoundationClassifier(backbone, backbone.out_dim, n_clases, freeze_backbone=True)

    if model_name == "virchow2":
        backbone = TimmHFBackbone(
            hf_hub_name="paige-ai/Virchow2",
            foundation_name="virchow2",
        )
        return FoundationClassifier(backbone, backbone.out_dim, n_clases, freeze_backbone=True)

    if model_name == "h_optimus_1":
        backbone = TimmHFBackbone(
            hf_hub_name="bioptimus/H-optimus-1",
            foundation_name="h_optimus_1",
        )
        return FoundationClassifier(backbone, backbone.out_dim, n_clases, freeze_backbone=True)

    raise ValueError(f"Modelo fundacional no soportado: {model_name}")


# =========================================================
# AUXILIARES
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predicciones patch-level en test para baseline original."
    )
    parser.add_argument("--model", type=str, required=True, choices=["h_optimus_1", "virchow2", "phikon"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method_name", type=str, default="baseline")
    parser.add_argument("--n_clases", type=int, default=7)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tam_imagen", type=int, default=224)
    parser.add_argument("--tipo_normalizacion", type=str, default="none", choices=["none", "imagenet"])
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def extract_roi_id(file_name: str) -> str:
    stem = Path(file_name).stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"No se puede derivar roi_id desde: {file_name}")
    return "_".join(parts[:-1])


def build_test_dataloader(args: argparse.Namespace):
    tfms = transformaciones_roi(
        tam_imagen=args.tam_imagen,
        nivel_augmentation="none",
        tipo_normalizacion=args.tipo_normalizacion,
    )

    dataset = ROIPatchesDataset(
        split="test",
        n_clases=args.n_clases,
        transform=tfms["test"],
        dataset_name=args.dataset_name,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, dataloader


@torch.no_grad()
def predict_patch_level(
    model: nn.Module,
    dataloader,
    device: torch.device,
    class_names: List[str],
    model_name: str,
    n_clases: int,
) -> pd.DataFrame:
    model.eval()

    rows = []
    row_counter = 0

    iterator = tqdm(dataloader, desc=f"Predict test patches [{model_name}]", leave=False)

    for imgs, labels, paths_batch in iterator:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        probs_np = probs.cpu().numpy()
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        for i in range(len(paths_batch)):
            patch_path = str(paths_batch[i])
            file_name = Path(patch_path).name
            roi_id = extract_roi_id(file_name)
            label_int = int(labels_np[i])

            row = {
                "row_id": row_counter,
                "path": patch_path,
                "file_name": file_name,
                "roi_id": roi_id,
                "label": label_int,
                "class_name": class_names[label_int],
                "split": "test",
                "model_name": model_name,
                "y_true_patch": label_int,
                "y_pred_patch": int(preds_np[i]),
            }

            for c in range(n_clases):
                row[f"prob_{c}"] = float(probs_np[i, c])

            rows.append(row)
            row_counter += 1

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Modelo: {args.model}")
    print(f"[INFO] Método: {args.method_name}")
    print(f"[INFO] Dispositivo: {device}")

    print("[INFO] Construyendo DataLoader de test...")
    dataset, dataloader = build_test_dataloader(args)
    print(f"[INFO] Nº patches test: {len(dataset)}")

    print("[INFO] Construyendo modelo baseline...")
    model = build_foundation_model(
        model_name=args.model,
        n_clases=args.n_clases,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.to(device)
    model.eval()

    class_names = get_class_names(args.n_clases)

    print("[INFO] Generando predicciones patch-level...")
    df = predict_patch_level(
        model=model,
        dataloader=dataloader,
        device=device,
        class_names=class_names,
        model_name=args.model,
        n_clases=args.n_clases,
    )

    out_dir = Path("outputs/predictions/test_patches") / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.method_name}_patch_predictions.csv"
    df.to_csv(out_path, index=False)

    print("[INFO] Predicciones patch-level baseline generadas correctamente.")
    print(f"[INFO] CSV guardado en: {out_path}")


if __name__ == "__main__":
    main()