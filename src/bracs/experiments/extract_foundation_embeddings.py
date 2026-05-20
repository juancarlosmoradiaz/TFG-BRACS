# ---------------------------------------------
# EXTRACCIÓN DE EMBEDDINGS DE MODELOS FUNDACIONALES
# SOBRE PATCHES
# ---------------------------------------------
#
# Objetivo:
#   - Recorrer un split del dataset ("train", "val" y "test")
#   - Pasar todos los patches por un modelo fundacional congelado
#   - Extraer sus embeddings
#   - Guardarlos en un fichero .h5
#   - Guardar en paralelo un .csv con la trazabilidad de cada patch
#
# Salidas:
#   - outputs/embeddings/<model>_<split>.h5
#   - outputs/embeddings/<model>_<split>_metadata.csv
#
# Contenido del .h5:
#   - features : matriz N x D con los embeddings
#   - labels   : vector N con las etiquetas enteras
#   - row_ids  : vector N con identificadores internos consecutivos
#
# Contenido del .csv:
#   - row_id
#   - path
#   - label
#   - class_name
#   - split
#   - model_name
#
# Notas importantes:
#   - Aquí NO se entrena nada.
#   - El backbone se usa en modo evaluación.
#   - Se trabaja sin gradientes (torch.no_grad()).
#   - La limpieza posterior se hará SOLO sobre train, aunque
#   - En esta fase también podremos extraer val y test:
# ---------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import timm
import torch
from timm.layers import SwiGLUPacked
from torch import nn
from tqdm.auto import tqdm
from transformers import ViTModel

from bracs.data.roi_dataset import ROIPatchesDataset, get_class_names
from bracs.data.transforms import transformaciones_roi
from bracs.utils import paths


# =========================================================
# BACKBONES FUNDACIONALES
# =========================================================
class PhikonBackbone(nn.Module):
    """
    Wrapper para usar Phikon como extractor de características.

    Usamos el token CLS de la última capa oculta como embedding global.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = ViTModel.from_pretrained("owkin/phikon")
        self.out_dim = self.model.config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]  # [B, hidden]
        return cls_token


class TimmHFBackbone(nn.Module):
    """
    Wrapper genérico para modelos cargados con timm desde HF Hub.

    Para algunos modelos fundacionales no basta con crear el modelo de forma
    genérica, sino que hay que respetar exactamente ciertos componentes
    de arquitectura definidos en su model card.
    """

    def __init__(self, hf_hub_name: str, foundation_name: str) -> None:
        super().__init__()
        self.foundation_name = foundation_name

        if foundation_name == "virchow2":
            # Según la model card de Virchow2, hay que especificar
            # explícitamente la MLP y la activación correctas.
            self.model = timm.create_model(
                f"hf-hub:{hf_hub_name}",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
            # La card indica que el modelo devuelve embeddings de dimensión 2560.
            self.out_dim = 2560
        elif foundation_name == "h_optimus_1":
            self.model = timm.create_model(
                f"hf-hub:{hf_hub_name}",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=False,
                num_classes=0,
            )
            # La card indica que devuelve embeddings de dimensión 1536.
            self.out_dim = 1536
        else:
            # Caso genérico para otros modelos
            self.model = timm.create_model(
                f"hf-hub:{hf_hub_name}",
                pretrained=True,
                num_classes=0,
            )
            self.out_dim = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.foundation_name == "virchow2":
            # Salida esperada: [B, 261, 1280]
            # token 0 = CLS
            # tokens 1-4 = register tokens
            # tokens 5: = patch tokens
            output = self.model(x)
            class_token = output[:, 0]       # [B, 1280]
            patch_tokens = output[:, 5:]     # [B, 256, 1280]
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # [B, 2560]
            return embedding

        feats = self.model(x)
        return feats



def build_foundation_backbone_for_extraction(model_name: str) -> Tuple[nn.Module, int]:
    """
    Construye únicamente el backbone fundacional para extracción de embeddings.

    Entradas:
    - model_name: nombre del modelo ("h_optimus_1" o "virchow2")

    Salidas:
    - backbone: módulo PyTorch listo para usar
    - out_dim: dimensión del embedding esperado
    """
    model_name = model_name.lower()

    if model_name == "h_optimus_1":
        backbone = TimmHFBackbone(
            hf_hub_name="bioptimus/H-optimus-1",
            foundation_name="h_optimus_1",
        )
        return backbone, backbone.out_dim

    if model_name == "virchow2":
        backbone = TimmHFBackbone(
            hf_hub_name="paige-ai/Virchow2",
            foundation_name="virchow2",
        )
        return backbone, backbone.out_dim

    if model_name == "phikon":
        backbone = PhikonBackbone()
        return backbone, backbone.out_dim

    raise ValueError(f"Modelo fundacional no soportado para extracción: {model_name}")


# =========================================================
# ARGUMENTOS
# =========================================================
def parse_args() -> argparse.Namespace:
    """
        - model: modelo fundacional a usar
        - split: split a procesar (train, val o test)
        - n_clases: número de clases
        - dataset_name: por si queremos despúes usar otro dataset .pkl
        - batch_size: tamaño de lote para extracción
        - tam_imagen: tamaño al que se redimensionan las imágenes
        - tipo_normalizacion: normalización a aplicar
        - num_workers: workers del DataLoader
    """
    parser = argparse.ArgumentParser(
        description="Extracción de embeddings con modelos fundacionales sobre patches RoI de BRACS."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["h_optimus_1", "virchow2", "phikon"],
        help="Modelo fundacional del que se extraerán embeddings.",
    )

    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Split del dataset a procesar.",
    )

    parser.add_argument(
        "--n_clases",
        type=int,
        default=7,
        help="Número de clases del problema.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Nombre del dataset .pkl, si se quiere sobreescribir el dataset por defecto.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Tamaño de batch para la extracción.",
    )

    parser.add_argument(
        "--tam_imagen",
        type=int,
        default=224,
        help="Tamaño de las imágenes a la entrada del modelo.",
    )

    parser.add_argument(
        "--tipo_normalizacion",
        type=str,
        default="none",
        choices=["none", "imagenet"],
        help="Normalización a aplicar a las imágenes.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Número de workers del DataLoader.",
    )

    return parser.parse_args()


# =========================================================
# CONSTRUCCIÓN DEL DATASET / DATALOADER
# =========================================================
def build_dataloader_for_split(args: argparse.Namespace):
    """
    Construye el DataLoader del split indicado.
    """

    # transformaciones_roi devuelve un diccionario con transforms para:
    #   - train
    #   - val
    #   - test
    tfms = transformaciones_roi(
        tam_imagen=args.tam_imagen,
        nivel_augmentation="none",  # fijado por diseño en esta fase
        tipo_normalizacion=args.tipo_normalizacion,
    )

    dataset = ROIPatchesDataset(
        split=args.split,
        n_clases=args.n_clases,
        transform=tfms[args.split],
        dataset_name=args.dataset_name,
    )

    # No hacemos shuffle porque queremos que el orden sea determinista.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return dataset, dataloader


# =========================================================
# EXTRACCIÓN DE EMBEDDINGS
# =========================================================
@torch.no_grad()
def extract_embeddings(
    backbone: nn.Module,
    dataloader,
    device: torch.device,
    class_names: List[str],
    model_name: str,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Recorre el DataLoader completo y extrae embeddings de todos los patches.

    Entradas:
    - backbone: extractor fundacional
    - dataloader: DataLoader del split
    - device: dispositivo ("cuda" o "cpu")
    - class_names: nombres de clases para hacer el CSV más legible
    - model_name: nombre del modelo usado
    - split: nombre del split ("train" o "val")

    Salidas:
    - features: array numpy [N, D]
    - labels: array numpy [N]
    - row_ids: array numpy [N]
    - metadata_rows: lista de diccionarios, una por patch

    Se usa torch.no_grad() porque no entrenamos nada.
    Se generan row_ids internos consecutivos.
    """

    backbone.eval()

    all_features: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_row_ids: List[np.ndarray] = []
    metadata_rows: List[Dict[str, Any]] = []

    # row_counter mantiene el identificador interno de fila.
    # Cada patch tendrá un row_id único y consecutivo.
    row_counter = 0

    iterator = tqdm(
        dataloader,
        desc=f"Extrayendo embeddings [{model_name} - {split}]",
        leave=False,
    )

    for imgs, labels, paths_batch in iterator:
        # Movemos las imágenes al dispositivo donde está el backbone.
        imgs = imgs.to(device)

        # El backbone devuelve embeddings con forma [B, D].
        feats = backbone(imgs)

        # Pasamos embeddings y labels a CPU para poder almacenarlos en estructuras NumPy.
        feats_np = feats.cpu().numpy()
        labels_np = labels.numpy()

        batch_size = feats_np.shape[0]

        # Generamos row_ids consecutivos para este batch.
        row_ids_np = np.arange(row_counter, row_counter + batch_size, dtype=np.int64)

        # Guardamos embeddings, labels e ids del batch.
        all_features.append(feats_np)
        all_labels.append(labels_np)
        all_row_ids.append(row_ids_np)

        # Construimos la metadata fila a fila.
        for local_idx in range(batch_size):
            label_int = int(labels_np[local_idx])

            metadata_rows.append(
                {
                    "row_id": int(row_ids_np[local_idx]),
                    "path": str(paths_batch[local_idx]),
                    "file_name": Path(paths_batch[local_idx]).name,
                    "label": label_int,
                    "class_name": class_names[label_int],
                    "split": split,
                    "model_name": model_name,
                }
            )

        # Actualizamos el contador global de filas.
        row_counter += batch_size

    # Concatenamos todos los batches en arrays finales.
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    row_ids = np.concatenate(all_row_ids, axis=0)

    return features, labels, row_ids, metadata_rows


# =========================================================
# VALIDACIONES
# =========================================================
def validate_alignment(
    features: np.ndarray,
    labels: np.ndarray,
    row_ids: np.ndarray,
    metadata_rows: List[Dict[str, Any]],
    expected_out_dim: int,
) -> None:
    """
    Comprueba que todas las estructuras generadas están alineadas y son consistentes.
    """
    n_samples = features.shape[0]

    if n_samples != len(labels):
        raise ValueError(
            f"Desalineación: features tiene {n_samples} filas, pero labels tiene {len(labels)} elementos."
        )

    if n_samples != len(row_ids):
        raise ValueError(
            f"Desalineación: features tiene {n_samples} filas, pero row_ids tiene {len(row_ids)} elementos."
        )

    if n_samples != len(metadata_rows):
        raise ValueError(
            f"Desalineación: features tiene {n_samples} filas, pero metadata_rows tiene {len(metadata_rows)} elementos."
        )

    if features.ndim != 2:
        raise ValueError(
            f"Se esperaba una matriz 2D de embeddings, pero features tiene shape {features.shape}."
        )

    if features.shape[1] != expected_out_dim:
        raise ValueError(
            f"La dimensión del embedding no coincide con la esperada. "
            f"Esperada: {expected_out_dim}, obtenida: {features.shape[1]}"
        )

    unique_row_ids = np.unique(row_ids)
    if len(unique_row_ids) != len(row_ids):
        raise ValueError("Se han detectado row_ids duplicados.")

    expected_row_ids = np.arange(len(row_ids), dtype=np.int64)
    if not np.array_equal(row_ids, expected_row_ids):
        raise ValueError("Los row_ids no son consecutivos desde 0 hasta N-1.")

    # Comprobación adicional: que cada metadata_rows[i]["row_id"] coincida con row_ids[i]
    for i, row in enumerate(metadata_rows):
        if int(row["row_id"]) != int(row_ids[i]):
            raise ValueError(
                f"Desalineación entre metadata_rows y row_ids en la posición {i}: "
                f"metadata row_id={row['row_id']} vs row_ids={row_ids[i]}"
            )


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
    Guarda embeddings, labels y row_ids en un fichero HDF5.
    Se crean tres datasets:
    - features
    - labels
    - row_ids
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Compresión gzip para reducir el tamaño del fichero.
        f.create_dataset("features", data=features, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
        f.create_dataset("row_ids", data=row_ids, compression="gzip")


def save_metadata_csv(output_path: Path, metadata_rows: List[Dict[str, Any]]) -> None:
    """
    Guarda la metadata externa en formato CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metadata_rows)
    df.to_csv(output_path, index=False)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    args = parse_args()

    # Aseguramos que las carpetas base del proyecto existan.
    paths.ensure_dirs()

    # Seleccionamos GPU si está disponible; si no, CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Dispositivo: {device}")
    print(f"[INFO] Modelo: {args.model}")
    print(f"[INFO] Split: {args.split}")

    # Construimos dataset y dataloader del split solicitado.
    dataset, dataloader = build_dataloader_for_split(args)

    # Recuperamos los nombres de clase para enriquecer el CSV de metadata.
    class_names = get_class_names(args.n_clases)

    # Construimos el backbone fundacional y recuperamos la dimensión esperada.
    backbone, out_dim = build_foundation_backbone_for_extraction(args.model)
    backbone.to(device)
    backbone.eval()

    print(f"[INFO] Dimensión esperada del embedding: {out_dim}")
    print(f"[INFO] Número de patches en el split '{args.split}': {len(dataset)}")

    # Extraemos embeddings, labels, ids y metadata.
    features, labels, row_ids, metadata_rows = extract_embeddings(
        backbone=backbone,
        dataloader=dataloader,
        device=device,
        class_names=class_names,
        model_name=args.model,
        split=args.split,
    )

    # Validamos que todo esté perfectamente alineado antes de guardar.
    validate_alignment(
        features=features,
        labels=labels,
        row_ids=row_ids,
        metadata_rows=metadata_rows,
        expected_out_dim=out_dim,
    )

    # Definimos nombres de salida.
    embeddings_output_path = paths.project_root() / "outputs" / "embeddings" / f"{args.model}_{args.split}.h5"
    metadata_output_path = paths.project_root() / "outputs" / "embeddings" / f"{args.model}_{args.split}_metadata.csv"

    # Guardamos resultados.
    save_embeddings_h5(
        output_path=embeddings_output_path,
        features=features,
        labels=labels,
        row_ids=row_ids,
    )
    save_metadata_csv(
        output_path=metadata_output_path,
        metadata_rows=metadata_rows,
    )

    # Resumen final en consola.
    print("[INFO] Extracción finalizada correctamente.")
    print(f"[INFO] Embeddings extraídos: {features.shape[0]}")
    print(f"[INFO] Dimensión del embedding guardado: {features.shape[1]}")
    print(f"[INFO] Fichero H5 guardado en: {embeddings_output_path}")
    print(f"[INFO] Metadata CSV guardada en: {metadata_output_path}")


if __name__ == "__main__":
    main()