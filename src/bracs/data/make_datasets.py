# ---------------------------------------------
# SCRIPT PARA CONSTRUIR LOS DATASETS DE PATCHES DE ROIs 
# - Leemos los patches ya generados (BRACS_RoI_patches_overlapping_512)
# - Construimos un diccionario con la estructura:
#     {
#         "meta": {...},
#         "splits": {
#             "train": {"x": np.array(...), "y": np.array(...)},
#             "val":   {...},
#             "test":  {...},
#         }
#     }
# - Podemos generar un dataset de:
#     - 7 clases
#     - 3 clases
# Guardamos el resultado tanto en .pkl como en .npy
# ---------------------------------------------

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from bracs.utils.paths import data_root, bracs_roi_patches_root


CLASS_NAMES_7 = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]

CLASS_NAMES_3 = ["AT", "BT", "MT"]

GROUPING_7_TO_3 = {
    "AT": ["FEA", "ADH"],
    "BT": ["N", "PB", "UDH"],
    "MT": ["DCIS", "IC"],
}


def build_label_mappings(n_classes: int):
    """
    Construimos los diccionarios de mapeo de etiquetas.

    Devolvemos:
        class_names: lista de nombres de clase en orden
        class_to_index: dict nombre -> índice
        index_to_class: dict índice -> nombre
        label7_to_label3: dict nombre_7cls -> nombre_3cls (o None si n_classes == 7)
    """
    if n_classes == 7:
        class_names = CLASS_NAMES_7
        label7_to_label3 = None
    elif n_classes == 3:
        class_names = CLASS_NAMES_3
        # Construimos un dict que nos diga, para cada clase 7cls, a qué grupo 3cls pertenece
        label7_to_label3 = {}
        for group_name, members in GROUPING_7_TO_3.items():
            for m in members:
                label7_to_label3[m] = group_name
    else:
        raise ValueError(f"n_classes debe ser 3 o 7, recibido: {n_classes}")

    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    index_to_class = {idx: name for name, idx in class_to_index.items()}

    return class_names, class_to_index, index_to_class, label7_to_label3


def infer_label_from_dirname(dirname: str) -> str:
    """
    A partir del nombre de la carpeta de clase, inferimos la etiqueta 7cls.

    En nuestros patches RoI, las carpetas son del estilo:
        '0_N', '1_PB', '2_UDH', '3_FEA', '4_ADH', '5_DCIS', '6_IC'

    Nosotros nos quedamos con la parte de la derecha (N, PB, UDH, FEA, ADH, DCIS, IC).
    """
    parts = dirname.split("_")
    return parts[-1]


def collect_split_data(split: str, n_classes: int, class_to_index: Dict[str, int],label7_to_label3: Dict[str, str] | None,) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recorremos los patches de un split (train / val / test) y devolvemos:
        x: array de rutas (str)
        y: array de índices de clase (int)
    """
    root_split = bracs_roi_patches_root() / split

    if not root_split.exists():
        raise FileNotFoundError(f"El directorio del split '{split}' no existe: {root_split}")

    paths_x: List[str] = []
    labels_y: List[int] = []

    # Lo usamos para hacer un pequeño resumen al final
    per_class_counter = defaultdict(int)

    for class_dir in sorted(root_split.iterdir()):
        if not class_dir.is_dir():
            continue

        # Nombre de la carpeta de clase, p.ej. '0_N'
        dirname = class_dir.name
        label_7 = infer_label_from_dirname(dirname)  # 'N', 'PB', ...

        if n_classes == 7:
            class_name = label_7
        else:
            # n_classes == 3: convertimos a AT/BT/MT
            if label7_to_label3 is None or label_7 not in label7_to_label3:
                raise ValueError(f"No sabemos agrupar la clase 7cls '{label_7}' a 3cls.")
            class_name = label7_to_label3[label_7]

        if class_name not in class_to_index:
            raise ValueError(f"Clase desconocida '{class_name}' para n_classes={n_classes}")

        class_idx = class_to_index[class_name]

        # Recorremos imágenes dentro de esa carpeta
        image_files = []
        image_files.extend(class_dir.rglob("*.jpeg"))
        image_files.extend(class_dir.rglob("*.jpg"))
        image_files.extend(class_dir.rglob("*.png"))

        for img_path in image_files:
            paths_x.append(str(img_path))
            labels_y.append(class_idx)
            per_class_counter[class_name] += 1

    x = np.array(paths_x, dtype=str)
    y = np.array(labels_y, dtype=int)

    print(f"\nResumen split '{split}' (n_classes={n_classes}):")
    print(f"  Total patches: {len(x)}")
    for cls_name in sorted(per_class_counter.keys()):
        print(f"  - {cls_name}: {per_class_counter[cls_name]} patches")

    return x, y


def main():
    parser = argparse.ArgumentParser(description="Construir dataset de patches RoI BRACS en formato .pkl/.npy")

    parser.add_argument(
        "--n_classes",
        type=int,
        choices=[3, 7],
        required=True,
        help="Número de clases del dataset (3 o 7).",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="Nombre base del fichero de salida (sin extensión). "
             "Se guardará en data/datasets/roi/<output_name>.pkl y .npy",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=512,
        help="Tamaño esperado de los patches (solo informativo, no validamos aún).",
    )

    args = parser.parse_args()

    n_classes = args.n_classes
    output_name = args.output_name
    patch_size = args.patch_size

    # Construimos mapeos de etiquetas
    class_names, class_to_index, index_to_class, label7_to_label3 = build_label_mappings(n_classes)

    print(f"  - n_clases: {n_classes}")
    print(f"  - class_names: {class_names}")
    print(f"  - raíz de patches RoI: {bracs_roi_patches_root()}")

    # Recolectamos datos por split
    splits_data = {}
    for split in ["train", "val", "test"]:
        x, y = collect_split_data(
            split=split,
            n_classes=n_classes,
            class_to_index=class_to_index,
            label7_to_label3=label7_to_label3,
        )
        splits_data[split] = {"x": x, "y": y}

    # Construimos el diccionario final
    dataset = {
        "meta": {
            "n_classes": n_classes,
            "class_names": class_names,
            "class_to_index": class_to_index,
            "index_to_class": index_to_class,
            "source_patches": str(bracs_roi_patches_root()),
            "patch_size": patch_size,
            "label_scheme": "7cls" if n_classes == 7 else "3cls_from_7cls",
            "version": "0.1",
        },
        "splits": splits_data,
    }

    # Directorio de salida: data/datasets/roi
    out_dir = data_root() / "datasets" / "roi"
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = out_dir / f"{output_name}.pkl"
    npy_path = out_dir / f"{output_name}.npy"

    # Guardamos en .pkl
    with open(pkl_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"\nDataset guardado en (pkl): {pkl_path}")

    # Guardamos también en .npy (por si queremos cargar solo con numpy)
    np.save(npy_path, dataset, allow_pickle=True)
    print(f"\nDataset guardado en (npy):  {npy_path}")



if __name__ == "__main__":
    main()