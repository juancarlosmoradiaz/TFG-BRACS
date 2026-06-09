# ---------------------------------------------
# INSPECCIONAR LOS PATCHES RoI DEL DATASET BRACS:
#   - Leemos las rutas de los patches usando utils/paths.py
#   - Recorremos las carpetas train/val/test
#   - Contamos cuántas imágenes hay por split y por clase
#   - Mostramos un resumen por consola
# ---------------------------------------------

from pathlib import Path
from collections import Counter

from bracs.utils.paths import bracs_roi_patches_root


def inspect_split(split: str) -> None:
    """
    Inspeccionamos un split concreto (train, val o test).
    Contamos cuántos patches hay por clase.
    """
    root = bracs_roi_patches_root() / split

    if not root.exists():
        print(f"El directorio para el split '{split}' no existe: {root}")
        return

    print(f"\n=== Split: {split} ===")
    print(f"Directorio base: {root}")

    # Contador de patches por clase
    class_counter = Counter()
    total_files = 0

    # Recorremos las subcarpetas: train/0_N, train/1_PB, etc.
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name  # '0_N', '1_PB', etc.

        # Contamos todos los ficheros de imagen dentro de esa carpeta
        files = list(class_dir.rglob("*.jpeg")) + list(class_dir.rglob("*.png")) + list(class_dir.rglob("*.jpg"))

        n_files = len(files)
        class_counter[class_name] += n_files
        total_files += n_files

    # Mostramos el resumen
    print(f"Total de patches en {split}: {total_files}")
    for cls, count in sorted(class_counter.items()):
        print(f"  Clase {cls}: {count} patches")


def main():
    print(f"Raíz de patches RoI: {bracs_roi_patches_root()}")

    for split in ["train", "val", "test"]:
        inspect_split(split)


if __name__ == "__main__":
    main()