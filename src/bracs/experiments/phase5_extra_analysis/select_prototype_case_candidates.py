# ---------------------------------------------
# SELECCIÓN DE CANDIDATOS PARA CASOS DE ESTUDIO PROTOTÍPICOS
# ---------------------------------------------
# Entrada:
#   - Archivo CSV con prototipos más cercanos para las ROIs dudosas (salida de find_nearest_clear_prototypes.py)
#
# Salida:
#   - Archivos CSV con casos seleccionados para análisis cualitativo y visual:
#       * Casos donde el prototipo más cercano coincide con la clase real (anchored_true)
#       * Casos donde el prototipo más cercano coincide con la clase top-2 (anchored_top2)
#       * Casos entre parejas conflictivas concretas (ej. ADH -> FEA, ADH -> DCIS)
# ---------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CLASS_NAMES = {
    0: "N",
    1: "PB",
    2: "UDH",
    3: "FEA",
    4: "ADH",
    5: "DCIS",
    6: "IC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selecciona casos candidatos para el análisis visual de prototipos."
    )
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    return parser.parse_args()


def save_group(df: pd.DataFrame, path: Path, title: str, top_k: int) -> None:
    """
    Guarda y muestra en consola un subconjunto de candidatos preseleccionados.
    """
    out = df.head(top_k).copy()
    out.to_csv(path, index=False)
    print(f"\n=== {title} ===")
    if len(out) == 0:
        print("(sin casos)")
    else:
        print(out.to_string(index=False))


def main() -> None:
    args = parse_args()

    in_path = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Nos quedamos con el prototipo número 1 (el más cercano/vecino inmediato)
    rank1 = df[df["prototype_rank"] == 1].copy()

    rank1["y_true_name"] = rank1["y_true_dudosa"].map(CLASS_NAMES)
    rank1["top1_name"] = rank1["top1_class_dudosa"].map(CLASS_NAMES)
    rank1["top2_name"] = rank1["top2_class_dudosa"].map(CLASS_NAMES)

    rank1["nearest_is_true_class"] = rank1["prototype_class"] == rank1["y_true_dudosa"]
    rank1["nearest_is_top2_class"] = rank1["prototype_class"] == rank1["top2_class_dudosa"]

    # Ordenamos priorizando:
    # 1) menor margen (mayor incertidumbre)
    # 2) mayor similitud coseno (vecino más representativo)
    # 3) mayor probabilidad de la clase top-2 (mayor fuerza en la competencia)
    sort_cols = ["margin_top1_top2_dudosa", "cosine_similarity", "top2_prob_dudosa"]
    ascending = [True, False, False]

    # Casos donde el vecino prototípico coincide con la clase diagnóstica correcta
    anchored_true = rank1[rank1["nearest_is_true_class"]].copy()
    anchored_true = anchored_true.sort_values(by=sort_cols, ascending=ascending)

    # Casos donde el vecino prototípico apoya a la clase top-2 (falsa alarma del clasificador)
    anchored_top2 = rank1[rank1["nearest_is_top2_class"]].copy()
    anchored_top2 = anchored_top2.sort_values(by=sort_cols, ascending=ascending)

    # Parejas conflictivas específicas de transición
    interesting_pairs = [
        ("ADH", "FEA"),
        ("ADH", "DCIS"),
        ("UDH", "FEA"),
        ("PB", "N"),
        ("DCIS", "IC"),
    ]

    pair_frames = []
    for true_name, proto_name in interesting_pairs:
        sub = rank1[
            (rank1["y_true_name"] == true_name)
            & (rank1["prototype_class_name"] == proto_name)
        ].copy()
        sub = sub.sort_values(by=sort_cols, ascending=ascending)
        pair_frames.append((true_name, proto_name, sub))

    # Columnas seleccionadas para el reporte y guardado
    cols_show = [
        "roi_id_dudosa",
        "y_true_name",
        "top1_name",
        "top1_prob_dudosa",
        "top2_name",
        "top2_prob_dudosa",
        "margin_top1_top2_dudosa",
        "entropy_dudosa",
        "n_patches_dudosa",
        "prototype_roi_id",
        "prototype_class_name",
        "prototype_selection_level",
        "prototype_top1_prob",
        "prototype_margin",
        "prototype_n_patches",
        "cosine_similarity",
    ]

    anchored_true = anchored_true[cols_show]
    anchored_top2 = anchored_top2[cols_show]

    # Guardamos cada grupo y mostramos el resultado
    save_group(
        anchored_true,
        out_dir / "candidates_anchored_true.csv",
        "Casos cuyo prototipo más cercano coincide con la clase real",
        args.top_k,
    )

    save_group(
        anchored_top2,
        out_dir / "candidates_anchored_top2.csv",
        "Casos cuyo prototipo más cercano coincide con la clase top2",
        args.top_k,
    )

    for true_name, proto_name, sub in pair_frames:
        sub = sub[cols_show]
        filename = f"candidates_{true_name.lower()}_to_{proto_name.lower()}.csv"
        save_group(
            sub,
            out_dir / filename,
            f"Casos {true_name} -> {proto_name}",
            args.top_k,
        )

    print(f"\n[INFO] CSVs guardados en: {out_dir}")


if __name__ == "__main__":
    main()