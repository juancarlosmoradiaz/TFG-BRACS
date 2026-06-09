# ---------------------------------------------
# RESUMEN DE VECINOS PROTOTÍPICOS
# ---------------------------------------------
# Entrada:
#   - Archivo CSV con los prototipos más cercanos para las ROIs dudosas (salida de find_nearest_clear_prototypes.py)
#
# Salida:
#   - Archivos CSV resumen de vecinos en la carpeta de destino:
#       * global_summary.csv: Totales globales de coincidencia de vecinos con clase real/top-2
#       * summary_by_true_class.csv: Agrupación y porcentajes por clase real
#       * nearest_pair_counts.csv: Frecuencias de parejas (clase real -> clase prototipo)
#       * nearest_matches_top2_cases.csv: Listado de casos que coinciden con la clase top-2
#       * nearest_matches_true_cases.csv: Listado de casos que coinciden con la clase real
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
        description="Resume vecinos prototípicos de ROIs dudosas."
    )
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Nos quedamos con el prototipo más cercano (inmediato superior rank 1)
    rank1 = df[df["prototype_rank"] == 1].copy()

    rank1["y_true_name"] = rank1["y_true_dudosa"].map(CLASS_NAMES)
    rank1["top1_name"] = rank1["top1_class_dudosa"].map(CLASS_NAMES)
    rank1["top2_name"] = rank1["top2_class_dudosa"].map(CLASS_NAMES)

    rank1["nearest_is_true_class"] = rank1["prototype_class"] == rank1["y_true_dudosa"]
    rank1["nearest_is_top2_class"] = rank1["prototype_class"] == rank1["top2_class_dudosa"]

    # Resumen global del comportamiento de los vecinos
    global_summary = pd.DataFrame([{
        "n_rois_dudosas": int(rank1["roi_id_dudosa"].nunique()),
        "pct_nearest_is_true_class": 100.0 * rank1["nearest_is_true_class"].mean(),
        "pct_nearest_is_top2_class": 100.0 * rank1["nearest_is_top2_class"].mean(),
    }])
    global_summary.to_csv(out_dir / "global_summary.csv", index=False)

    # Resumen desglosado por clase real
    by_true = (
        rank1.groupby(["y_true_dudosa", "y_true_name"])
        .agg(
            n_cases=("roi_id_dudosa", "count"),
            pct_nearest_is_true_class=("nearest_is_true_class", lambda x: 100.0 * x.mean()),
            pct_nearest_is_top2_class=("nearest_is_top2_class", lambda x: 100.0 * x.mean()),
            mean_similarity=("cosine_similarity", "mean"),
        )
        .reset_index()
        .sort_values("y_true_dudosa")
    )
    by_true.to_csv(out_dir / "summary_by_true_class.csv", index=False)

    # Conteo de frecuencia de emparejamientos clase real -> clase del prototipo
    pair_counts = (
        rank1.groupby(["y_true_name", "prototype_class_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["y_true_name", "count"], ascending=[True, False])
    )
    pair_counts.to_csv(out_dir / "nearest_pair_counts.csv", index=False)

    # Exportamos listados específicos de casos de interés (acierto vs falsas alarmas)
    top2_cases = rank1[rank1["nearest_is_top2_class"]].copy()
    top2_cases.to_csv(out_dir / "nearest_matches_top2_cases.csv", index=False)

    true_cases = rank1[rank1["nearest_is_true_class"]].copy()
    true_cases.to_csv(out_dir / "nearest_matches_true_cases.csv", index=False)

    print("[INFO] Resúmenes generados correctamente.")
    print("\n=== Global summary ===")
    print(global_summary.to_string(index=False))

    print("\n=== Summary by true class ===")
    print(by_true.to_string(index=False))

    print("\n=== Top pairs (true class -> nearest prototype class) ===")
    print(pair_counts.head(20).to_string(index=False))


if __name__ == "__main__":
    main()