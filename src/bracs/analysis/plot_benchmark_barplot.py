from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera una gráfica de barras a partir de un CSV con resultados agregados del benchmark."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Ruta al CSV con columnas: model, mean, std",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Benchmark ranking",
        help="Título de la figura.",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default="Valor",
        help="Etiqueta del eje Y.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Ruta del PNG de salida.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    expected = {"model", "mean", "std"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {sorted(missing)}")

    # Ordenamos de mayor a menor
    df = df.sort_values("mean", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(9, 5.5))
    plt.bar(df["model"], df["mean"], yerr=df["std"], capsize=5)
    plt.title(args.title)
    plt.ylabel(args.ylabel)
    plt.xlabel("Modelo")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Figura guardada en: {output_path}")


if __name__ == "__main__":
    main()