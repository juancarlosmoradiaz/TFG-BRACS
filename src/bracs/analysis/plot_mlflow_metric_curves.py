from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera una gráfica comparando dos métricas exportadas desde MLflow en CSV."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Ruta al CSV exportado desde MLflow.",
    )
    parser.add_argument(
        "--metric_a",
        type=str,
        required=True,
        help="Nombre de la primera métrica, por ejemplo: train_loss",
    )
    parser.add_argument(
        "--metric_b",
        type=str,
        required=True,
        help="Nombre de la segunda métrica, por ejemplo: val_loss",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Comparación de métricas",
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

    expected_cols = {"metric", "step", "value"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"El CSV no tiene las columnas esperadas. Faltan: {sorted(missing_cols)}"
        )

    df_a = (
        df[df["metric"] == args.metric_a][["step", "value"]]
        .sort_values("step")
        .reset_index(drop=True)
    )
    df_b = (
        df[df["metric"] == args.metric_b][["step", "value"]]
        .sort_values("step")
        .reset_index(drop=True)
    )

    if df_a.empty:
        raise ValueError(f"No se ha encontrado la métrica '{args.metric_a}' en el CSV.")
    if df_b.empty:
        raise ValueError(f"No se ha encontrado la métrica '{args.metric_b}' en el CSV.")

    plt.figure(figsize=(8, 5))
    plt.plot(df_a["step"], df_a["value"], marker="o", label=args.metric_a)
    plt.plot(df_b["step"], df_b["value"], marker="o", label=args.metric_b)

    plt.title(args.title)
    plt.xlabel("Época")
    plt.ylabel(args.ylabel)
    plt.xticks(df_a["step"].tolist())
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Figura guardada en: {output_path}")


if __name__ == "__main__":
    main()