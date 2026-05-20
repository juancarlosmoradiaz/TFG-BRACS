from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_output_dir() -> Path:
    out_dir = Path("memoria/imagenes")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    csv_path = Path("outputs/metrics/test_roi_abstention/abstention_tau010_final_summary.csv")
    df = pd.read_csv(csv_path)

    order = [
        ("h_optimus_1", "baseline", "H-Optimus1\nBaseline"),
        ("h_optimus_1", "im_alpha000025", "H-Optimus1\nIM"),
        ("h_optimus_1", "random_under", "H-Optimus1\nRU"),
        ("h_optimus_1", "ncr_k45", "H-Optimus1\nNCR"),
        ("virchow2", "baseline", "Virchow2\nBaseline"),
        ("virchow2", "im_alpha00002", "Virchow2\nIM"),
        ("virchow2", "random_under", "Virchow2\nRU"),
        ("virchow2", "ncr_k45", "Virchow2\nNCR"),
    ]

    color_map = {
        ("h_optimus_1", "baseline"): "#1f77b4",       # azul
        ("h_optimus_1", "im_alpha000025"): "#d62728", # rojo
        ("h_optimus_1", "random_under"): "#ff7f0e",   # naranja
        ("h_optimus_1", "ncr_k45"): "#2ca02c",        # verde
        ("virchow2", "baseline"): "#9467bd",          # morado
        ("virchow2", "im_alpha00002"): "#8c564b",     # marrón
        ("virchow2", "random_under"): "#e377c2",      # rosa
        ("virchow2", "ncr_k45"): "#17becf",           # cian
    }

    rows = []
    colors = []
    for model, method, label in order:
        sub = df[(df["model"] == model) & (df["method"] == method)]
        if len(sub) != 1:
            raise ValueError(f"No se encontró una única fila para {(model, method)}")
        row = sub.iloc[0].to_dict()
        row["label"] = label
        rows.append(row)
        colors.append(color_map[(model, method)])

    plot_df = pd.DataFrame(rows)

    labels = plot_df["label"].tolist()
    f1_vals = plot_df["f1_macro_accepted"].to_numpy(dtype=float)
    acc_vals = plot_df["accuracy_accepted"].to_numpy(dtype=float)

    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))

    def add_bar_labels(ax, bars, values, offset):
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + offset,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                weight="bold",
            )

    # -----------------------------
    # Panel 1: F1 macro accepted
    # -----------------------------
    ax = axes[0]
    bars = ax.bar(
        x,
        f1_vals,
        color=colors,
        edgecolor="black",
        linewidth=0.9,
    )
    ax.set_title("F1 macro sobre ROIs aceptadas", fontsize=13, weight="bold")
    ax.set_ylabel("F1 macro accepted")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylim(0.50, 0.61)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    add_bar_labels(ax, bars, f1_vals, 0.002)

    # -----------------------------
    # Panel 2: Accuracy accepted
    # -----------------------------
    ax = axes[1]
    bars = ax.bar(
        x,
        acc_vals,
        color=colors,
        edgecolor="black",
        linewidth=0.9,
    )
    ax.set_title("Accuracy sobre ROIs aceptadas", fontsize=13, weight="bold")
    ax.set_ylabel("Accuracy accepted")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylim(0.59, 0.65)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    add_bar_labels(ax, bars, acc_vals, 0.0015)

    fig.suptitle(
        "Resultados globales con abstención ($\\tau = 0.10$)",
        fontsize=16,
        weight="bold",
        y=1.02,
    )
    plt.tight_layout()

    out_dir = ensure_output_dir()
    out_path = out_dir / "roi_abstention_global_summary.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Figura guardada en: {out_path}")


if __name__ == "__main__":
    main()