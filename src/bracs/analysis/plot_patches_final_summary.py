from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ensure_output_dir() -> Path:
    out_dir = Path("memoria/imagenes")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_performance_summary(out_dir: Path) -> None:
    """
    Genera una figura con dos subplots:
    - izquierda: F1 macro medio
    - derecha: Accuracy media
    para los cuatro métodos en ambos modelos.
    """

    methods = [
        "Baseline",
        "IM",
        "Random\n(70%)",
        "NCR\n(k=45)",
    ]

    # H-Optimus1
    hopt_f1 = [0.6118, 0.6012, 0.6001, 0.5980]
    hopt_acc = [0.7998, 0.7953, 0.7942, 0.7984]

    # Virchow2
    vir_f1 = [0.6011, 0.5908, 0.5920, 0.5785]
    vir_acc = [0.7891, 0.7917, 0.7908, 0.7895]

    # Colores bien diferenciados por método
    colors = {
        "Baseline": "#1f77b4",      # azul
        "IM": "#d62728",            # rojo
        "Random\n(70%)": "#ff7f0e", # naranja
        "NCR\n(k=45)": "#2ca02c",   # verde
    }

    method_colors = [colors[m] for m in methods]

    x = np.arange(len(methods))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    # -------------------------
    # Subplot 1: F1 macro
    # -------------------------
    ax = axes[0]
    bars1 = ax.bar(
        x - width / 2,
        hopt_f1,
        width,
        label="H-Optimus1",
        color=method_colors,
        alpha=0.95,
        edgecolor="black",
        linewidth=0.7,
    )
    bars2 = ax.bar(
        x + width / 2,
        vir_f1,
        width,
        label="Virchow2",
        color=method_colors,
        alpha=0.55,
        edgecolor="black",
        linewidth=0.7,
        hatch="//",
    )

    ax.set_title("Comparación global de rendimiento (F1 macro medio)", fontsize=13, weight="bold")
    ax.set_ylabel("F1 macro medio")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.57, 0.615)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.0006,
            f"{h:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    # -------------------------
    # Subplot 2: Accuracy
    # -------------------------
    ax = axes[1]
    bars1 = ax.bar(
        x - width / 2,
        hopt_acc,
        width,
        label="H-Optimus1",
        color=method_colors,
        alpha=0.95,
        edgecolor="black",
        linewidth=0.7,
    )
    bars2 = ax.bar(
        x + width / 2,
        vir_acc,
        width,
        label="Virchow2",
        color=method_colors,
        alpha=0.55,
        edgecolor="black",
        linewidth=0.7,
        hatch="//",
    )

    ax.set_title("Comparación global de rendimiento (accuracy media)", fontsize=13, weight="bold")
    ax.set_ylabel("Accuracy media")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.787, 0.8025)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.0002,
            f"{h:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    fig.suptitle(
        "Selección de patches: comparación final de rendimiento",
        fontsize=15,
        weight="bold",
        y=1.02,
    )
    plt.tight_layout()

    out_path = out_dir / "patches_performance_summary.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_computational_summary(out_dir: Path) -> None:
    """
    Genera una figura con dos subplots:
    - izquierda: tiempo de construcción del subconjunto
    - derecha: tiempo de reentrenamiento por semilla
    usando escala logarítmica en Y para apreciar mejor las diferencias.
    """

    methods = [
        "Baseline",
        "IM",
        "Random\n(70%)",
        "NCR\n(k=45)",
    ]

    # Tiempos de construcción (s)
    # Baseline no tiene construcción de subconjunto: lo representamos como 1 s
    # y lo anotamos como "---" visualmente.
    hopt_build = [1, 36 * 3600, 10, 26]
    vir_build = [1, 36.5 * 3600, 10, 42]

    # Tiempos de reentrenamiento por semilla (s)
    hopt_train = [3600, 9.6, 7.8, 8.8]
    vir_train = [3600, 8.4, 8.6, 8.9]

    colors = {
        "Baseline": "#1f77b4",
        "IM": "#d62728",
        "Random\n(70%)": "#ff7f0e",
        "NCR\n(k=45)": "#2ca02c",
    }
    method_colors = [colors[m] for m in methods]

    x = np.arange(len(methods))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    # -------------------------
    # Subplot 1: construcción
    # -------------------------
    ax = axes[0]
    bars1 = ax.bar(
        x - width / 2,
        hopt_build,
        width,
        label="H-Optimus1",
        color=method_colors,
        alpha=0.95,
        edgecolor="black",
        linewidth=0.7,
    )
    bars2 = ax.bar(
        x + width / 2,
        vir_build,
        width,
        label="Virchow2",
        color=method_colors,
        alpha=0.55,
        edgecolor="black",
        linewidth=0.7,
        hatch="//",
    )

    ax.set_title("Coste de construcción del subconjunto", fontsize=13, weight="bold")
    ax.set_ylabel("Tiempo (s) [escala log]")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.35, which="both")
    ax.legend()

    labels_hopt = ["---", "36 h", "10 s", "26 s"]
    labels_vir = ["---", "36.5 h", "10 s", "42 s"]

    for bar, txt in zip(bars1, labels_hopt):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.10,
            txt,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    for bar, txt in zip(bars2, labels_vir):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.10,
            txt,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    # -------------------------
    # Subplot 2: reentrenamiento
    # -------------------------
    ax = axes[1]
    bars1 = ax.bar(
        x - width / 2,
        hopt_train,
        width,
        label="H-Optimus1",
        color=method_colors,
        alpha=0.95,
        edgecolor="black",
        linewidth=0.7,
    )
    bars2 = ax.bar(
        x + width / 2,
        vir_train,
        width,
        label="Virchow2",
        color=method_colors,
        alpha=0.55,
        edgecolor="black",
        linewidth=0.7,
        hatch="//",
    )

    ax.set_title("Coste de reentrenamiento por semilla", fontsize=13, weight="bold")
    ax.set_ylabel("Tiempo (s) [escala log]")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.35, which="both")
    ax.legend()

    labels_hopt = ["1 h", "9.6 s", "7.8 s", "8.8 s"]
    labels_vir = ["1 h", "8.4 s", "8.6 s", "8.9 s"]

    for bar, txt in zip(bars1, labels_hopt):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.10,
            txt,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    for bar, txt in zip(bars2, labels_vir):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.10,
            txt,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    fig.suptitle(
        "Selección de patches: comparación computacional final",
        fontsize=15,
        weight="bold",
        y=1.02,
    )
    plt.tight_layout()

    out_path = out_dir / "patches_computational_summary.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    out_dir = ensure_output_dir()
    plot_performance_summary(out_dir)
    plot_computational_summary(out_dir)
    print(f"[INFO] Figuras guardadas en: {out_dir}")


if __name__ == "__main__":
    main()