from __future__ import annotations

from pathlib import Path
import math
import textwrap

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd


CLASS_ID_TO_NAME = {
    0: "N",
    1: "PB",
    2: "UDH",
    3: "FEA",
    4: "ADH",
    5: "DCIS",
    6: "IC",
}

CLASS_TO_FOLDER = {
    "N": "0_N",
    "PB": "1_PB",
    "UDH": "2_UDH",
    "FEA": "3_FEA",
    "ADH": "4_ADH",
    "DCIS": "5_DCIS",
    "IC": "6_IC",
}


def ensure_output_dir() -> Path:
    out_dir = Path("memoria/imagenes/roi_case_studies")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def add_name_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "y_true_roi" in df.columns and "y_true_name" not in df.columns:
        df["y_true_name"] = df["y_true_roi"].map(CLASS_ID_TO_NAME)
    for col in ["top1_class", "top2_class", "top3_class"]:
        name_col = col.replace("_class", "_name")
        if col in df.columns and name_col not in df.columns:
            df[name_col] = df[col].map(CLASS_ID_TO_NAME)
    return df


def load_case_row(
    csv_path: Path,
    model: str,
    method: str,
    roi_id: str,
    add_model_method: bool = False,
) -> pd.Series:
    df = pd.read_csv(csv_path)
    df = add_name_columns(df)

    if add_model_method:
        df["model"] = model
        df["method"] = method

    required = {"roi_id", "model", "method"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {csv_path}: {sorted(missing)}")

    sub = df[(df["model"] == model) & (df["method"] == method) & (df["roi_id"] == roi_id)]
    if len(sub) != 1:
        raise ValueError(
            f"No se encontró una única fila para model={model}, method={method}, roi_id={roi_id} en {csv_path}"
        )
    return sub.iloc[0]


def find_roi_image(roi_id: str, y_true_name: str) -> Path:
    roi_root = Path("data/histoimage/BRACS_RoI/latest_version/test")
    class_folder = CLASS_TO_FOLDER[y_true_name]
    roi_path = roi_root / class_folder / f"{roi_id}.png"
    if not roi_path.exists():
        raise FileNotFoundError(f"No existe la ROI original: {roi_path}")
    return roi_path


def find_patch_images(roi_id: str, y_true_name: str) -> list[Path]:
    patch_root = Path("data/histoimage/BRACS_RoI_patches_512_overlap_full/test")
    class_folder = CLASS_TO_FOLDER[y_true_name]
    patch_dir = patch_root / class_folder
    patch_paths = sorted(patch_dir.glob(f"{roi_id}_*.jpeg"))
    if not patch_paths:
        raise FileNotFoundError(f"No se encontraron patches para {roi_id} en {patch_dir}")
    return patch_paths


def format_prob_line(row: pd.Series) -> str:
    return (
        f"Top-1: {row['top1_name']} ({row['top1_prob']:.3f})   |   "
        f"Top-2: {row['top2_name']} ({row['top2_prob']:.3f})   |   "
        f"Top-3: {row['top3_name']} ({row['top3_prob']:.3f})"
    )


def format_meta_line(row: pd.Series) -> str:
    return (
        f"Clase real: {row['y_true_name']}   |   "
        f"n_patches: {int(row['n_patches'])}   |   "
        f"Margen top1-top2: {row['margin_top1_top2']:.4f}   |   "
        f"Entropía: {row['entropy']:.4f}"
    )


def pretty_model_method(model: str, method: str) -> str:
    model_name = "H-Optimus1" if model == "h_optimus_1" else "Virchow2"
    method_name_map = {
        "baseline": "Baseline",
        "random_under": "RandomUnderSampler",
        "im_alpha000025": "Información mutua",
        "im_alpha00002": "Información mutua",
        "ncr_k45": "NCR (k=45)",
    }
    method_name = method_name_map.get(method, method)
    return f"{model_name} + {method_name}"


def choose_patch_subset(patch_paths: list[Path], max_patches: int = 9) -> list[Path]:
    if len(patch_paths) <= max_patches:
        return patch_paths

    idxs = []
    for i in range(max_patches):
        pos = round(i * (len(patch_paths) - 1) / (max_patches - 1))
        idxs.append(pos)

    idxs = sorted(set(idxs))
    selected = [patch_paths[i] for i in idxs]

    if len(selected) < max_patches:
        used = set(idxs)
        for i in range(len(patch_paths)):
            if i not in used:
                selected.append(patch_paths[i])
            if len(selected) == max_patches:
                break

    return selected[:max_patches]


def draw_case_figure(
    row: pd.Series,
    output_path: Path,
    max_patches: int = 9,
) -> None:
    roi_id = row["roi_id"]
    y_true_name = row["y_true_name"]
    model = row["model"]
    method = row["method"]

    roi_img_path = find_roi_image(roi_id, y_true_name)
    patch_paths = find_patch_images(roi_id, y_true_name)
    patch_subset = choose_patch_subset(patch_paths, max_patches=max_patches)

    roi_img = mpimg.imread(roi_img_path)
    patch_imgs = [mpimg.imread(p) for p in patch_subset]

    n_cols = 3
    n_rows = math.ceil(len(patch_imgs) / n_cols)

    fig = plt.figure(figsize=(15, 9))

    # Reservamos más espacio arriba para que no haya solape
    gs = fig.add_gridspec(
        nrows=max(n_rows, 3),
        ncols=5,
        left=0.04,
        right=0.98,
        bottom=0.08,
        top=0.74,   # <- clave: baja la zona de imágenes
        width_ratios=[2.2, 0.08, 1, 1, 1],
        hspace=0.28,
        wspace=0.18,
    )

    # ROI original
    ax_roi = fig.add_subplot(gs[:, 0])
    ax_roi.imshow(roi_img)
    ax_roi.set_title("ROI original", fontsize=13, weight="bold", pad=10)
    ax_roi.axis("off")

    # Patches
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx >= len(patch_imgs):
                continue
            ax = fig.add_subplot(gs[i, 2 + j])
            ax.imshow(patch_imgs[idx])
            ax.set_title(f"Patch {idx+1}", fontsize=9, pad=4)
            ax.axis("off")

    # Cabecera superior
    title_line = pretty_model_method(model, method)
    fig.suptitle(
        f"{title_line}  |  ROI: {roi_id}",
        fontsize=17,
        weight="bold",
        y=0.965,
    )

    meta_line = format_meta_line(row)
    prob_line = format_prob_line(row)

    fig.text(
        0.5,
        0.905,
        "\n".join(textwrap.wrap(meta_line, width=110)),
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
    )
    fig.text(
        0.5,
        0.855,
        "\n".join(textwrap.wrap(prob_line, width=110)),
        ha="center",
        va="center",
        fontsize=11,
    )

    note = (
        f"Patches mostrados: {len(patch_subset)} de {len(patch_paths)} totales"
        if len(patch_paths) > len(patch_subset)
        else f"Patches mostrados: {len(patch_subset)}"
    )
    fig.text(
        0.5,
        0.03,
        note,
        ha="center",
        fontsize=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    review_csv = Path("outputs/metrics/test_roi_abstention/summaries/all_review_cases_tau010.csv")
    decisions_base = Path("outputs/metrics/test_roi_abstention")

    selected_cases = [
        {
            "source": "review",
            "model": "virchow2",
            "method": "random_under",
            "roi_id": "BRACS_1599_N_2",
            "output_name": "case_pb_vs_n_virchow2_ru.png",
        },
        {
            "source": "review",
            "model": "h_optimus_1",
            "method": "baseline",
            "roi_id": "BRACS_1283_FEA_8",
            "output_name": "case_fea_vs_adh_hoptimus1_baseline.png",
        },
        {
            "source": "decision",
            "model": "virchow2",
            "method": "random_under",
            "roi_id": "BRACS_1826_IC_1",
            "output_name": "case_clear_ic_virchow2_ru.png",
        },
    ]

    out_dir = ensure_output_dir()

    for case in selected_cases:
        if case["source"] == "review":
            row = load_case_row(
                csv_path=review_csv,
                model=case["model"],
                method=case["method"],
                roi_id=case["roi_id"],
                add_model_method=False,
            )
        else:
            decision_csv = decisions_base / case["model"] / f"{case['method']}_tau010_all_decisions.csv"
            row = load_case_row(
                csv_path=decision_csv,
                model=case["model"],
                method=case["method"],
                roi_id=case["roi_id"],
                add_model_method=True,   # <- clave: añade model/method al CSV
            )

        out_path = out_dir / case["output_name"]
        draw_case_figure(row=row, output_path=out_path, max_patches=9)
        print(f"[INFO] Figura guardada en: {out_path}")


if __name__ == "__main__":
    main()