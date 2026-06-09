# ---------------------------------------------
# SELECCIÓN DE CASOS DE ESTUDIO DE ROI
# ---------------------------------------------
# Entrada:
#   - Archivos CSV con casos de revisión y decisiones de abstención
#
# Salida:
#   - Archivos CSV con candidatos de interés seleccionados para análisis cualitativo:
#       * Casos dudosos entre PB (Hiperplasia benigna) y N (Tejido normal)
#       * Casos de transición (FEA, ADH, DCIS)
#       * Casos de diagnóstico claro (N o IC)
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
        description="Selecciona candidatos ROI para análisis cualitativo."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Directorio base que contiene los CSV de métricas de abstención.",
    )
    parser.add_argument(
        "--tau_tag",
        type=str,
        default="tau010",
        help="Etiqueta correspondiente al valor de tau (ej. tau010).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Número de candidatos a preseleccionar para cada categoría.",
    )
    return parser.parse_args()


def add_name_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas con los nombres de clases diagnósticas basados en los identificadores numéricos.
    """
    df = df.copy()
    for col in ["y_true_roi", "top1_class", "top2_class", "top3_class"]:
        if col in df.columns:
            name_col = col.replace("_class", "_name").replace("y_true_roi", "y_true_name")
            df[name_col] = df[col].map(CLASS_NAMES)
    return df


def filter_pair(df: pd.DataFrame, a: int, b: int) -> pd.DataFrame:
    """
    Filtra los registros donde las dos primeras predicciones más probables correspondan a la pareja (a, b).
    """
    mask = ((df["top1_class"] == a) & (df["top2_class"] == b)) | (
        (df["top1_class"] == b) & (df["top2_class"] == a)
    )
    return df[mask].copy()


def load_review_cases(base_dir: Path, model: str, method: str, tau_tag: str) -> pd.DataFrame:
    """
    Carga el CSV de casos dudosos enviados a revisión para una configuración específica.
    """
    path = base_dir / model / f"{method}_{tau_tag}_review_cases.csv"
    df = pd.read_csv(path)
    df["model"] = model
    df["method"] = method
    return add_name_columns(df)


def load_all_decisions(base_dir: Path, model: str, method: str, tau_tag: str) -> pd.DataFrame:
    """
    Carga el CSV de todas las decisiones (aceptados y rechazados) para una configuración.
    """
    path = base_dir / model / f"{method}_{tau_tag}_all_decisions.csv"
    df = pd.read_csv(path)
    df["model"] = model
    df["method"] = method
    return add_name_columns(df)


def sort_review_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena los casos para revisión priorizando el menor margen top1-top2 y la mayor entropía.
    """
    return df.sort_values(
        by=["margin_top1_top2", "entropy", "n_patches"],
        ascending=[True, False, False],
    )


def sort_clear_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena los casos con predicción clara priorizando el mayor margen top1-top2 y la menor entropía.
    """
    return df.sort_values(
        by=["margin_top1_top2", "entropy", "top1_prob"],
        ascending=[False, True, False],
    )


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = base_dir / "case_studies"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configuraciones específicas elegidas para el análisis cualitativo
    configs = [
        ("h_optimus_1", "baseline"),
        ("virchow2", "random_under"),
    ]

    review_frames = []
    decision_frames = []

    for model, method in configs:
        review_frames.append(load_review_cases(base_dir, model, method, args.tau_tag))
        decision_frames.append(load_all_decisions(base_dir, model, method, args.tau_tag))

    review_df = pd.concat(review_frames, ignore_index=True)
    decisions_df = pd.concat(decision_frames, ignore_index=True)

    # Candidatos dudosos entre PB (hiperplasia benigna) y N (tejido normal)
    pb_n_df = filter_pair(review_df, 0, 1)
    pb_n_df = sort_review_candidates(pb_n_df).head(args.top_k)

    # Candidatos en zonas de transición diagnóstica (FEA <-> ADH y ADH <-> DCIS)
    fea_adh_df = filter_pair(review_df, 3, 4)
    adh_dcis_df = filter_pair(review_df, 4, 5)
    transition_df = pd.concat([fea_adh_df, adh_dcis_df], ignore_index=True)
    transition_df = sort_review_candidates(transition_df).head(args.top_k)

    # Casos con diagnósticos claros y bien aceptados por el clasificador (clase real Normal o Carcinoma Invasivo)
    clear_df = decisions_df[
        (decisions_df["accepted"] == True) &
        (decisions_df["y_true_roi"].isin([0, 6]))
    ].copy()
    clear_df = sort_clear_candidates(clear_df).head(args.top_k)

    # Definimos las columnas útiles que se guardarán para el análisis cualitativo
    cols = [
        "model",
        "method",
        "roi_id",
        "y_true_roi",
        "y_true_name",
        "n_patches",
        "top1_class",
        "top1_name",
        "top1_prob",
        "top2_class",
        "top2_name",
        "top2_prob",
        "top3_class",
        "top3_name",
        "top3_prob",
        "margin_top1_top2",
        "entropy",
    ]

    pb_n_path = out_dir / f"candidates_pb_n_{args.tau_tag}.csv"
    transition_path = out_dir / f"candidates_transition_{args.tau_tag}.csv"
    clear_path = out_dir / f"candidates_clear_{args.tau_tag}.csv"

    # Exportamos los candidatos a sus respectivos archivos CSV
    pb_n_df[cols].to_csv(pb_n_path, index=False)
    transition_df[cols].to_csv(transition_path, index=False)
    clear_df[cols].to_csv(clear_path, index=False)

    print("\n[INFO] Candidatos PB <-> N")
    print(pb_n_df[cols].to_string(index=False))

    print("\n[INFO] Candidatos de transición (FEA <-> ADH, ADH <-> DCIS)")
    print(transition_df[cols].to_string(index=False))

    print("\n[INFO] Casos claros (N o IC)")
    print(clear_df[cols].to_string(index=False))

    print(f"\n[INFO] CSV guardado en: {pb_n_path}")
    print(f"[INFO] CSV guardado en: {transition_path}")
    print(f"[INFO] CSV guardado en: {clear_path}")


if __name__ == "__main__":
    main()