from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Mapeo desde data_roi_3cls_full.pkl
# AT (0) <- FEA(3), ADH(4)
# BT (1) <- N(0), PB(1), UDH(2)
# MT (2) <- DCIS(5), IC(6)

THREE_CLASS_NAMES = {
    0: "AT",
    1: "BT",
    2: "MT",
}

SEVEN_TO_THREE = {
    0: 1,  # N   -> BT
    1: 1,  # PB  -> BT
    2: 1,  # UDH -> BT
    3: 0,  # FEA -> AT
    4: 0,  # ADH -> AT
    5: 2,  # DCIS -> MT
    6: 2,  # IC   -> MT
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye predicciones ROI en 3 clases a partir de probabilidades ROI en 7 clases."
    )
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Cargando CSV ROI 7-clases: {in_path}")
    df = pd.read_csv(in_path)

    required_cols = {
        "roi_id",
        "y_true_roi",
        "n_patches",
        "mean_prob_0",
        "mean_prob_1",
        "mean_prob_2",
        "mean_prob_3",
        "mean_prob_4",
        "mean_prob_5",
        "mean_prob_6",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")

    out_rows = []

    for _, row in df.iterrows():
        y_true_7 = int(row["y_true_roi"])
        y_true_3 = SEVEN_TO_THREE[y_true_7]

        prob_AT = float(row["mean_prob_3"] + row["mean_prob_4"])
        prob_BT = float(row["mean_prob_0"] + row["mean_prob_1"] + row["mean_prob_2"])
        prob_MT = float(row["mean_prob_5"] + row["mean_prob_6"])

        probs_3 = np.array([prob_AT, prob_BT, prob_MT], dtype=float)

        # pequeña comprobación numérica
        if not np.isclose(probs_3.sum(), 1.0, atol=1e-6):
            raise ValueError(
                f"Las probabilidades 3-clases no suman 1 en ROI {row['roi_id']}: {probs_3.sum()}"
            )

        order = np.argsort(probs_3)[::-1]
        top1 = int(order[0])
        top2 = int(order[1])

        out_rows.append({
            "roi_id": row["roi_id"],
            "model": args.model,
            "method": args.method,
            "y_true_roi_7cls": y_true_7,
            "y_true_roi_3cls": y_true_3,
            "y_true_roi_3cls_name": THREE_CLASS_NAMES[y_true_3],
            "n_patches": int(row["n_patches"]),
            "prob_AT": prob_AT,
            "prob_BT": prob_BT,
            "prob_MT": prob_MT,
            "y_pred_3cls": top1,
            "y_pred_3cls_name": THREE_CLASS_NAMES[top1],
            "top1_3cls": top1,
            "top1_3cls_name": THREE_CLASS_NAMES[top1],
            "top1_3cls_prob": float(probs_3[top1]),
            "top2_3cls": top2,
            "top2_3cls_name": THREE_CLASS_NAMES[top2],
            "top2_3cls_prob": float(probs_3[top2]),
            "margin_top1_top2_3cls": float(probs_3[top1] - probs_3[top2]),
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_path, index=False)

    print("[INFO] CSV 3-clases generado correctamente.")
    print(f"[INFO] Nº ROIs: {len(out_df)}")
    print(f"[INFO] CSV guardado en: {out_path}")


if __name__ == "__main__":
    main()