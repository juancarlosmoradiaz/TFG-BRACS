from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def load_hyindex_function(hyindex_repo: Path):
    """
    Carga get_hy_index_from_data desde el repo de Hy-index
    importándolo como paquete (src.hy_index), para que
    funcionen correctamente los imports relativos internos.
    """
    import sys

    hyindex_repo = hyindex_repo.resolve()
    if not hyindex_repo.exists():
        raise FileNotFoundError(f"No existe el repositorio Hy-index: {hyindex_repo}")

    # Muy importante: meter el repo al principio del sys.path
    repo_str = str(hyindex_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from src.hy_index import get_hy_index_from_data

    return get_hy_index_from_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta selección de variables con Hy-index sobre embeddings de train."
    )
    parser.add_argument("--train_h5", type=str, required=True)
    parser.add_argument("--hyindex_repo", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_subsets", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_h5 = Path(args.train_h5)
    hyindex_repo = Path(args.hyindex_repo)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_h5.exists():
        raise FileNotFoundError(f"No existe train_h5: {train_h5}")

    print(f"[INFO] Cargando H5: {train_h5}")
    with h5py.File(train_h5, "r") as f:
        X = f["features"][:].astype(np.float32)
        y = f["labels"][:].astype(np.int64)

    print(f"[INFO] Shape X_train: {X.shape}")
    print(f"[INFO] Shape y_train: {y.shape}")

    if args.max_samples is not None:
        rng = np.random.default_rng(args.random_state)
        n_total = X.shape[0]
        n_keep = min(args.max_samples, n_total)

        idx = rng.choice(n_total, size=n_keep, replace=False)
        idx = np.sort(idx)

        X = X[idx]
        y = y[idx]

        print(f"[INFO] Submuestreo activado: {n_keep} muestras de {n_total}")
        print(f"[INFO] Nueva shape X_train: {X.shape}")
        print(f"[INFO] Nueva shape y_train: {y.shape}")

    get_hy_index_from_data = load_hyindex_function(hyindex_repo)
    print("[INFO] Hy-index cargado correctamente.")

    print(
        f"[INFO] Ejecutando Hy-index con n_subsets={args.n_subsets}, "
        f"threshold={args.threshold}, k={args.k}, random_state={args.random_state}"
    )

    result = get_hy_index_from_data(
        X,
        y,
        n_subsets=args.n_subsets,
        threshold=args.threshold,
        k=args.k,
        random_state=args.random_state,
    )

    if not isinstance(result, dict):
        raise TypeError("Hy-index no ha devuelto un diccionario como indica su docstring.")

    print(f"[INFO] Claves devueltas por Hy-index: {list(result.keys())}")

    selected = result.get("selected_features", None)
    if selected is None:
        raise KeyError("El resultado no contiene la clave 'selected_features'.")

    selected = np.array(selected)
    if selected.dtype == bool:
        selected_idx = np.flatnonzero(selected)
    else:
        selected_idx = selected.astype(int)

    n_total = X.shape[1]
    n_selected = len(selected_idx)
    pct_selected = 100.0 * n_selected / n_total

    print(f"[INFO] Variables totales: {n_total}")
    print(f"[INFO] Variables seleccionadas: {n_selected}")
    print(f"[INFO] Porcentaje retenido: {pct_selected:.2f}%")

    # Guardamos índices seleccionados
    selected_idx_path = output_dir / f"{args.model}_hyindex_selected_idx_thr{str(args.threshold).replace('.', '')}.npy"
    np.save(selected_idx_path, selected_idx)

    # Guardamos máscara booleana por comodidad
    mask = np.zeros(n_total, dtype=bool)
    mask[selected_idx] = True
    mask_path = output_dir / f"{args.model}_hyindex_selected_mask_thr{str(args.threshold).replace('.', '')}.npy"
    np.save(mask_path, mask)

    # Guardamos resumen JSON
    summary = {
        "model": args.model,
        "train_h5": str(train_h5),
        "n_samples": int(X.shape[0]),
        "n_features_total": int(n_total),
        "n_features_selected": int(n_selected),
        "pct_selected": float(pct_selected),
        "n_subsets": int(args.n_subsets),
        "threshold": float(args.threshold),
        "k": int(args.k),
        "random_state": int(args.random_state),
        "result_keys": list(result.keys()),
        "max_samples": None if args.max_samples is None else int(args.max_samples),
    }

    # Si viene hy_index en el dict, lo guardamos
    if "hy_index" in result:
        try:
            summary["hy_index"] = float(result["hy_index"])
        except Exception:
            summary["hy_index"] = str(result["hy_index"])

    summary_path = output_dir / f"{args.model}_hyindex_summary_thr{str(args.threshold).replace('.', '')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Índices guardados en: {selected_idx_path}")
    print(f"[INFO] Máscara guardada en: {mask_path}")
    print(f"[INFO] Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()