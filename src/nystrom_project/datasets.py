"""Dataset provenance, training-only preprocessing and dense accuracy references."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.spatial.distance import cdist, pdist


SOURCES = {
    "concrete": {
        "url": "https://archive.ics.uci.edu/static/public/165/concrete%2Bcompressive%2Bstrength.zip",
        "page": "https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength",
        "doi": "10.24432/C5PK67", "rows": 1030, "features": 8,
        "citation": "Yeh, I. (1998). Concrete Compressive Strength. UCI Machine Learning Repository.",
    },
    "airfoil": {
        "url": "https://archive.ics.uci.edu/static/public/291/airfoil%2Bself%2Bnoise.zip",
        "page": "https://archive.ics.uci.edu/dataset/291/airfoil+self+noise",
        "doi": "10.24432/C5VW2C", "rows": 1503, "features": 5,
        "citation": "Brooks, T., Pope, D., and Marcolini, M. (1989). Airfoil Self-Noise. UCI Machine Learning Repository.",
    },
}


def acquire(root):
    root = Path(root)
    raw = root/"data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, source in SOURCES.items():
        archive = raw/f"{name}.zip"
        if not archive.exists():
            with urlopen(source["url"], timeout=45) as response:
                content = response.read()
            archive.write_bytes(content)
        content = archive.read_bytes()
        with ZipFile(BytesIO(content)) as z:
            if name == "concrete":
                import xlrd
                member = next(p for p in z.namelist() if p.lower().endswith("concrete_data.xls"))
                sheet = xlrd.open_workbook(file_contents=z.read(member)).sheet_by_index(0)
                values = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)], dtype=float)
            else:
                member = next(p for p in z.namelist() if p.endswith("airfoil_self_noise.dat"))
                values = np.loadtxt(BytesIO(z.read(member)))
        assert values.shape == (source["rows"], source["features"]+1)
        assert np.all(np.isfinite(values))
        dest = root/"data"/f"{name}.npz"
        np.savez_compressed(dest, X=values[:, :-1], y=values[:, -1])
        manifest[name] = dict(source, archive_sha256=sha256(content).hexdigest(),
                              extracted_member=member,
                              arrays_sha256=sha256(values.tobytes()).hexdigest(),
                              license="CC BY 4.0", obtained="2026-11-23")
    (root/"data/manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


def mixed_eigenspace_targets(X, seed):
    """Construct two equally weighted spectral target components on training X.

    The leading band contains descending eigenvectors 1:9 (zero based),
    excluding the first vector. The second band starts at max(16,n_train//8)
    and also contains eight vectors. Each component has zero training mean
    and squared norm n_train/2. Test labels extend those same kernel
    eigenfunctions; test inputs do not determine the bands or coefficients.
    This dense target construction is offline and never enters a solver.
    """
    X = np.asarray(X, dtype=float)
    order = np.random.default_rng(np.random.SeedSequence([seed, 6220, 12])).permutation(len(X))
    cut = int(.8*len(X))
    middle = max(16, cut//8)
    if middle+8 > cut:
        raise ValueError("mixed eigenspace targets require at least 24 training examples")
    train, test = order[:cut], order[cut:]
    mean, std = X[train].mean(axis=0), X[train].std(axis=0)
    std[std == 0] = 1
    Xtr, Xte = (X[train]-mean)/std, (X[test]-mean)/std
    bandwidth = float(np.median(pdist(Xtr)))
    if bandwidth <= 0:
        raise ValueError("Median training distance is zero")
    K = np.exp(-cdist(Xtr, Xtr, "sqeuclidean")/(2*bandwidth**2))
    values, basis = eigh(K, check_finite=False)
    values, basis = values[::-1], basis[:, ::-1]
    # Fix eigenvector signs so LAPACK sign choices do not change the targets.
    largest = np.argmax(np.abs(basis), axis=0)
    signs = np.sign(basis[largest, np.arange(cut)])
    basis = basis*signs[None, :]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 6220, 17]))
    train_target = np.zeros(cut)
    coefficients = np.zeros(cut)
    for band in (np.arange(1, 9), np.arange(middle, middle+8)):
        if values[band[-1]] <= 1e-12*values[0]:
            raise ValueError("Requested target band is numerically unresolved")
        V = basis[:, band]
        c = rng.normal(size=len(band))
        means = V.sum(axis=0)
        if np.linalg.norm(means) > 1e-12*np.sqrt(cut):
            c -= means*(means@c)/(means@means)
        c *= np.sqrt(cut/2)/np.linalg.norm(c)
        train_target += V@c
        coefficients += V@(c/values[band])
    cross = np.exp(-cdist(Xte, Xtr, "sqeuclidean")/(2*bandwidth**2))
    y = np.empty(len(X))
    y[train], y[test] = train_target, cross@coefficients
    return y


def raw_data(root, dataset, seed, target="observed", synthetic_n=640):
    if dataset != "synthetic":
        with np.load(Path(root)/"data"/f"{dataset}.npz") as data:
            return data["X"], data["y"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 6220, 10]))
    centers = rng.normal(0, 1.5, (4, 8))
    labels = rng.choice(4, synthetic_n, p=[.5, .25, .15, .10])
    X = centers[labels]+rng.normal(0, .45, (synthetic_n, 8))
    target_rng = np.random.default_rng(np.random.SeedSequence([seed, 6220, 11]))
    if target == "smooth":
        y = np.sin(X[:, 0])+.3*X[:, 1]**2-.2*X[:, 2]+.05*target_rng.normal(size=synthetic_n)
    elif target == "random":
        y = target_rng.normal(size=synthetic_n)
    elif target == "mixed":
        y = mixed_eigenspace_targets(X, seed)
    else:
        raise ValueError(f"Unknown synthetic target {target}")
    return X, y


def prepare(root, dataset, seed, regularization, target="observed", synthetic_n=640):
    """Prepare a case; reference Cholesky and dense K never enter the solve worker."""
    started = perf_counter()
    root = Path(root)
    X, y = raw_data(root, dataset, seed, target, synthetic_n)
    split_rng = np.random.default_rng(np.random.SeedSequence([seed, 6220, 12]))
    order = split_rng.permutation(len(y))
    cut = int(.8*len(y))
    train, test = order[:cut], order[cut:]
    Xtr, Xte = X[train], X[test]
    mean, std = Xtr.mean(axis=0), Xtr.std(axis=0)
    std[std == 0] = 1
    Xtr, Xte = (Xtr-mean)/std, (Xte-mean)/std
    ymean, ystd = float(y[train].mean()), float(y[train].std())
    ystd = ystd if ystd else 1.0
    ytr, yte = (y[train]-ymean)/ystd, (y[test]-ymean)/ystd
    bandwidth = float(np.median(pdist(Xtr)))
    if bandwidth <= 0:
        raise ValueError("Median training distance is zero")
    preprocessing_seconds = perf_counter()-started
    reference_started = perf_counter()
    K = np.exp(-cdist(Xtr, Xtr, "sqeuclidean")/(2*bandwidth**2))
    K.flat[::len(ytr)+1] += regularization
    alpha = cho_solve(cho_factor(K, lower=True, check_finite=False), ytr, check_finite=False)
    norm_ytr = np.linalg.norm(ytr)
    reference_residual = float(np.linalg.norm(ytr-K @ alpha)/norm_ytr) if norm_ytr else 0.0
    reference_solve_seconds = perf_counter()-reference_started
    predictions = np.exp(-cdist(Xte, Xtr, "sqeuclidean")/(2*bandwidth**2)) @ alpha
    reference_seconds = perf_counter()-reference_started
    config = dict(dataset=dataset, seed=seed, target=target, regularization=regularization,
                  synthetic_n=synthetic_n if dataset == "synthetic" else None,
                  n_train=len(ytr), n_test=len(yte), bandwidth=bandwidth,
                  train_fraction=.8, preprocessing_seconds=preprocessing_seconds,
                  reference_preparation_seconds=reference_seconds,
                  reference_solve_seconds=reference_solve_seconds,
                  reference_relative_residual=reference_residual,
                  dataset_sha256=sha256(np.ascontiguousarray(np.column_stack((X, y))).tobytes()).hexdigest(),
                  train_indices_sha256=sha256(train.tobytes()).hexdigest())
    if dataset == "synthetic" and target == "mixed":
        middle = max(16, len(ytr)//8)
        config["target_construction"] = dict(
            type="mixture of training-kernel eigenspaces",
            descending_zero_based_bands=[[1, 9], [middle, middle+8]],
            band_interval="half-open", equal_training_squared_norm=True,
            zero_training_mean_per_component=True,
            coefficient_seed=[seed, 6220, 17],
            test_targets="kernel eigenfunction extension from training inputs",
            offline_eigendecomposition=True,
            eigenvectors_available_to_solver=False)
    size = f"-n{synthetic_n}" if dataset == "synthetic" else ""
    case_id = f"{dataset}{size}-{target}-s{seed}-lambda{regularization:g}"
    prepared = root/"data/prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    path = prepared/f"{case_id}.npz"
    np.savez_compressed(path, X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
                        alpha_reference=alpha, prediction_reference=predictions,
                        bandwidth=bandwidth, y_std=ystd, config=json.dumps(config))
    return path, config


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(acquire(args.root), indent=2))
