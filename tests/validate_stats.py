"""Check app/stats.js against numpy, pandas and scipy on the real dataset.

Runs the JavaScript through Node (tests/stats_runner.js) on inhalation_dataset
and compares every statistic the app shows. Fails with an AssertionError if a
difference exceeds the tolerance.

Usage:  py -3.13 tests/validate_stats.py      (needs Node.js on PATH)
Without a local build it uses the database shipped with the app (identical file).
"""

from __future__ import annotations

import base64
import gzip
import json
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
VARIABLES = ["mw_freebase", "alogp", "psa", "hbd", "hba", "rtb", "aromatic_rings", "heavy_atoms", "qed_weighted"]
TOL = 1e-9          # deterministic arithmetic
TOL_P = 1e-6        # p-values: erfc approximation has relative error < 1.2e-7


def check(label: str, js, py, tol: float = TOL, relative: bool = False) -> None:
    js, py = np.asarray(js, dtype=float), np.asarray(py, dtype=float)
    diff = np.abs(js - py)
    if relative:
        diff = diff / np.maximum(np.abs(py), 1e-300)
    worst = float(np.nanmax(diff))
    status = "ok " if worst <= tol else "FAIL"
    print(f"{status} {label:42s} max {'rel ' if relative else 'abs '}diff {worst:.2e}")
    assert worst <= tol, label


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def database() -> Path:
    built = ROOT / "data" / "processed" / "pharma_inhalation.sqlite"
    if built.exists():
        return built
    shipped = ROOT / "app" / "pharma_inhalation.sqlite.gz.b64.txt"
    copy = Path(tempfile.gettempdir()) / "pharma_inhalation_app_copy.sqlite"
    copy.write_bytes(gzip.decompress(base64.b64decode(shipped.read_bytes())))
    return copy


def main() -> None:
    con = sqlite3.connect(database())
    df = pd.read_sql_query(f"""SELECT route_group, CAST(num_ro5_violations AS TEXT) AS ro5_class,
                                      {', '.join(VARIABLES)} FROM inhalation_dataset""", con)
    # Some correlations and the PCA need missing values: blank 5% of alogp and psa at random.
    rng = np.random.default_rng(7)
    for v in ("alogp", "psa"):
        df.loc[rng.random(len(df)) < 0.05, v] = np.nan
    payload = {
        "columns": {v: [None if pd.isna(x) else float(x) for x in df[v]] for v in VARIABLES},
        "groups": df["route_group"].tolist(),
        "classes": df["ro5_class"].tolist(),
    }
    out = subprocess.run(["node", str(ROOT / "tests" / "stats_runner.js")], input=json.dumps(payload),
                         capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f"stats_runner.js failed:\n{out.stderr}")
    js = json.loads(out.stdout)
    print(f"rows {len(df)}, variables {len(VARIABLES)}")

    # Descriptive statistics
    for v in VARIABLES:
        x = df[v].dropna().to_numpy()
        d = js["describe"][v]
        check(f"describe {v}", [d["mean"], d["sd"], d["q1"], d["median"], d["q3"]],
              [x.mean(), x.std(ddof=1), *np.percentile(x, [25, 50, 75])])

    # Correlations (pandas uses pairwise-complete observations)
    check("pearson matrix", js["pearson"], df[VARIABLES].corr("pearson").to_numpy())
    check("spearman matrix", js["spearman"], df[VARIABLES].corr("spearman").to_numpy())

    # Mann-Whitney inhaled vs oral, Holm correction
    p_py = []
    for v, r in zip(VARIABLES, js["mw"]):
        a = df.loc[df.route_group == "inhaled", v].dropna()
        b = df.loc[df.route_group == "oral", v].dropna()
        res = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic", use_continuity=True)
        check(f"mann-whitney U {v}", r["U"], res.statistic)
        check(f"mann-whitney p {v}", r["p"], res.pvalue, TOL_P, relative=True)
        p_py.append(res.pvalue)
    check("holm adjusted p", js["holm"], holm(np.array(p_py)), TOL_P, relative=True)

    # Kruskal-Wallis on psa by number of Lipinski violations (0-4: five groups)
    groups = [df.loc[df.ro5_class == s, "psa"].dropna() for s in js["classesLabels"]]
    kw = stats.kruskal(*groups)
    check(f"kruskal H (psa, {len(groups)} groups)", js["kw"]["H"], kw.statistic)
    check(f"kruskal p (psa, {len(groups)} groups)", js["kw"]["p"], kw.pvalue, TOL_P, relative=True)
    cases = [(1, 0.5), (1, 3.84), (2, 5.99), (4, 9.49), (4, 2), (10, 25), (0.5, 0.1), (20, 60)]
    check("chi-square tail (Kruskal p-value)", js["chi2Sf"], [stats.chi2.sf(x, df) for df, x in cases],
          1e-10, relative=True)

    # PCA on complete rows, standardised: eigen-decomposition of the correlation matrix
    complete = df[VARIABLES].dropna().to_numpy()
    assert js["pca"]["n"] == len(complete)
    z = (complete - complete.mean(0)) / complete.std(0, ddof=1)
    values, vectors = np.linalg.eigh(np.cov(z, rowvar=False))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    flip = np.sign(vectors[np.abs(vectors).argmax(0), range(vectors.shape[1])])
    vectors = vectors * flip
    check("pca eigenvalues", js["pca"]["eigenvalues"], values)
    check("pca explained variance", js["pca"]["explained"], values / values.sum())
    check("pca coefficients", js["pca"]["coefficients"], vectors)
    check("pca loadings", js["pca"]["loadings"], vectors * np.sqrt(values))
    check("pca scores", js["pca"]["scores"], z @ vectors)
    # Cross-check: loadings equal the correlation between each variable and each score
    corr = np.array([[np.corrcoef(complete[:, j], (z @ vectors)[:, c])[0, 1] for c in range(len(VARIABLES))]
                     for j in range(len(VARIABLES))])
    check("pca loadings = corr(variable, score)", js["pca"]["loadings"], corr, 1e-8)

    # PCA on the covariance matrix (not standardised)
    cv, cvec = np.linalg.eigh(np.cov(complete, rowvar=False))
    check("pca (covariance) explained variance", js["pcaRaw"]["explained"], np.sort(cv)[::-1] / cv.sum())
    print("All checks passed.")


if __name__ == "__main__":
    main()
