"""Step 4 - Map the PDE4 chemical space and group compounds by substructure.

Pipeline (each step is explained on app/case-study-pde4.html):
1. Compounds: every PDE4 compound with a structure (SMILES) and a clean
   potency in activity_potency; best_p is its highest median pIC50/pKi.
2. Fingerprints: Morgan (ECFP4-like, radius 2, 2048 bits). Each bit records
   whether a small circular substructure is present.
3. Dimensionality reduction: fingerprints are L2-normalised and reduced to 50
   principal components (PCA), which keeps the main structural variation and
   makes k-means faster and less noisy.
4. Model selection: k-means for k = 4..16, scored by inertia (elbow),
   silhouette and stability (mean adjusted Rand index over 5 random seeds).
5. Clustering: k-means with K clusters (chosen for stability and readability).
6. Map: t-SNE turns the 50 components into 2 coordinates for plotting.
7. Description: per cluster, the most enriched substructure motif, potency,
   properties, first year, PDE4B/PDE4D selectivity, the medoid structure and
   the named compounds.

Output: tables pde4_chemical_space, pde4_clusters, pde4_k_selection (database
and app copy), app/pde4-map.json for the interactive map, figures in
reports/figures/ and app/figures/.
Usage:  py -3.13 src/pde4_chemical_space.py      (after cleaning.py)
"""

from __future__ import annotations

import base64
import gzip
import itertools
import json
import re
import shutil
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "processed" / "pharma_inhalation.sqlite"
APP = ROOT / "app"
FIGURES = ROOT / "reports" / "figures"
DOCS_CACHE = ROOT / "data" / "raw" / "chembl_pde4_documents.json"

K = 8                     # chosen number of clusters (see pde4_k_selection)
K_RANGE = range(4, 17, 2)
N_COMPONENTS = 50
SEED = 0
POTENT = 8.0              # pIC50 >= 8 means IC50 <= 10 nM

# Substructures used to describe clusters (SMARTS). Names are what a medicinal
# chemist would call them; the first matching rule with the highest enrichment labels a cluster.
MOTIFS = {
    "catechol diether": "c1ccc(O[CX4])c(O[CX4])c1",
    "3,5-dichloropyridine": "Clc1cncc(Cl)c1",
    "pyridine N-oxide": "[n+][O-]",
    "difluoromethoxy aryl": "[CX4](F)(F)[OX2]c",
    "quinoline": "c1ccc2ncccc2c1",
    "pyridazinone": "O=c1cccn[nX3]1",
    "phthalazinone": "O=c1[nX3]ncc2ccccc12",
    "benzoxaborole": "[#5]1[#8][#6]c2ccccc21",
    "phthalimide": "O=C1c2ccccc2C(=O)N1",
    "xanthine (purine-dione)": "O=c1[nX3]c(=O)c2[nX3]c[nX2]c2[nX3]1",
    "sulfonamide": "S(=O)(=O)[NX3]",
    "carboxylic acid": "[CX3](=O)[OX2H1]",
    "ester": "[#6][CX3](=O)[OX2][#6;!$(C=O)]",
    "pyrazole": "c1cn[nX3]c1",
    "pyrimidine": "c1cncnc1",
    "thiophene": "c1ccsc1",
    "piperidine/piperazine": "[NX3]1CC[N,C]CC1",
    "indole": "c1ccc2[nX3]ccc2c1",
}
# Chemist-readable names, assigned after reading each cluster's motif profile,
# medoid and most frequent Murcko scaffolds. Keyed by the medoid's ChEMBL ID, so
# a rerun that produces different clusters falls back to the automatic motif label.
CLUSTER_NAMES = {
    "CHEMBL5561298": "Catechol diethers, newer variants",
    "CHEMBL4637027": "Diverse heteroaryl amides",
    "CHEMBL3545215": "Purine and cAMP-like heterocycles",
    "CHEMBL255610": "Classic catechol diethers (rolipram type)",
    "CHEMBL4764438": "Aza-bicyclic carboxamides",
    "CHEMBL4534321": "Roflumilast type (difluoromethoxy catechol)",
    "CHEMBL313982": "Aryl-tetrahydrophthalazinones",
    "CHEMBL2385565": "Anilino-aryl-pyrimidines",
}
CLUSTER_COLORS =["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def load(con: sqlite3.Connection) -> pd.DataFrame:
    d = pd.read_sql_query("""
        SELECT ap.molecule_chembl_id, m.pref_name, m.canonical_smiles,
               MAX(ap.p_median) AS best_p,
               MAX(CASE WHEN ap.target_name = 'PDE4B' THEN ap.p_median END) AS p_pde4b,
               MAX(CASE WHEN ap.target_name = 'PDE4D' THEN ap.p_median END) AS p_pde4d,
               m.mw_freebase, m.alogp, m.heavy_atoms
        FROM activity_potency AS ap
        JOIN molecules AS m ON m.chembl_id = ap.molecule_chembl_id
        WHERE m.canonical_smiles IS NOT NULL
        GROUP BY ap.molecule_chembl_id
    """, con)
    years = pd.read_sql_query("""
        SELECT molecule_chembl_id, MIN(document_year) AS first_year,
               COUNT(DISTINCT document_chembl_id) AS n_documents
        FROM activities_clean GROUP BY molecule_chembl_id
    """, con)
    d = d.merge(years, on="molecule_chembl_id", how="left")
    d["selectivity_bd"] = d["p_pde4b"] - d["p_pde4d"]
    d["ligand_efficiency"] = 1.37 * d["best_p"] / d["heavy_atoms"]
    d["lipophilic_efficiency"] = d["best_p"] - d["alogp"]
    # Molecules reported in papers whose title is about inhaled PDE4 inhibitors (same rule as explore_chemistry.py).
    titles = json.loads(DOCS_CACHE.read_text(encoding="utf-8")) if DOCS_CACHE.exists() else {}
    inhaled_docs = [doc for doc, j in titles.items()
                    if re.search(r"inhal|nebuli|aerosol|lung retention|dry powder", j.get("title") or "", re.I)]
    inhaled = set()
    if inhaled_docs:
        marks = ",".join("?" * len(inhaled_docs))
        inhaled = {r[0] for r in con.execute(
            f"SELECT DISTINCT molecule_chembl_id FROM activities_clean WHERE document_chembl_id IN ({marks})", inhaled_docs)}
    d["inhaled_paper"] = d["molecule_chembl_id"].isin(inhaled).astype(int)
    return d


def fingerprints(smiles: pd.Series) -> tuple[np.ndarray, list]:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in mols):
        raise ValueError("unparsable SMILES in the PDE4 set")
    return np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32), mols


def select_k(Z: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(Z), size=min(3000, len(Z)), replace=False)
    rows = []
    for k in K_RANGE:
        labels = [KMeans(n_clusters=k, n_init=10, random_state=s).fit(Z) for s in range(5)]
        ari = np.mean([adjusted_rand_score(a.labels_, b.labels_) for a, b in itertools.combinations(labels, 2)])
        rows.append({"k": k, "inertia": labels[0].inertia_,
                     "silhouette": silhouette_score(Z[sample], labels[0].labels_[sample]), "stability_ari": ari})
        print(f"  k={k:2d} inertia={labels[0].inertia_:8.0f} silhouette={rows[-1]['silhouette']:.3f} stability={ari:.2f}")
    return pd.DataFrame(rows)


def describe_clusters(d: pd.DataFrame, mols: list, Z: np.ndarray, centers: np.ndarray) -> pd.DataFrame:
    patterns = {name: Chem.MolFromSmarts(s) for name, s in MOTIFS.items()}
    has = pd.DataFrame({name: [int(m.HasSubstructMatch(p)) for m in mols] for name, p in patterns.items()})
    overall = has.mean()
    rows, used = [], {}
    for c in sorted(d["cluster"].unique()):
        idx = np.where(d["cluster"].to_numpy() == c)[0]
        g = d.iloc[idx]
        share = has.iloc[idx].mean()
        enrichment = (share / overall.clip(lower=1e-6)).where(share >= 0.4)
        motif = enrichment.idxmax() if enrichment.notna().any() else None
        medoid = idx[np.argmin(((Z[idx] - centers[c - 1]) ** 2).sum(axis=1))]
        label = CLUSTER_NAMES.get(d.iloc[medoid]["molecule_chembl_id"]) or motif or "mixed scaffolds"
        used[label] = used.get(label, 0) + 1
        if used[label] > 1:
            label = f"{label} ({used[label]})"
        named = g[g["pref_name"].notna()].sort_values("best_p", ascending=False)
        sel = g["selectivity_bd"].dropna()
        rows.append({
            "cluster": int(c), "cluster_label": label, "top_motif": motif,
            "motif_share": float(share[motif]) if motif else None, "n_molecules": len(g),
            "median_p": g["best_p"].median(), "share_potent": float((g["best_p"] >= POTENT).mean()),
            "median_mw": g["mw_freebase"].median(), "median_alogp": g["alogp"].median(),
            "median_le": g["ligand_efficiency"].median(), "median_first_year": g["first_year"].median(),
            "n_selectivity": len(sel), "median_selectivity_bd": sel.median() if len(sel) else None,
            "n_inhaled_paper": int(g["inhaled_paper"].sum()),
            "medoid_chembl_id": d.iloc[medoid]["molecule_chembl_id"], "medoid_smiles": d.iloc[medoid]["canonical_smiles"],
            "named_compounds": ", ".join(named["pref_name"].str.title().head(8)),
            "_medoid_index": int(medoid), "_motifs": {k: round(float(v), 3) for k, v in share.sort_values(ascending=False).head(4).items()},
        })
    return pd.DataFrame(rows)


def figures(d: pd.DataFrame, clusters: pd.DataFrame, ksel: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    (APP / "figures").mkdir(exist_ok=True)
    style = {"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e1e0d9", "axes.axisbelow": True}
    with plt.rc_context(style):
        # Model selection
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, col, title in [(axes[0], "inertia", "Elbow: inertia"), (axes[1], "silhouette", "Silhouette (higher = better separated)"),
                               (axes[2], "stability_ari", "Stability over 5 seeds (ARI)")]:
            ax.plot(ksel["k"], ksel[col], marker="o", color="#2a78d6", linewidth=2)
            ax.axvline(K, color="#eb6834", linestyle="--", linewidth=1)
            ax.set_xlabel("number of clusters k"); ax.set_title(title, fontsize=11)
        fig.tight_layout(); fig.savefig(FIGURES / "pde4_k_selection.png", dpi=140); plt.close(fig)

        # Map by cluster
        fig, ax = plt.subplots(figsize=(9, 7))
        for _, c in clusters.iterrows():
            g = d[d["cluster"] == c["cluster"]]
            ax.scatter(g["map_x"], g["map_y"], s=7, alpha=0.7, color=CLUSTER_COLORS[(c["cluster"] - 1) % 8],
                       label=f"{c['cluster']} · {c['cluster_label']} (n={c['n_molecules']})")
            ax.annotate(str(c["cluster"]), (g["map_x"].median(), g["map_y"].median()), fontsize=13, fontweight="bold",
                        ha="center", va="center", bbox={"boxstyle": "circle", "fc": "white", "ec": "#52514e"})
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title("PDE4 chemical space: t-SNE map coloured by k-means cluster")
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5), markerscale=2.5)
        fig.tight_layout(); fig.savefig(FIGURES / "pde4_map_clusters.png", dpi=140); plt.close(fig)

        # Map by potency
        fig, ax = plt.subplots(figsize=(8, 6.6))
        order = d.sort_values("best_p")
        sc = ax.scatter(order["map_x"], order["map_y"], c=order["best_p"], cmap="Blues", s=7, vmin=4, vmax=10)
        fig.colorbar(sc, ax=ax, label="best pIC50 / pKi")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Same map coloured by potency")
        fig.tight_layout(); fig.savefig(FIGURES / "pde4_map_potency.png", dpi=140); plt.close(fig)

        # Potency and timeline by cluster
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
        ids = clusters["cluster"].tolist()
        axes[0].boxplot([d.loc[d["cluster"] == c, "best_p"] for c in ids], tick_labels=[str(c) for c in ids], patch_artist=True,
                        boxprops={"facecolor": "#cde2fb", "color": "#2a78d6"}, medianprops={"color": "#0b0b0b"}, showfliers=False)
        axes[0].axhline(POTENT, color="#eb6834", linestyle="--", linewidth=1)
        axes[0].set_xlabel("cluster"); axes[0].set_ylabel("best pIC50 / pKi"); axes[0].set_title("Potency by cluster", fontsize=11)
        bins = [1979, 2000, 2008, 2014, 2026]
        period = pd.cut(d["first_year"], bins, labels=["1980-2000", "2001-2008", "2009-2014", "2015-2025"])
        share = pd.crosstab(period, d["cluster"], normalize="index")
        bottom = np.zeros(len(share))
        for c in ids:
            axes[1].bar(share.index.astype(str), share[c].to_numpy(), bottom=bottom, color=CLUSTER_COLORS[(c - 1) % 8], label=str(c))
            bottom += share[c].to_numpy()
        axes[1].set_ylabel("share of new compounds"); axes[1].set_title("Which chemotypes were made when", fontsize=11)
        axes[1].legend(title="cluster", frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
        fig.tight_layout(); fig.savefig(FIGURES / "pde4_cluster_potency_timeline.png", dpi=140); plt.close(fig)
    for f in ["pde4_k_selection.png", "pde4_map_clusters.png", "pde4_map_potency.png", "pde4_cluster_potency_timeline.png"]:
        shutil.copy(FIGURES / f, APP / "figures" / f)


def main() -> None:
    con = sqlite3.connect(DB)
    d = load(con)
    print(f"{len(d)} PDE4 compounds with a structure and a clean potency")
    X, mols = fingerprints(d["canonical_smiles"])
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(Xn)
    Z = pca.transform(Xn)
    print(f"PCA: {N_COMPONENTS} components keep {pca.explained_variance_ratio_.sum():.1%} of the fingerprint variance")
    print("Model selection:")
    ksel = select_k(Z)
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Z)
    # Number clusters 1..K by size, largest first, so labels are stable to read.
    order = np.argsort(-np.bincount(km.labels_))
    remap = {old: new + 1 for new, old in enumerate(order)}
    d["cluster"] = [remap[l] for l in km.labels_]
    centers = km.cluster_centers_[order]
    xy = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto", random_state=SEED).fit_transform(Z)
    d["map_x"], d["map_y"] = xy[:, 0].round(3), xy[:, 1].round(3)
    clusters = describe_clusters(d, mols, Z, centers)
    d["cluster_label"] = d["cluster"].map(dict(zip(clusters["cluster"], clusters["cluster_label"])))
    d["is_medoid"] = 0
    d.loc[d.index[clusters["_medoid_index"]], "is_medoid"] = 1
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(clusters.drop(columns=["_medoid_index", "medoid_smiles"]).to_string(index=False))

    con.executescript((ROOT / "sql" / "chemical_space_schema.sql").read_text(encoding="utf-8"))
    cols = ["molecule_chembl_id", "pref_name", "cluster", "cluster_label", "map_x", "map_y", "best_p", "p_pde4b", "p_pde4d",
            "selectivity_bd", "mw_freebase", "alogp", "heavy_atoms", "ligand_efficiency", "lipophilic_efficiency",
            "first_year", "n_documents", "inhaled_paper", "is_medoid"]
    d[cols].to_sql("pde4_chemical_space", con, if_exists="append", index=False)
    clusters.drop(columns=["_medoid_index", "_motifs"]).to_sql("pde4_clusters", con, if_exists="append", index=False)
    ksel.to_sql("pde4_k_selection", con, if_exists="append", index=False)
    con.commit()
    con.execute("VACUUM")
    con.close()
    (APP / "pharma_inhalation.sqlite.gz.b64.txt").write_bytes(base64.b64encode(gzip.compress(DB.read_bytes(), compresslevel=9)))

    points = [[float(r.map_x), float(r.map_y), int(r.cluster), round(float(r.best_p), 2),
               r.pref_name if isinstance(r.pref_name, str) else "",
               r.molecule_chembl_id, r.canonical_smiles, int(r.inhaled_paper),
               None if pd.isna(r.first_year) else int(r.first_year)] for r in d.itertuples()]
    (APP / "pde4-map.json").write_text(json.dumps({
        "k": K, "n": len(d), "pca_variance": round(float(pca.explained_variance_ratio_.sum()), 3),
        "clusters": json.loads(clusters.drop(columns=["_medoid_index"]).to_json(orient="records")),
        "k_selection": json.loads(ksel.to_json(orient="records")),
        "fields": ["x", "y", "cluster", "best_p", "name", "chembl_id", "smiles", "inhaled_paper", "first_year"],
        "points": points,
    }, separators=(",", ":")), encoding="utf-8")
    figures(d, clusters, ksel)
    print("Saved tables, app/pde4-map.json and figures.")


if __name__ == "__main__":
    main()
