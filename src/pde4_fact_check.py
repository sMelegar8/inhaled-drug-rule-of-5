"""Step 6 - Fact-check the PDE4 case study page, and test how robust its claims are.

Part 1, fact check: every number written on app/case-study-pde4.html is listed in
CLAIMS with the value the page states. The script recomputes each one from the
database (and from the fingerprints, where the claim is about similarity) and
prints PASS or FAIL. A page is only as good as the last time somebody checked it
against the data, so this makes that check a command instead of a memory.

Part 2, robustness: the page's two strongest claims are tested the way a referee
would attack them.

  R1  Potency differs by family, but with n = 5,211 any difference is
      "significant". Reported with the effect size, and repeated after dropping
      the largest contributing paper.
  R2  Family 5 is PDE4B-selective. The published value mixes PDE4B and PDE4D
      numbers measured in different papers, under different assay conditions,
      and a few large series dominate. Recomputed two stricter ways:
        a) within-paper selectivity only (both isoforms from the same document);
        b) one value per paper instead of one per molecule;
      and then leaving out the largest contributing paper.
  R3  The families come from k-means, which imposes a grouping. Agreement with
      Ward agglomerative clustering on the same components says how much of the
      grouping is the data and how much is the algorithm.

Output: reports/pde4_fact_check.csv (one row per claim) and
reports/pde4_robustness.csv, plus everything printed.
Usage:  py -3.13 src/pde4_fact_check.py      (after pde4_chemical_space.py)
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_samples

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "processed" / "pharma_inhalation.sqlite"
REPORTS = ROOT / "reports"
SEED = 0
N_PAIRS = 200_000          # sampled molecule pairs for the Tanimoto averages
SELECTIVE_FAMILY = 5       # the family the page calls PDE4B-selective

# Every number the page states, as (claim id, what it says, value, tolerance).
# Tolerance is absolute, and matches how the page rounds the number.
CLAIMS = [
    ("n_compounds", "5,211 compounds with a structure and a clean potency", 5211, 0),
    ("n_measurements", "11,954 raw measurements", 11954, 0),
    ("n_potent", "1,183 compounds with best pIC50 >= 8", 1183, 0),
    ("n_both_isoforms", "675 measured on both PDE4B and PDE4D", 675, 0),
    ("bits_mean", "a molecule sets 53.4 of the 2,048 bits on average", 53.4, 0.05),
    ("bits_methoxyethane", "methoxyethane (CCOC) sets 8 bits", 8, 0),
    ("bits_glycol", "ethylene glycol (OCCO) sets 5 bits despite having more atoms", 5, 0),
    ("n_documents", "430 papers and patents behind the clean measurements", 430, 0),
    ("bits_roflumilast", "roflumilast sets 50 bits", 50, 0),
    ("tan_rolipram_roflumilast", "Tanimoto rolipram / roflumilast 0.154", 0.154, 0.001),
    ("tan_rolipram_piclamilast", "Tanimoto rolipram / piclamilast 0.387", 0.387, 0.001),
    ("tan_roflumilast_tanimilast", "Tanimoto roflumilast / tanimilast 0.440", 0.440, 0.001),
    ("pca_pc1", "the first component carries 5.0% of the variance", 5.0, 0.05),
    ("pca_first2", "a plot of PC1 against PC2 would show only 7.9%", 7.9, 0.05),
    ("pca_first10", "the first 10 components carry 22.7%", 22.7, 0.05),
    ("pca_variance", "50 components keep 51.4% of the variance", 51.4, 0.1),
    ("pca_first100", "100 components would keep 66.1%", 66.1, 0.05),
    ("sil_share_negative", "12.4% of molecules have a negative silhouette", 12.4, 0.05),
    ("tanimoto_c2", "family 2 is loose, mean Tanimoto 0.14", 0.14, 0.005),
    ("tanimoto_c3", "family 3 is loose, 0.13", 0.13, 0.005),
    ("tanimoto_c7", "family 7 is the tightest, 0.41", 0.41, 0.005),
    ("k_chosen", "k = 8 clusters", 8, 0),
    ("ari_k8", "stability ARI 0.75 at k = 8", 0.75, 0.005),
    ("silhouette_min", "silhouettes from 0.07", 0.07, 0.005),
    ("silhouette_max", "silhouettes to 0.15", 0.15, 0.005),
    ("tanimoto_within", "within-family Tanimoto 0.181", 0.181, 0.002),
    ("tanimoto_between", "between-family Tanimoto 0.127", 0.127, 0.002),
    ("n_scaffolds", "1,350 generic Murcko scaffolds", 1350, 0),
    ("nmi_scaffold", "NMI between clusters and scaffolds 0.42", 0.42, 0.005),
    ("kw_h", "Kruskal-Wallis H = 805 for potency by family", 805, 1.0),
    ("kw_eps2", "epsilon-squared = 0.15", 0.15, 0.005),
    ("median_p_c7", "family 7 median pIC50 7.8", 7.8, 0.05),
    ("median_p_c6", "family 6 median pIC50 7.7", 7.7, 0.05),
    ("median_p_c5", "family 5 median pIC50 7.4", 7.4, 0.05),
    ("median_p_c3", "family 3 median pIC50 5.6", 5.6, 0.05),
    ("le_c5", "family 5 median ligand efficiency 0.39", 0.39, 0.005),
    ("lle_c5", "family 5 median lipophilic efficiency 4.4", 4.4, 0.05),
    ("mw_c5", "family 5 median MW 342 Da", 342, 0.5),
    ("mw_c6", "family 6 median MW 475 Da", 475, 0.5),
    ("alogp_c6", "family 6 median ALogP 5.2", 5.2, 0.05),
    ("lle_c6", "family 6 median lipophilic efficiency 2.6", 2.6, 0.05),
    ("sel_c5", "family 5 median PDE4B - PDE4D +1.41", 1.41, 0.005),
    ("sel_others", "all other families +0.33", 0.33, 0.005),
    ("sel_r", "Mann-Whitney r = +0.58 for family 5 vs the rest", 0.58, 0.005),
    ("sel_n_c5", "217 selectivity values in family 5", 217, 0),
    ("sel_share_10x", "71% of family 5 at least ten-fold PDE4B-selective", 71, 0.5),
    ("sel_papers_c5", "those values come from 13 papers", 13, 0),
    ("zatolmilast_sel", "zatolmilast sits at -2.4", -2.4, 0.05),
    ("tanimilast_p", "tanimilast best pIC50 10.4", 10.4, 0.05),
    ("tanimilast_mw", "tanimilast 688 Da", 688, 0.5),
    ("n_inhaled", "248 compounds from inhaled-PDE4 papers", 248, 0),
    ("inhaled_c2", "of which 115 in family 2", 115, 0),
    ("inhaled_c6", "57 in family 6", 57, 0),
    ("inhaled_c5", "44 in family 5", 44, 0),
    ("inhaled_papers", "inhaled programmes identified from 7 papers", 7, 0),
    ("period_1", "947 compounds first reported 1980-2000", 947, 0),
    ("period_2", "782 compounds 2001-2008", 782, 0),
    ("period_3", "1,295 compounds 2009-2014", 1295, 0),
    ("period_4", "2,187 compounds 2015-2025", 2187, 0),
    ("share_c4_early", "family 4 is 23.8% of 1980-2000 compounds", 23.8, 0.05),
    ("share_c4_late", "and 5.4% of 2015-2025", 5.4, 0.05),
    ("share_c3_early", "family 3 is 39.3% of 1980-2000", 39.3, 0.05),
    ("share_c3_late", "and 15.8% of 2015-2025", 15.8, 0.05),
    ("share_c5_late", "family 5 is 17.1% of 2015-2025", 17.1, 0.05),
    ("share_c5_early", "and 0.0% of 1980-2000", 0.0, 0.05),
]
# Claims that are a family label rather than a number.
MEMBERSHIP = [
    ("ROFLUMILAST", 6), ("TRANIMILAST", 6), ("ORISMILAST", 6),
    ("ROLIPRAM", 4), ("CILOMILAST", 4), ("PICLAMILAST", 4),
    ("APREMILAST", 1), ("CRISABOROLE", 2), ("ZATOLMILAST", 8), ("PF-06445974", 5),
]


def fingerprints(smiles: pd.Series) -> list:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def tanimoto_means(fps: list, labels: np.ndarray) -> tuple[float, float, dict[int, float]]:
    """Mean Tanimoto of sampled pairs, within the same family and between families."""
    rng = np.random.default_rng(SEED)
    i, j = rng.integers(0, len(fps), N_PAIRS), rng.integers(0, len(fps), N_PAIRS)
    keep = i != j
    i, j = i[keep], j[keep]
    sim = np.array([DataStructs.TanimotoSimilarity(fps[a], fps[b]) for a, b in zip(i, j)])
    same = labels[i] == labels[j]
    per_cluster = {}
    for c in np.unique(labels):
        m = same & (labels[i] == c)
        per_cluster[int(c)] = float(sim[m].mean()) if m.sum() > 50 else float("nan")
    return float(sim[same].mean()), float(sim[~same].mean()), per_cluster


def rank_biserial(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return 2 * u.statistic / (len(a) * len(b)) - 1, u.pvalue


def recompute(con: sqlite3.Connection) -> tuple[dict, pd.DataFrame, list, np.ndarray]:
    """Recompute every claimed number from the database. Returns the values and the data reused later."""
    v: dict[str, float] = {}
    space = pd.read_sql_query("SELECT * FROM pde4_chemical_space", con)
    clusters = pd.read_sql_query("SELECT * FROM pde4_clusters", con)
    ksel = pd.read_sql_query("SELECT * FROM pde4_k_selection", con)
    smiles = pd.read_sql_query("""
        SELECT ps.molecule_chembl_id, m.canonical_smiles
        FROM pde4_chemical_space AS ps JOIN molecules AS m ON m.chembl_id = ps.molecule_chembl_id
        ORDER BY ps.rowid""", con)
    assert (smiles.molecule_chembl_id == space.molecule_chembl_id).all(), "row order differs"

    v["n_compounds"] = len(space)
    v["n_measurements"] = con.execute("SELECT COUNT(*) FROM activities").fetchone()[0]
    v["n_potent"] = int((space.best_p >= 8).sum())
    v["n_both_isoforms"] = int((space.p_pde4b.notna() & space.p_pde4d.notna()).sum())
    v["k_chosen"] = len(clusters)
    v["ari_k8"] = float(ksel.loc[ksel.k == 8, "stability_ari"].iloc[0])
    v["silhouette_min"], v["silhouette_max"] = ksel.silhouette.min(), ksel.silhouette.max()

    # Fingerprints: the teaching numbers, the similarity claims and the independent clustering of R3.
    fps = fingerprints(smiles.canonical_smiles)
    X = np.array([np.frombuffer(bytes(fp.ToBitString(), "ascii"), dtype=np.uint8) - 48 for fp in fps], dtype=np.float32)
    v["bits_mean"] = float(X.sum(axis=1).mean())
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    v["bits_methoxyethane"] = gen.GetFingerprint(Chem.MolFromSmiles("CCOC")).GetNumOnBits()
    v["bits_glycol"] = gen.GetFingerprint(Chem.MolFromSmiles("OCCO")).GetNumOnBits()
    v["n_documents"] = con.execute("SELECT COUNT(DISTINCT document_chembl_id) FROM activities_clean").fetchone()[0]
    by_name = dict(zip(space.pref_name, fps))
    v["bits_roflumilast"] = by_name["ROFLUMILAST"].GetNumOnBits()
    for a, b, key in [("ROLIPRAM", "ROFLUMILAST", "tan_rolipram_roflumilast"),
                      ("ROLIPRAM", "PICLAMILAST", "tan_rolipram_piclamilast"),
                      ("ROFLUMILAST", "TRANIMILAST", "tan_roflumilast_tanimilast")]:
        v[key] = DataStructs.TanimotoSimilarity(by_name[a], by_name[b])

    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    # The variance profile is a teaching number, so it is read from a 100-component fit.
    ev = PCA(n_components=100, random_state=SEED).fit(Xn).explained_variance_ratio_
    v["pca_pc1"], v["pca_first2"] = 100 * float(ev[0]), 100 * float(ev[:2].sum())
    v["pca_first10"] = 100 * float(ev[:10].sum())
    v["pca_first100"] = 100 * float(ev.sum())
    # Z is built exactly as src/pde4_chemical_space.py builds it, so the checks below
    # run on the same input the published clusters came from.
    pca = PCA(n_components=50, random_state=SEED).fit(Xn)
    Z = pca.transform(Xn)
    v["pca_variance"] = 100 * float(pca.explained_variance_ratio_.sum())
    labels = space.cluster.to_numpy()
    v["tanimoto_within"], v["tanimoto_between"], per_cluster = tanimoto_means(fps, labels)
    for c, t in per_cluster.items():
        v[f"tanimoto_c{c}"] = t

    # How many molecules sit on a border: the honest reading of a low silhouette.
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(Z), size=3000, replace=False)
    km = KMeans(n_clusters=8, n_init=10, random_state=SEED).fit(Z)
    sil = silhouette_samples(Z[sample], km.labels_[sample])
    v["sil_share_negative"] = 100 * float((sil < 0).mean())

    scaffolds = [MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s)))
                 for s in smiles.canonical_smiles]
    scaffold_smiles = [Chem.MolToSmiles(s) for s in scaffolds]
    v["n_scaffolds"] = len(set(scaffold_smiles))
    v["nmi_scaffold"] = float(normalized_mutual_info_score(scaffold_smiles, labels))

    # Potency by family.
    groups = [g.best_p.dropna() for _, g in space.groupby("cluster")]
    kw = stats.kruskal(*groups)
    v["kw_h"], v["kw_eps2"] = kw.statistic, kw.statistic / (len(space) - 1)
    for c in (3, 5, 6, 7):
        v[f"median_p_c{c}"] = space.loc[space.cluster == c, "best_p"].median()
    v["le_c5"] = space.loc[space.cluster == 5, "ligand_efficiency"].median()
    v["lle_c5"] = space.loc[space.cluster == 5, "lipophilic_efficiency"].median()
    v["mw_c5"] = space.loc[space.cluster == 5, "mw_freebase"].median()
    v["mw_c6"] = space.loc[space.cluster == 6, "mw_freebase"].median()
    v["alogp_c6"] = space.loc[space.cluster == 6, "alogp"].median()
    v["lle_c6"] = space.loc[space.cluster == 6, "lipophilic_efficiency"].median()

    # Selectivity of the family the page highlights.
    sel = space[space.selectivity_bd.notna()]
    in5, out5 = sel[sel.cluster == SELECTIVE_FAMILY], sel[sel.cluster != SELECTIVE_FAMILY]
    v["sel_c5"], v["sel_others"] = in5.selectivity_bd.median(), out5.selectivity_bd.median()
    v["sel_r"], p_sel = rank_biserial(in5.selectivity_bd, out5.selectivity_bd)
    v["sel_n_c5"] = len(in5)
    v["sel_share_10x"] = 100 * float((in5.selectivity_bd >= 1).mean())
    marks = ",".join("?" * len(in5))
    v["sel_papers_c5"] = con.execute(
        f"""SELECT COUNT(DISTINCT document_chembl_id) FROM activities_clean
            WHERE molecule_chembl_id IN ({marks}) AND target_name IN ('PDE4B', 'PDE4D')""",
        in5.molecule_chembl_id.tolist()).fetchone()[0]
    print(f"  (selectivity of family {SELECTIVE_FAMILY} vs the rest: p = {p_sel:.3g})")

    named = space.set_index(space.pref_name.fillna(""))
    v["zatolmilast_sel"] = named.loc["ZATOLMILAST", "selectivity_bd"]
    v["tanimilast_p"] = named.loc["TRANIMILAST", "best_p"]
    v["tanimilast_mw"] = named.loc["TRANIMILAST", "mw_freebase"]

    v["n_inhaled"] = int(space.inhaled_paper.sum())
    for c in (2, 5, 6):
        v[f"inhaled_c{c}"] = int(space.loc[space.cluster == c, "inhaled_paper"].sum())
    # The flag comes from paper titles, so the claim is about how many titles match.
    titles = json.loads((ROOT / "data" / "raw" / "chembl_pde4_documents.json").read_text(encoding="utf-8"))
    v["inhaled_papers"] = sum(
        bool(re.search(r"inhal|nebuli|aerosol|lung retention|dry powder", j.get("title") or "", re.I))
        for j in titles.values())

    # Timeline table.
    period = pd.cut(space.first_year, [1979, 2000, 2008, 2014, 2026],
                    labels=["1980-2000", "2001-2008", "2009-2014", "2015-2025"])
    counts = period.value_counts().sort_index()
    for i, name in enumerate(counts.index, start=1):
        v[f"period_{i}"] = int(counts[name])
    share = 100 * pd.crosstab(period, space.cluster, normalize="index")
    v["share_c4_early"], v["share_c4_late"] = share.loc["1980-2000", 4], share.loc["2015-2025", 4]
    v["share_c3_early"], v["share_c3_late"] = share.loc["1980-2000", 3], share.loc["2015-2025", 3]
    v["share_c5_early"], v["share_c5_late"] = share.loc["1980-2000", 5], share.loc["2015-2025", 5]

    print("\nMean within-family Tanimoto per family (how tight each family is):")
    print("  " + "  ".join(f"{c}: {t:.2f}" for c, t in sorted(per_cluster.items())))
    return v, space, fps, Z


def robustness(con: sqlite3.Connection, space: pd.DataFrame, Z: np.ndarray) -> pd.DataFrame:
    rows = []

    # R1 - potency by family, without the largest paper.
    docs = pd.read_sql_query("""
        SELECT molecule_chembl_id, document_chembl_id, COUNT(*) AS n
        FROM activities_clean GROUP BY molecule_chembl_id, document_chembl_id""", con)
    main_doc = docs.sort_values("n").groupby("molecule_chembl_id").last()["document_chembl_id"]
    space = space.assign(main_doc=space.molecule_chembl_id.map(main_doc))
    biggest = space.main_doc.value_counts().idxmax()
    n_biggest = int(space.main_doc.value_counts().max())
    kw_all = stats.kruskal(*[g.best_p.dropna() for _, g in space.groupby("cluster")])
    sub = space[space.main_doc != biggest]
    kw_sub = stats.kruskal(*[g.best_p.dropna() for _, g in sub.groupby("cluster")])
    rows += [{"check": "R1 potency by family", "variant": "published (all compounds)", "n": len(space),
              "effect": kw_all.statistic / (len(space) - 1), "effect_name": "epsilon-squared", "p": kw_all.pvalue},
             {"check": "R1 potency by family", "variant": f"without the largest paper ({biggest}, {n_biggest} compounds)",
              "n": len(sub), "effect": kw_sub.statistic / (len(sub) - 1), "effect_name": "epsilon-squared", "p": kw_sub.pvalue}]

    # R2 - selectivity of the highlighted family, three stricter ways.
    sel = space[space.selectivity_bd.notna()]
    r, p = rank_biserial(sel[sel.cluster == SELECTIVE_FAMILY].selectivity_bd, sel[sel.cluster != SELECTIVE_FAMILY].selectivity_bd)
    rows.append({"check": "R2 family 5 selectivity", "variant": "published (molecule level, across papers)",
                 "n": len(sel), "effect": r, "effect_name": "rank-biserial r", "p": p})

    # (a) both isoforms measured inside the same paper.
    iso = pd.read_sql_query("""
        SELECT molecule_chembl_id, document_chembl_id, target_name, AVG(p_activity) AS p
        FROM activities_clean WHERE target_name IN ('PDE4B', 'PDE4D') AND p_activity IS NOT NULL
        GROUP BY molecule_chembl_id, document_chembl_id, target_name""", con)
    wide = iso.pivot_table(index=["molecule_chembl_id", "document_chembl_id"], columns="target_name", values="p").dropna()
    within = wide.assign(sel=wide.PDE4B - wide.PDE4D).reset_index()
    within["cluster"] = within.molecule_chembl_id.map(space.set_index("molecule_chembl_id").cluster)
    within = within[within.cluster.notna()]
    r, p = rank_biserial(within[within.cluster == SELECTIVE_FAMILY].sel, within[within.cluster != SELECTIVE_FAMILY].sel)
    rows.append({"check": "R2 family 5 selectivity", "variant": "within-paper only (same document, same assay)",
                 "n": len(within), "effect": r, "effect_name": "rank-biserial r", "p": p})
    med_in = within[within.cluster == SELECTIVE_FAMILY].sel.median()
    med_out = within[within.cluster != SELECTIVE_FAMILY].sel.median()
    print(f"\nR2a within-paper selectivity: family {SELECTIVE_FAMILY} median {med_in:+.2f} "
          f"(n = {(within.cluster == SELECTIVE_FAMILY).sum()}) vs {med_out:+.2f} for the rest")

    # (b) one value per paper instead of one per molecule.
    paper = within.groupby(["document_chembl_id", within.cluster == SELECTIVE_FAMILY]).sel.median().reset_index()
    paper.columns = ["document_chembl_id", "is_family", "sel"]
    a, b = paper[paper.is_family].sel, paper[~paper.is_family].sel
    r, p = rank_biserial(a, b)
    rows.append({"check": "R2 family 5 selectivity", "variant": f"one value per paper ({len(a)} vs {len(b)} papers)",
                 "n": len(paper), "effect": r, "effect_name": "rank-biserial r", "p": p})
    print(f"R2b paper level: {len(a)} papers with family {SELECTIVE_FAMILY} compounds, median {a.median():+.2f}; "
          f"{len(b)} other papers, median {b.median():+.2f}")

    # (c) paper level, minus the paper contributing most family-5 values.
    top_doc = within[within.cluster == SELECTIVE_FAMILY].document_chembl_id.value_counts()
    drop = top_doc.idxmax()
    keep = paper[paper.document_chembl_id != drop]
    a2, b2 = keep[keep.is_family].sel, keep[~keep.is_family].sel
    r, p = rank_biserial(a2, b2)
    rows.append({"check": "R2 family 5 selectivity", "variant": f"paper level without {drop} ({top_doc.max()} values)",
                 "n": len(keep), "effect": r, "effect_name": "rank-biserial r", "p": p})

    # R3 - would another algorithm have told the same story?
    ward = AgglomerativeClustering(n_clusters=8, linkage="ward").fit_predict(Z)
    ari = adjusted_rand_score(space.cluster.to_numpy(), ward)
    nmi = normalized_mutual_info_score(space.cluster.to_numpy(), ward)
    rows.append({"check": "R3 grouping vs algorithm", "variant": "k-means vs Ward agglomerative, same 50 components",
                 "n": len(space), "effect": ari, "effect_name": "adjusted Rand index", "p": np.nan})
    # The partition is partly a choice of algorithm; the question that matters is whether
    # the finding survives it. Take the Ward cluster that best overlaps the family and
    # re-run the selectivity test on Ward's own grouping.
    overlap = pd.crosstab(space.cluster.to_numpy(), ward).loc[SELECTIVE_FAMILY]
    twin = int(overlap.idxmax())
    space_w = space.assign(ward=ward)
    sel_w = space_w[space_w.selectivity_bd.notna()]
    r, p = rank_biserial(sel_w[sel_w.ward == twin].selectivity_bd, sel_w[sel_w.ward != twin].selectivity_bd)
    rows.append({"check": "R3 grouping vs algorithm",
                 "variant": f"selectivity test on Ward's own cluster {twin}: {int((ward == twin).sum())} compounds, "
                            f"holding {overlap.max()} of {overlap.sum()} family-{SELECTIVE_FAMILY} ones",
                 "n": len(sel_w), "effect": r, "effect_name": "rank-biserial r", "p": p})
    print(f"\nR3 k-means vs Ward: ARI {ari:.2f}, NMI {nmi:.2f}. Ward cluster {twin} holds "
          f"{overlap.max()}/{overlap.sum()} of family {SELECTIVE_FAMILY}; its median selectivity is "
          f"{sel_w[sel_w.ward == twin].selectivity_bd.median():+.2f} (r = {r:+.2f}, p = {p:.3g})")
    return pd.DataFrame(rows)


def main() -> None:
    con = sqlite3.connect(DB)
    print("Recomputing every number on app/case-study-pde4.html ...")
    values, space, fps, Z = recompute(con)

    rows = []
    for key, text, claimed, tol in CLAIMS:
        got = float(values[key])
        ok = abs(got - claimed) <= tol
        rows.append({"claim": key, "page_says": text, "claimed": claimed, "recomputed": round(got, 4), "pass": ok})
    space_named = space.set_index(space.pref_name.fillna(""))
    for name, family in MEMBERSHIP:
        got = int(space_named.loc[name, "cluster"]) if name in space_named.index else -1
        rows.append({"claim": f"family_{name.lower()}", "page_says": f"{name.title()} is in family {family}",
                     "claimed": family, "recomputed": got, "pass": got == family})
    check = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    check.to_csv(REPORTS / "pde4_fact_check.csv", index=False)

    failed = check[~check["pass"]]
    print(f"\nFact check: {len(check) - len(failed)} of {len(check)} claims match the data.")
    if len(failed):
        print("\nMISMATCHES (the page must be corrected):")
        with pd.option_context("display.width", 200, "display.max_colwidth", 60):
            print(failed[["claim", "page_says", "claimed", "recomputed"]].to_string(index=False))

    print("\nRobustness checks:")
    rob = robustness(con, space, Z)
    rob.to_csv(REPORTS / "pde4_robustness.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_colwidth", 70):
        print(rob.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    con.close()
    print("\nSaved reports/pde4_fact_check.csv and reports/pde4_robustness.csv")
    if len(failed):
        raise SystemExit(f"{len(failed)} claim(s) on the page do not match the data")


if __name__ == "__main__":
    main()
