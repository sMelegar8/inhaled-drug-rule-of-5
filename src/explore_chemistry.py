"""Exploration - five structural trends in inhaled drugs and PDE4 chemistry.

T1 Fluorine: oral drugs became fluorinated over time, inhaled drugs did not
   (logistic model fluorine ~ year * route, likelihood-ratio test).
T2 Permanent positive charge: quaternary ammonium enrichment in inhaled drugs
   (Fisher exact, Holm-corrected across a panel of structural motifs).
T3 PDE4B over PDE4D: isoform selectivity measured in the same paper, by year
   (compound-level and paper-level Spearman).
T4 Scaffold migration: share of PDE4 papers built on the catechol diether
   (rolipram / roflumilast pharmacophore), by era (paper-level Spearman).
T5 Inhaled PDE4 programmes: compounds from papers whose title mentions
   inhalation vs all other PDE4 papers (Mann-Whitney, rank-biserial r).

Exploratory: associations on public data, not causal claims. Structural motifs
come from RDKit SMARTS on ChEMBL canonical SMILES.

Output: printed results, reports/chemistry_trends.csv, reports/figures/chemistry_trends.png
Paper titles are cached in data/raw/chembl_pde4_documents.json (ChEMBL API).
Usage:  py -3.13 src/explore_chemistry.py
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from scipy import stats
from scipy.optimize import minimize

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "processed" / "pharma_inhalation.sqlite"
DOCS_CACHE = ROOT / "data" / "raw" / "chembl_pde4_documents.json"
REPORTS = ROOT / "reports"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"

MOTIFS = {
    "quaternary_ammonium": "[NX4+;H0;!$([N+]~[O-])]",
    "ester": "[#6][CX3](=O)[OX2][#6;!$(C=O)]",
    "fluorine": "[F]",
    "chlorine": "[Cl]",
    "amide": "[CX3](=O)[NX3]",
    "carboxylic_acid": "[CX3](=O)[OX2H,OX1-]",
    "sulfonamide": "S(=O)(=O)N",
    "thiophene": "c1ccsc1",
    "basic_aliphatic_amine": "[NX3;+0;!$(N-[#6,#16,#15]=[O,S,N]);!$(N-a);!$(N-[#7,#8]);!$(N-C#N)][CX4]",
    "phenol": "c[OX2H]",
}
CATECHOL_DIETHER = "c1ccc(O[CX4])c(O[CX4])c1"
ERAS = [1900, 1980, 2000, 2030]
PDE4_ERAS = [1985, 2000, 2008, 2014, 2030]


def holm(p: np.ndarray) -> np.ndarray:
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def has(smarts: str):
    pattern = Chem.MolFromSmarts(smarts)
    return lambda mol: int(mol is not None and mol.HasSubstructMatch(pattern))


def logistic_loglik(X: np.ndarray, y: np.ndarray) -> float:
    nll = lambda b: -np.sum(y * (X @ b) - np.logaddexp(0, X @ b))
    return -minimize(nll, np.zeros(X.shape[1]), method="BFGS").fun


def load_titles(doc_ids: list[str]) -> dict[str, str]:
    cache = json.loads(DOCS_CACHE.read_text(encoding="utf-8")) if DOCS_CACHE.exists() else {}
    for doc in [d for d in doc_ids if d not in cache]:
        j = requests.get(f"{CHEMBL}/document/{doc}.json", params={"only": "title,journal,year,doc_type"}, timeout=60).json()
        cache[doc] = j
    DOCS_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return {d: (cache[d].get("title") or "") for d in doc_ids}


def main() -> None:
    con = sqlite3.connect(DB)
    summary = []

    # ---------------- approved drugs: inhaled vs oral
    drugs = pd.read_sql_query("""SELECT i.chembl_id, i.pref_name, i.route_group, i.first_approval, m.canonical_smiles
                                 FROM inhalation_dataset i JOIN molecules m USING (chembl_id)""", con)
    mols = drugs["canonical_smiles"].apply(Chem.MolFromSmiles)
    for name, smarts in MOTIFS.items():
        drugs[name] = mols.apply(has(smarts))
    drugs["fsp3"] = mols.apply(rdMolDescriptors.CalcFractionCSP3)
    inh, oral = drugs[drugs.route_group == "inhaled"], drugs[drugs.route_group == "oral"]

    # T2 motif panel with Holm correction
    panel = []
    for name in MOTIFS:
        a, b = int(inh[name].sum()), int(oral[name].sum())
        odds, p = stats.fisher_exact([[a, len(inh) - a], [b, len(oral) - b]])
        panel.append({"motif": name, "inhaled": f"{a}/{len(inh)}", "oral": f"{b}/{len(oral)}",
                      "share_inhaled": a / len(inh), "share_oral": b / len(oral), "odds_ratio": odds, "p": p})
    panel = pd.DataFrame(panel)
    panel["p_holm"] = holm(panel["p"].to_numpy())
    print("Motif panel, inhaled vs oral (Fisher exact, Holm):")
    print(panel.sort_values("p").to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    q = panel.set_index("motif").loc["quaternary_ammonium"]
    print("  quaternary ammonium inhaled:", ", ".join(inh.loc[inh.quaternary_ammonium == 1, "pref_name"].str.title()))
    summary.append({"trend": "T2 quaternary ammonium enriched in inhaled drugs",
                    "effect": f"{q.share_inhaled:.1%} vs {q.share_oral:.1%}, OR {q.odds_ratio:.1f}",
                    "p": q.p_holm, "test": "Fisher exact, Holm over 10 motifs"})

    # T1 fluorine over time, interaction test
    dated = drugs[drugs.first_approval.notna()].copy()
    dated["era"] = pd.cut(dated.first_approval, ERAS, labels=["<1980", "1980-1999", "2000+"])
    f_era = dated.groupby(["era", "route_group"], observed=True)["fluorine"].agg(["sum", "size", "mean"]).unstack()
    print("\nT1 fluorine share by era:\n", f_era.round(3))
    y = dated["fluorine"].to_numpy()
    decade = (dated["first_approval"].to_numpy() - 1990) / 10
    is_inh = (dated["route_group"] == "inhaled").to_numpy(float)
    full = np.column_stack([np.ones_like(decade), decade, is_inh, decade * is_inh])
    lr = 2 * (logistic_loglik(full, y) - logistic_loglik(full[:, :3], y))
    p_lr = stats.chi2.sf(lr, 1)
    print(f"  interaction year x route: LR chi2 = {lr:.2f}, p = {p_lr:.4f}")
    print("  fluorinated inhaled drugs:", ", ".join(inh.loc[inh.fluorine == 1, "pref_name"].str.title()))
    last = f_era.loc["2000+"]
    summary.append({"trend": "T1 inhaled drugs did not follow the fluorination trend",
                    "effect": f"approved 2000+: inhaled {int(last[('sum', 'inhaled')])}/{int(last[('size', 'inhaled')])}, "
                              f"oral {int(last[('sum', 'oral')])}/{int(last[('size', 'oral')])} fluorinated",
                    "p": p_lr, "test": "logistic LR test, year x route interaction"})

    # Supporting check: Fsp3 (reported, not one of the five)
    u = stats.mannwhitneyu(inh.fsp3, oral.fsp3)
    print(f"\n  Fsp3 inhaled {inh.fsp3.median():.2f} vs oral {oral.fsp3.median():.2f}, "
          f"r = {2 * u.statistic / (len(inh) * len(oral)) - 1:+.2f}, p = {u.pvalue:.1e}")

    # ---------------- PDE4 chemistry
    acts = pd.read_sql_query("""SELECT a.document_chembl_id AS doc, a.document_year AS year, a.molecule_chembl_id,
                                       a.target_name, a.standard_type, a.p_activity, a.target_type
                                FROM activities_clean a WHERE a.document_year IS NOT NULL""", con)

    # T3 PDE4B vs PDE4D within the same paper
    ic50 = acts[(acts.standard_type == "IC50") & (acts.target_type == "SINGLE PROTEIN")]
    paired = (ic50.groupby(["doc", "year", "molecule_chembl_id", "target_name"])["p_activity"].median()
                  .unstack("target_name").dropna(subset=["PDE4B", "PDE4D"]).reset_index())
    paired["delta_BD"] = paired["PDE4B"] - paired["PDE4D"]
    paired["era"] = pd.cut(paired.year, PDE4_ERAS)
    by_era = paired.groupby("era", observed=True).agg(pairs=("delta_BD", "size"), papers=("doc", "nunique"),
                                                      median_delta=("delta_BD", "median"),
                                                      share_B_selective=("delta_BD", lambda s: (s >= 1).mean()))
    print("\nT3 PDE4B - PDE4D (same paper):\n", by_era.round(2))
    rho_c = stats.spearmanr(paired.year, paired.delta_BD)
    per_doc = paired.groupby("doc").agg(year=("year", "first"), delta=("delta_BD", "median"))
    rho_d = stats.spearmanr(per_doc.year, per_doc.delta)
    print(f"  compound level rho = {rho_c.statistic:+.2f} (p = {rho_c.pvalue:.1e}, n = {len(paired)}); "
          f"paper level rho = {rho_d.statistic:+.2f} (p = {rho_d.pvalue:.3f}, n = {len(per_doc)})")
    summary.append({"trend": "T3 PDE4 chemistry moved towards PDE4B over PDE4D selectivity",
                    "effect": f"median pB-pD {by_era.median_delta.iloc[1]:.2f} (2001-08) -> {by_era.median_delta.iloc[-1]:.2f} (2015+); "
                              f"B-selective {by_era.share_B_selective.iloc[1]:.0%} -> {by_era.share_B_selective.iloc[-1]:.0%}",
                    "p": rho_d.pvalue, "test": f"Spearman, paper level (compound level p={rho_c.pvalue:.1e})"})

    # T4 catechol diether share of papers
    smiles = pd.read_sql_query("SELECT chembl_id AS molecule_chembl_id, canonical_smiles, mw_freebase, alogp, psa, rtb, hbd FROM molecules", con)
    comp = acts[["doc", "year", "molecule_chembl_id"]].drop_duplicates().merge(smiles, on="molecule_chembl_id")
    is_cat = has(CATECHOL_DIETHER)
    cat_by_smiles = {s: is_cat(Chem.MolFromSmiles(s)) for s in comp.canonical_smiles.dropna().unique()}
    comp["catechol_diether"] = comp.canonical_smiles.map(cat_by_smiles)
    papers = comp.groupby("doc").agg(year=("year", "first"), n=("molecule_chembl_id", "nunique"),
                                     cat_share=("catechol_diether", "mean"))
    papers["catechol_paper"] = (papers.cat_share >= 0.5).astype(int)
    papers["era"] = pd.cut(papers.year, PDE4_ERAS)
    cat_era = papers.groupby("era", observed=True).agg(papers=("n", "size"), catechol_papers=("catechol_paper", "mean"))
    rho_cat = stats.spearmanr(papers.year, papers.cat_share)
    print("\nT4 papers mostly built on the catechol diether:\n", cat_era.round(2))
    print(f"  paper level rho(year, catechol share) = {rho_cat.statistic:+.2f}, p = {rho_cat.pvalue:.4f}")
    summary.append({"trend": "T4 PDE4 papers moved away from the catechol diether pharmacophore",
                    "effect": f"papers mostly catechol diether: {cat_era.catechol_papers.iloc[1]:.0%} (2001-08) -> "
                              f"{cat_era.catechol_papers.iloc[-1]:.0%} (2015+)",
                    "p": rho_cat.pvalue, "test": f"Spearman, paper level (n={len(papers)})"})

    # T5 inhaled PDE4 programmes are molecular giants
    titles = load_titles(sorted(papers.index))
    inhaled_docs = {d for d, t in titles.items() if re.search(r"inhal|inhaled|nebuli|aerosol|lung retention|dry powder", t, re.I)}
    comp["inhaled_paper"] = comp.doc.isin(inhaled_docs)
    mol = comp.groupby("molecule_chembl_id").agg(inhaled_paper=("inhaled_paper", "max"), mw=("mw_freebase", "first"),
                                                 alogp=("alogp", "first"), psa=("psa", "first"), rtb=("rtb", "first")).dropna()
    print(f"\nT5 papers with inhalation in the title: {len(inhaled_docs)}")
    for d in sorted(inhaled_docs, key=lambda d: papers.loc[d, "year"]):
        print(f"  {int(papers.loc[d, 'year'])} {d}: {titles[d][:120]}")
    t5 = {}
    for prop in ["mw", "alogp", "psa", "rtb"]:
        a, b = mol.loc[mol.inhaled_paper, prop], mol.loc[~mol.inhaled_paper, prop]
        u = stats.mannwhitneyu(a, b)
        t5[prop] = (a.median(), b.median(), 2 * u.statistic / (len(a) * len(b)) - 1, u.pvalue)
        print(f"  {prop:6s} inhaled-paper compounds {a.median():7.1f} (n={len(a)}) vs others {b.median():7.1f} (n={len(b)}): "
              f"r = {t5[prop][2]:+.2f}, p = {u.pvalue:.1e}")
    # Paper-level check: one median per paper, so large series do not dominate.
    per_paper = comp.groupby("doc").agg(mw=("mw_freebase", "median")).dropna()
    per_paper["inhaled"] = per_paper.index.isin(inhaled_docs)
    a, b = per_paper.loc[per_paper.inhaled, "mw"], per_paper.loc[~per_paper.inhaled, "mw"]
    u_doc = stats.mannwhitneyu(a, b)
    rank = (b < a.min()).mean()
    print(f"  paper level: median MW of inhaled papers {a.median():.0f} vs {b.median():.0f} "
          f"(p = {u_doc.pvalue:.1e}); the smallest inhaled paper is above {rank:.0%} of the other papers")
    summary.append({"trend": "T5 compounds from inhaled PDE4 programmes are much larger and more flexible",
                    "effect": f"median MW {t5['mw'][0]:.0f} vs {t5['mw'][1]:.0f} Da, rotatable bonds {t5['rtb'][0]:.0f} vs {t5['rtb'][1]:.0f}, "
                              f"r(MW) = {t5['mw'][2]:+.2f}",
                    "p": u_doc.pvalue, "test": f"Mann-Whitney, paper level ({len(a)} inhaled papers vs {len(b)}; compound level p={t5['mw'][3]:.1e})"})

    REPORTS.mkdir(exist_ok=True)
    pd.DataFrame(summary).to_csv(REPORTS / "chemistry_trends.csv", index=False)
    panel.to_csv(REPORTS / "chemistry_motif_panel.csv", index=False)

    # ---------------- figure
    blue, orange, grey = "#2a78d6", "#eb6834", "#898781"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    eras = list(f_era.index.astype(str))
    xpos = np.arange(len(eras))
    ax = axes[0, 0]
    ax.bar(xpos - 0.2, f_era[("mean", "oral")] * 100, 0.4, color=grey, label="oral")
    ax.bar(xpos + 0.2, f_era[("mean", "inhaled")] * 100, 0.4, color=blue, label="inhaled")
    ax.set_xticks(xpos, eras); ax.set_ylabel("% fluorinated"); ax.set_title("T1 · Fluorination did not reach inhaled drugs")
    ax.legend(frameon=False)
    ax = axes[0, 1]
    top = panel.sort_values("p").head(6)
    yy = np.arange(len(top))
    ax.barh(yy - 0.2, top.share_oral * 100, 0.4, color=grey, label="oral")
    ax.barh(yy + 0.2, top.share_inhaled * 100, 0.4, color=blue, label="inhaled")
    ax.set_yticks(yy, top.motif.str.replace("_", " ")); ax.invert_yaxis(); ax.set_xlabel("% of molecules")
    ax.set_title("T2 · Structural motifs: inhaled vs oral"); ax.legend(frameon=False)
    ax = axes[0, 2]
    labels = [f"{int(i.left) + 1}-{int(i.right) if i.right < 2030 else ''}" for i in by_era.index]
    ax.boxplot([paired.loc[paired.era == e, "delta_BD"] for e in by_era.index], tick_labels=labels, patch_artist=True,
               boxprops={"facecolor": "#cde2fb", "color": blue}, medianprops={"color": "#0b0b0b"}, showfliers=False)
    ax.axhline(0, color=grey, linewidth=1); ax.axhline(1, color=orange, linestyle="--", linewidth=1)
    ax.set_ylabel("pIC50 PDE4B − pIC50 PDE4D"); ax.set_title("T3 · Increasingly PDE4B-selective")
    ax = axes[1, 0]
    ax.bar(labels, cat_era.catechol_papers * 100, color=blue)
    ax.set_ylabel("% of papers"); ax.set_title("T4 · PDE4 papers built on the catechol diether")
    ax = axes[1, 1]
    bins = np.linspace(150, 900, 31)
    ax.hist(mol.loc[~mol.inhaled_paper, "mw"], bins=bins, color=grey, alpha=0.7, density=True, label="other papers")
    ax.hist(mol.loc[mol.inhaled_paper, "mw"], bins=bins, color=blue, alpha=0.75, density=True, label="inhaled PDE4 papers")
    ax.set_xlabel("molecular weight (Da)"); ax.set_ylabel("density"); ax.set_title("T5 · Inhaled PDE4 compounds are giants")
    ax.legend(frameon=False)
    ax = axes[1, 2]
    ax.scatter(mol.loc[~mol.inhaled_paper, "alogp"], mol.loc[~mol.inhaled_paper, "mw"], s=6, color=grey, alpha=0.35, label="other")
    ax.scatter(mol.loc[mol.inhaled_paper, "alogp"], mol.loc[mol.inhaled_paper, "mw"], s=14, color=blue, alpha=0.8, label="inhaled")
    ax.set_xlabel("ALogP"); ax.set_ylabel("molecular weight (Da)"); ax.set_title("T5 · Larger, not only more lipophilic")
    ax.legend(frameon=False)
    for a in axes.flat:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(color="#e1e0d9", linewidth=0.8); a.set_axisbelow(True)
    fig.tight_layout()
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / "chemistry_trends.png", dpi=140)
    print("\nSummary:\n", pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
