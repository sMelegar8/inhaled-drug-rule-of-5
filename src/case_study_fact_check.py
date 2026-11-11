"""Step 7 - Fact-check the Rule of 5 case study page.

Same idea as src/pde4_fact_check.py, applied to app/case-study.html: every number
written on the page is listed in CLAIMS with the value the page states, recomputed,
and reported as PASS or FAIL. The script exits non-zero if any claim has drifted,
so a stale number on the page fails a run instead of surviving unnoticed.

Each claim records where its value is checked against:
  db       recomputed from data/processed/pharma_inhalation.sqlite, the strongest check;
  report   read from a table in reports/, written by the script named in the claim.
           This catches a page that drifted from its own analysis, but not an analysis
           that drifted from the data, so the expensive resampling results (bootstrap
           confidence intervals, permutation tests) are the ones kept there.

Usage:  py -3.13 src/case_study_fact_check.py      (after cleaning.py and the explore scripts)
"""

from __future__ import annotations

import ast
import importlib
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_inhaled_oral import compare, holm  # noqa: E402  (the published comparison, not a copy of it)
from explore_trends import respiratory_class  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "processed" / "pharma_inhalation.sqlite"
REPORTS = ROOT / "reports"
PAGE = ROOT / "app" / "case-study.html"
ALPHA = 0.05
PCA_DESCRIPTORS = ["mw_freebase", "alogp", "psa", "hbd", "hba", "rtb", "aromatic_rings", "qed_weighted"]

# (claim id, what the page says, claimed value, tolerance, source)
CLAIMS = [
    # Data
    ("n_approved_drugs", "4,225 approved drugs with calculated properties", 4225, 0, "db"),
    ("n_pde4_measurements", "11,954 IC50/Ki measurements on human PDE4", 11954, 0, "db"),
    ("n_pde4_compounds", "for 7,352 compounds", 7352, 0, "db"),
    ("n_fda_products", "51,886 FDA products", 51886, 0, "db"),
    ("match_exact", "92.3% of FDA ingredient names matched exactly", 92.3, 0.05, "db"),
    ("match_salt", "0.3% matched after removing salt words", 0.3, 0.05, "db"),
    ("match_none", "7.4% unmatched", 7.4, 0.05, "db"),
    # The name-matching audit (src/audit_name_matching.py)
    ("audit_matched", "2,695 matched ingredient names", 2695, 0, "db"),
    ("audit_inhaled_names", "82 ingredient names on an inhalation product", 82, 0, "db"),
    ("audit_inhaled_matched", "76 of them matched and were reviewed", 76, 0, "db"),
    ("audit_inhaled_unmatched", "6 unmatched inhaled names", 6, 0, "db"),
    ("audit_inhaled_gas", "all 6 carry the dosage form GAS", 6, 0, "db"),
    ("audit_verdicts", "76 reviewed verdicts on file", 76, 0, "db"),
    ("audit_verdicts_wrong", "none of them was wrong", 0, 0, "db"),
    ("audit_ambiguous", "290 ambiguous names in the index", 290, 0, "db"),
    ("audit_ambiguous_used", "25 of them used by an FDA name", 25, 0, "db"),
    ("audit_ambiguous_same", "23 resolve to the same parent molecule", 23, 0, "db"),
    ("audit_trade_name", "2 matches made only through a trade-name synonym", 2, 0, "db"),
    ("audit_un_single", "143 single names ChEMBL does not hold", 143, 0, "db"),
    ("audit_un_combo", "42 combination products in one ingredient field", 42, 0, "db"),
    ("audit_un_salt", "26 salt forms whose parent is also unknown", 26, 0, "db"),
    ("audit_un_other", "4 biologics or descriptive names", 4, 0, "db"),
    ("audit_recoverable", "stripping pharmacopoeia suffixes would recover 1 name", 1, 0, "db"),
    # Ionisation state (src/compute_charge.py)
    ("charge_perm_inh", "10.9% of inhaled drugs are permanent cations", 10.9, 0.05, "db"),
    ("charge_perm_oral", "2.2% of oral drugs are", 2.2, 0.05, "db"),
    ("charge_perm_or", "odds ratio 5.5", 5.5, 0.05, "db"),
    ("charge_perm_p", "Fisher p = 0.005", 0.005, 0.0005, "db"),
    ("charge_cation_inh", "45.7% of inhaled drugs are cations", 45.7, 0.05, "db"),
    ("charge_cation_oral", "36.4% of oral drugs are", 36.4, 0.05, "db"),
    ("charge_anion_inh", "15.2% of inhaled drugs are anions", 15.2, 0.05, "db"),
    ("charge_anion_oral", "12.2% of oral drugs are", 12.2, 0.05, "db"),
    ("charge_zwit_inh", "4.3% of inhaled drugs are zwitterions", 4.3, 0.05, "db"),
    ("charge_zwit_oral", "6.0% of oral drugs are", 6.0, 0.05, "db"),
    ("charge_neutral_inh", "23.9% of inhaled drugs are neutral", 23.9, 0.05, "db"),
    ("charge_neutral_oral", "43.2% of oral drugs are", 43.2, 0.05, "db"),
    ("charge_neutral_or", "odds ratio 0.41 for being neutral", 0.41, 0.005, "db"),
    ("charge_neutral_p", "Fisher p = 0.009", 0.009, 0.0005, "db"),
    ("charge_charged_inh", "76.1% of inhaled drugs carry a charge", 76.1, 0.05, "db"),
    ("charge_charged_oral", "56.8% of oral drugs do", 56.8, 0.05, "db"),
    ("charge_n_perm_inh", "5 of the 46 inhaled drugs", 5, 0, "db"),
    ("charge_n_perm_oral", "24 of the 1,108 oral ones", 24, 0, "db"),
    ("charge_antimusc_total", "the antimuscarinics are 6 molecules", 6, 0, "db"),
    ("charge_antimusc_quat", "5 of them are quaternary", 5, 0, "db"),
    ("charge_antimusc_left", "removing them leaves one molecule", 1, 0, "db"),
    ("charge_panel_oral", "the substructure panel reports 1.7% of oral drugs", 1.7, 0.05, "db"),
    ("charge_extra_oral", "the charge rules add 5 oral drugs the panel missed", 5, 0, "db"),
    ("charge_neutral_subset", "the 11 inhaled drugs that are fully neutral", 11, 0, "db"),
    ("charge_hbd_after", "H-bond donors go from +0.48 to +0.57", 0.57, 0.005, "db"),
    ("charge_arom_after", "aromatic rings from -0.27 to -0.32", -0.32, 0.005, "db"),
    ("charge_psa_after", "polar surface area from +0.26 to +0.33", 0.33, 0.005, "db"),
    ("charge_psa_after_p", "Holm p = 0.002 there", 0.002, 0.0005, "db"),
    ("charge_slope_after", "+0.43 logP per decade on the remaining molecules", 0.43, 0.005, "report: compute_charge.py"),
    ("charge_slope_after_n", "on 24 molecules", 24, 0, "report: compute_charge.py"),
    ("charge_slope_after_p", "permutation p = 0.009", 0.009, 0.0005, "report: compute_charge.py"),
    ("charge_b2_rho", "beta2-agonists rho = +0.66", 0.66, 0.005, "report: compute_charge.py"),
    ("charge_b2_p", "p = 0.007", 0.007, 0.0005, "report: compute_charge.py"),
    ("charge_cortico_rho", "corticosteroids rho = +0.67", 0.67, 0.005, "report: compute_charge.py"),
    ("charge_antimusc_p", "the withdrawn antimuscarinic value had p = 0.07", 0.07, 0.005, "report: compute_charge.py"),
    # Cleaning pipeline
    ("clean_start", "1,305 parent molecules at the start", 1305, 0, "db"),
    ("clean_m1", "M1 removes 37", 37, 0, "db"),
    ("clean_m2", "M2 removes 63", 63, 0, "db"),
    ("clean_m3", "M3 removes 10 medical gases", 10, 0, "db"),
    ("clean_m4", "M4 removes 12 diagnostic agents", 12, 0, "db"),
    ("clean_m5", "M5 removes 9 CNS-acting inhaled drugs", 9, 0, "db"),
    ("clean_m6", "M6 removes 20 gut-local oral drugs", 20, 0, "db"),
    ("n_inhaled", "46 lung-targeted inhaled drugs", 46, 0, "db"),
    ("n_oral", "1,108 systemic oral drugs", 1108, 0, "db"),
    ("n_also_oral_flagged", "15 molecules flagged as also oral", 15, 0, "db"),
    ("n_also_oral_kept", "11 of them still in the analysis set", 11, 0, "db"),
    ("n_pde4_clean", "PDE4 measurements 11,954 to 8,793 after cleaning", 8793, 0, "db"),
    ("n_molecules_clean", "molecules 1,305 to 1,154", 1154, 0, "db"),
    ("n_no_atc", "seven inhaled molecules have no ATC code", 7, 0, "db"),
    # Q1 - rule of five and the group comparison
    ("ro5_inhaled", "26.1% of inhaled drugs break at least one rule", 26.1, 0.05, "db"),
    ("ro5_oral", "22.8% of oral drugs break at least one rule", 22.8, 0.05, "db"),
    ("ro5_p", "the number of violations does not differ, p = 0.77", 0.77, 0.005, "db"),
    # Q1 - without the also-oral molecules
    ("sub_n", "35 inhaled drugs left without the also-oral ones", 35, 0, "db"),
    ("sub_hbd", "H-bond donors r = +0.46 there", 0.46, 0.005, "db"),
    ("sub_qed", "QED r = -0.42", -0.42, 0.005, "db"),
    ("sub_psa", "PSA r = +0.32", 0.32, 0.005, "db"),
    ("sub_arom", "aromatic rings r = -0.23", -0.23, 0.005, "db"),
    ("sub_alogp", "ALogP r = -0.10", -0.10, 0.005, "db"),
    ("sub_alogp_p", "ALogP p = 0.30 there", 0.30, 0.005, "db"),
    # Q1 - composition of the inhaled group
    ("cls_b2", "15 beta2-agonists", 15, 0, "db"),
    ("cls_cortico", "8 corticosteroids", 8, 0, "db"),
    ("cls_anti", "6 antimuscarinics", 6, 0, "db"),
    ("cls_other", "17 other inhaled drugs", 17, 0, "db"),
    ("n_extreme_logp", "six of them have an ALogP below -2", 6, 0, "db"),
    ("n_respiratory", "29 drugs whose ATC code says they treat airway disease", 29, 0, "db"),
    # Q1 - the sensitivity table
    ("resp_hbd", "respiratory only: H-bond donors +0.35", 0.35, 0.005, "db"),
    ("resp_hbd_p", "p = 0.001", 0.001, 0.0005, "db"),
    ("resp_qed", "QED -0.24", -0.24, 0.005, "db"),
    ("resp_qed_p", "p = 0.03", 0.03, 0.005, "db"),
    ("resp_arom", "aromatic rings -0.20", -0.20, 0.005, "db"),
    ("resp_arom_p", "p = 0.06", 0.06, 0.005, "db"),
    ("resp_hba", "H-bond acceptors +0.16", 0.16, 0.005, "db"),
    ("resp_hba_p", "p = 0.14", 0.14, 0.005, "db"),
    ("resp_psa", "polar surface area +0.08", 0.08, 0.005, "db"),
    ("resp_psa_p", "p = 0.49", 0.49, 0.005, "db"),
    ("resp_alogp", "ALogP +0.02", 0.02, 0.005, "db"),
    ("resp_alogp_p", "p = 0.82", 0.82, 0.005, "db"),
    # Q1 - PCA
    ("pca_n", "PCA on 1,143 molecules without the also-oral group", 1143, 0, "db"),
    ("pca_two", "two components for 72.5% of the variance", 72.5, 0.05, "db"),
    ("pca_pc1", "PC1 46.8%", 46.8, 0.05, "db"),
    ("pca_pc2", "PC2 25.7%", 25.7, 0.05, "db"),
    ("pca_pc1_inhaled", "inhaled drugs mean +1.26 on PC1", 1.26, 0.005, "db"),
    ("pca_pc1_oral", "oral drugs -0.04", -0.04, 0.005, "db"),
    ("pca_pc1_r", "r = +0.44 on PC1", 0.44, 0.005, "db"),
    ("pca_pc2_inhaled", "inhaled -1.05 on PC2", -1.05, 0.005, "db"),
    ("pca_pc2_oral", "oral +0.03", 0.03, 0.005, "db"),
    ("pca_pc2_r", "r = -0.26 on PC2", -0.26, 0.005, "db"),
    # Q2 - time trends
    ("rho_inhaled", "inhaled drugs Spearman rho = +0.51 for year vs logP", 0.51, 0.005, "report: explore_trends.py"),
    ("rho_oral", "oral drugs rho = +0.25", 0.25, 0.005, "report: explore_trends.py"),
    ("logp_decade", "+0.43 logP per decade adjusted for class", 0.43, 0.005, "report: explore_trends.py"),
    ("rtb_decade", "+1.1 rotatable bonds per decade", 1.1, 0.05, "report: explore_trends.py"),
    # The figure draws these two; the caption prints them, so they are checked like any other number.
    ("fig_slope_inhaled", "the figure's inhaled line is +0.50 logP per decade", 0.50, 0.005, "db"),
    ("fig_slope_oral", "its oral line is +0.22", 0.22, 0.005, "db"),
    ("perm_p_logp", "permutation p = 0.002 for logP", 0.002, 0.0005, "report: explore_trends.py"),
    ("albuterol_year", "albuterol approved 1981", 1981, 0, "db"),
    ("albuterol_logp", "albuterol logP 1.3", 1.3, 0.05, "db"),
    ("vilanterol_logp", "vilanterol logP 4.6", 4.6, 0.05, "db"),
    ("umeclidinium_logp", "umeclidinium logP 5.1", 5.1, 0.05, "db"),
    # Q3 - structural signatures
    ("quat_inhaled", "quaternary ammonium in 10.9% of inhaled drugs", 10.9, 0.05, "db"),
    ("quat_oral", "1.7% of oral drugs", 1.7, 0.05, "db"),
    ("quat_or", "odds ratio 7.0", 7.0, 0.05, "db"),
    ("quat_p", "Holm p = 0.017", 0.017, 0.0005, "report: explore_chemistry.py"),
    ("f_oral_n", "136 of 411 oral drugs approved since 2000 contain fluorine", 136, 0, "report: explore_chemistry.py"),
    ("f_oral_total", "of 411", 411, 0, "report: explore_chemistry.py"),
    ("f_inhaled_n", "1 of 15 inhaled drugs", 1, 0, "report: explore_chemistry.py"),
    ("f_inhaled_total", "of 15", 15, 0, "report: explore_chemistry.py"),
    ("f_p", "year x route interaction p = 0.005", 0.005, 0.0005, "report: explore_chemistry.py"),
    # Q4 - PDE4
    ("pde4_papers", "430 PDE4 papers", 430, 0, "db"),
    ("pde4_inhaled_papers", "seven PDE4 papers about inhaled administration", 7, 0, "report: explore_chemistry.py"),
    ("pde4_mw_inhaled", "median weight 571 Da in inhaled papers", 571, 0.5, "report: explore_chemistry.py"),
    ("pde4_mw_other", "393 Da in the others", 393, 0.5, "report: explore_chemistry.py"),
    ("pde4_t5_p", "p = 0.002 at paper level", 0.002, 0.0005, "report: explore_chemistry.py"),
    ("tanimilast_mw", "tanimilast weighs 688 Da", 688, 0.5, "db"),
    ("tanimilast_alogp", "tanimilast ALogP 6.3", 6.3, 0.05, "db"),
    ("roflumilast_mw", "roflumilast 403 Da", 403, 0.5, "db"),
    ("roflumilast_alogp", "roflumilast ALogP 5.0", 5.0, 0.05, "db"),
    ("pde4b_p", "the selectivity trend is borderline per paper, p = 0.051", 0.051, 0.0005, "report: explore_chemistry.py"),
    ("catechol_p", "the catechol diether fell from 48% to 28% of papers, p = 0.003", 0.003, 0.0005, "report: explore_chemistry.py"),
]


def effects_from_page() -> list[list]:
    """The EFFECTS array the page hard-codes, parsed out of the HTML."""
    block = re.search(r"const EFFECTS = (\[.*?\n\];)", PAGE.read_text(encoding="utf-8"), re.S)
    if not block:
        raise SystemExit("could not find the EFFECTS array in app/case-study.html")
    return ast.literal_eval(block.group(1).rstrip(";"))


def orphan_claims(claims: list[tuple]) -> list[str]:
    """Claims whose value no longer appears anywhere in the page.

    The weak point of checking a page against a list is the list going stale: a number
    removed from the page keeps passing forever against data nobody reads any more. This
    looks for each claimed value in the HTML, in the shapes the page writes numbers
    (thousands separators, a unicode minus, a trailing percent), and reports what it
    cannot find, as a warning rather than a failure.
    """
    text = PAGE.read_text(encoding="utf-8")
    orphans = []
    for key, _, claimed, _, _ in claims:
        v = float(claimed)
        forms = {f"{v:g}", f"{abs(v):g}", f"{v:,.0f}" if v == int(v) else f"{v:g}"}
        for d in (1, 2, 3):
            forms.add(f"{abs(v):.{d}f}")
        if not any(f in text for f in forms):
            orphans.append(key)
    return orphans


def rank_biserial(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return 2 * u.statistic / (len(a) * len(b)) - 1, u.pvalue


def recompute(con: sqlite3.Connection) -> dict:
    v: dict[str, float] = {}
    data = pd.read_sql_query("SELECT * FROM inhalation_dataset", con)
    inhaled = data[data.route_group == "inhaled"].copy()
    oral = data[data.route_group == "oral"]
    inhaled["cls"] = inhaled.apply(respiratory_class, axis=1)

    # Raw sources.
    # The molecules table also holds the PDE4 compounds, so "approved drugs" means max_phase 4.
    v["n_approved_drugs"] = con.execute("SELECT COUNT(*) FROM molecules WHERE max_phase = 4").fetchone()[0]
    v["n_pde4_measurements"] = con.execute("SELECT COUNT(*) FROM activities").fetchone()[0]
    v["n_pde4_compounds"] = con.execute("SELECT COUNT(DISTINCT molecule_chembl_id) FROM activities").fetchone()[0]
    v["n_fda_products"] = con.execute("SELECT COUNT(*) FROM fda_products").fetchone()[0]
    methods = pd.read_sql_query("SELECT match_method, COUNT(*) AS n FROM fda_ingredient_map GROUP BY match_method", con)
    total = methods.n.sum()
    share = {r.match_method: 100 * r.n / total for r in methods.itertuples()}
    v["match_exact"], v["match_salt"] = share.get("exact", 0), share.get("salt_stripped", 0)
    v["match_none"] = 100 - v["match_exact"] - v["match_salt"]

    # The name-matching audit: the same numbers src/audit_name_matching.py reports,
    # recomputed here so the page cannot drift from them.
    audit = importlib.import_module("audit_name_matching")
    candidates, _ = audit.index_provenance(con)
    mapping = pd.read_sql_query("SELECT * FROM fda_ingredient_map", con)
    v["audit_matched"] = int((mapping.match_method != "unmatched").sum())
    used = set(mapping.ingredient_name) | {audit.build.strip_salts(n) for n in mapping.ingredient_name}
    ambiguous = {n: c for n, c in candidates.items() if len({cid for cid, _ in c}) > 1}
    in_use = {n: c for n, c in ambiguous.items() if n in used}
    v["audit_ambiguous"], v["audit_ambiguous_used"] = len(ambiguous), len(in_use)
    parent_of = dict(con.execute("SELECT chembl_id, parent_chembl_id FROM molecules"))
    v["audit_ambiguous_same"] = sum(
        len({parent_of.get(cid, cid) for cid, _ in c}) == 1 for c in in_use.values())
    winner_type = {n: c[0][1] for n, c in candidates.items()}
    v["audit_trade_name"] = sum(
        winner_type.get(n if n in candidates else audit.build.strip_salts(n)) == "TRADE_NAME"
        for n in mapping[mapping.match_method != "unmatched"].ingredient_name)

    inhaled_names = pd.read_sql_query("""
        SELECT DISTINCT i.ingredient_name FROM fda_products AS p
        JOIN fda_product_ingredients AS i ON i.application_number = p.application_number
                                         AND i.product_number = p.product_number
        WHERE UPPER(p.route) LIKE '%INHALATION%'""", con).ingredient_name
    census = mapping[mapping.ingredient_name.isin(inhaled_names)]
    v["audit_inhaled_names"] = len(census)
    v["audit_inhaled_matched"] = int((census.match_method != "unmatched").sum())
    unmatched_inhaled = census[census.match_method == "unmatched"].ingredient_name
    v["audit_inhaled_unmatched"] = len(unmatched_inhaled)
    marks = ",".join("?" * len(unmatched_inhaled))
    v["audit_inhaled_gas"] = con.execute(f"""
        SELECT COUNT(*) FROM (SELECT i.ingredient_name FROM fda_product_ingredients AS i
        JOIN fda_products AS p ON p.application_number = i.application_number
                              AND p.product_number = i.product_number
        WHERE UPPER(p.route) LIKE '%INHALATION%' AND i.ingredient_name IN ({marks})
        GROUP BY 1 HAVING GROUP_CONCAT(DISTINCT p.dosage_form) = 'GAS')""",
        unmatched_inhaled.tolist()).fetchone()[0]

    reviewed = pd.read_csv(ROOT / "data" / "name_match_verdicts.csv")
    v["audit_verdicts"] = len(reviewed)
    v["audit_verdicts_wrong"] = int((reviewed.verdict != "correct").sum())

    reasons = mapping[mapping.match_method == "unmatched"].ingredient_name.map(audit.classify_unmatched).value_counts()
    v["audit_un_single"] = int(reasons.get("single name not found in ChEMBL", 0))
    v["audit_un_combo"] = int(reasons.get("combination of several ingredients in one field", 0))
    v["audit_un_salt"] = int(reasons.get("salt form whose parent is also unknown to ChEMBL", 0))
    v["audit_un_other"] = int(reasons.sum() - v["audit_un_single"] - v["audit_un_combo"] - v["audit_un_salt"])
    v["audit_recoverable"] = sum(
        audit.PHARMACOPOEIA.sub("", n).strip() != n
        and (audit.PHARMACOPOEIA.sub("", n).strip() in candidates
             or audit.build.strip_salts(audit.PHARMACOPOEIA.sub("", n).strip()) in candidates)
        for n in mapping[mapping.match_method == "unmatched"].ingredient_name)

    # Ionisation state, recomputed from the molecule_charge table written by compute_charge.py.
    charge = pd.read_sql_query("""
        SELECT d.route_group, d.pref_name, c.charge_class, c.n_quaternary_n
        FROM inhalation_dataset AS d JOIN molecule_charge AS c ON c.chembl_id = d.chembl_id""", con)
    share = 100 * pd.crosstab(charge.charge_class, charge.route_group, normalize="columns")
    for cls, key in [("permanent cation", "perm"), ("cation", "cation"), ("anion", "anion"),
                     ("zwitterion", "zwit"), ("neutral", "neutral")]:
        v[f"charge_{key}_inh"], v[f"charge_{key}_oral"] = share.loc[cls, "inhaled"], share.loc[cls, "oral"]
    v["charge_charged_inh"] = 100 - v["charge_neutral_inh"]
    v["charge_charged_oral"] = 100 - v["charge_neutral_oral"]
    counts = pd.crosstab(charge.charge_class, charge.route_group)
    v["charge_n_perm_inh"] = int(counts.loc["permanent cation", "inhaled"])
    v["charge_n_perm_oral"] = int(counts.loc["permanent cation", "oral"])
    v["charge_neutral_subset"] = int(counts.loc["neutral", "inhaled"])
    for cls, key in [("permanent cation", "perm"), ("neutral", "neutral")]:
        a = int(counts.loc[cls, "inhaled"]); b = int(counts.loc[cls, "oral"])
        table = [[a, len(charge[charge.route_group == "inhaled"]) - a],
                 [b, len(charge[charge.route_group == "oral"]) - b]]
        v[f"charge_{key}_or"], v[f"charge_{key}_p"] = stats.fisher_exact(table)
    # The substructure panel counts only sp3 quaternary nitrogen; the charge rules also
    # catch aromatic quaternary nitrogen, so the two oral figures differ by design.
    oral_charge = charge[charge.route_group == "oral"]
    v["charge_panel_oral"] = 100 * float((oral_charge.n_quaternary_n > 0).mean())
    v["charge_extra_oral"] = int(((oral_charge.charge_class == "permanent cation")
                                  & (oral_charge.n_quaternary_n == 0)).sum())
    inhaled_charge = charge[charge.route_group == "inhaled"].copy()
    inhaled_charge["cls"] = pd.read_sql_query("SELECT * FROM inhalation_dataset", con).query(
        "route_group == 'inhaled'").reset_index(drop=True).apply(respiratory_class, axis=1).to_numpy()
    anti = inhaled_charge[inhaled_charge.cls == "antimuscarinics"]
    v["charge_antimusc_total"] = len(anti)
    v["charge_antimusc_quat"] = int((anti.charge_class == "permanent cation").sum())
    v["charge_antimusc_left"] = v["charge_antimusc_total"] - v["charge_antimusc_quat"]

    # The group comparison without the permanently charged molecules.
    full = pd.read_sql_query("""
        SELECT d.*, c.charge_class FROM inhalation_dataset AS d
        JOIN molecule_charge AS c ON c.chembl_id = d.chembl_id""", con)
    kept = full[full.charge_class != "permanent cation"]
    ki, ko = kept[kept.route_group == "inhaled"], kept[kept.route_group == "oral"]
    # Holm is applied across the same ten properties as the published comparison, so the
    # corrected p of one property is only meaningful alongside the other nine.
    props = ["hbd", "qed_weighted", "aromatic_rings", "psa", "hba", "rtb", "alogp",
             "heavy_atoms", "mw_freebase", "num_ro5_violations"]
    after = {p: compare(ki, ko, p) for p in props}
    adjusted_after = dict(zip(props, holm(np.array([after[p]["p"] for p in props]))))
    v["charge_hbd_after"], v["charge_arom_after"] = after["hbd"]["r"], after["aromatic_rings"]["r"]
    v["charge_psa_after"], v["charge_psa_after_p"] = after["psa"]["r"], adjusted_after["psa"]

    # Trend values kept in the report table, since they come from a 5,000-draw permutation test.
    trend = pd.read_csv(REPORTS / "charge_trend_sensitivity.csv")
    no_perm = trend[trend.scenario == "Without permanently charged"]
    slope = no_perm[no_perm.analysis == "Class-adjusted slope"].iloc[0]
    v["charge_slope_after"], v["charge_slope_after_n"], v["charge_slope_after_p"] = (
        slope.per_decade, slope.n, slope.p)
    spearman = no_perm.set_index("group")
    v["charge_b2_rho"], v["charge_b2_p"] = spearman.rho["β2-agonists"], spearman.p["β2-agonists"]
    v["charge_cortico_rho"] = spearman.rho["corticosteroids"]
    published = trend[trend.scenario == "All inhaled drugs"].set_index("group")
    v["charge_antimusc_p"] = published.p["antimuscarinics"]

    # Cleaning ledger.
    log = pd.read_sql_query("SELECT * FROM cleaning_log WHERE dataset = 'molecules' ORDER BY step", con)
    removed = dict(zip(log.rule_id, log.rows_removed))
    v["clean_start"] = int(log.rows_after.iloc[0] + log.rows_removed.iloc[0])
    for rule in ("M1", "M2", "M3", "M4", "M5", "M6"):
        v[f"clean_{rule.lower()}"] = int(removed.get(rule, -1))
    # M6 is the last exclusion rule; M7i/M7o only split what is left into the two groups.
    v["n_molecules_clean"] = int(log.loc[log.rule_id == "M6", "rows_after"].iloc[0])
    v["n_inhaled"], v["n_oral"] = len(inhaled), len(oral)
    v["n_also_oral_kept"] = int(data.also_oral.sum())
    v["n_also_oral_flagged"] = con.execute(
        "SELECT COUNT(*) FROM molecule_routes_clean WHERE route = 'INHALATION' AND chembl_id IN "
        "(SELECT chembl_id FROM molecule_routes_clean WHERE route = 'ORAL')").fetchone()[0]
    # A6 is the last rule that removes measurements; A7 then summarises repeats by their median.
    act_log = pd.read_sql_query("SELECT * FROM cleaning_log WHERE dataset = 'activities' ORDER BY step", con)
    v["n_pde4_clean"] = int(act_log.loc[act_log.rule_id == "A6", "rows_after"].iloc[0])
    v["n_no_atc"] = int((inhaled.atc_codes.isna() | (inhaled.atc_codes == "")).sum())
    v["pde4_papers"] = con.execute("SELECT COUNT(DISTINCT document_chembl_id) FROM activities_clean").fetchone()[0]

    # Rule of five.
    v["ro5_inhaled"] = 100 * float((inhaled.num_ro5_violations > 0).mean())
    v["ro5_oral"] = 100 * float((oral.num_ro5_violations > 0).mean())
    v["ro5_p"] = stats.mannwhitneyu(inhaled.num_ro5_violations.dropna(),
                                    oral.num_ro5_violations.dropna(), alternative="two-sided").pvalue

    # Subsets of the inhaled group.
    sub = inhaled[inhaled.also_oral == 0]
    v["sub_n"] = len(sub)
    for prop, key in [("hbd", "sub_hbd"), ("qed_weighted", "sub_qed"), ("psa", "sub_psa"),
                      ("aromatic_rings", "sub_arom"), ("alogp", "sub_alogp")]:
        res = compare(sub, oral, prop)
        v[key] = res["r"]
        if key == "sub_alogp":
            v["sub_alogp_p"] = res["p"]

    counts = inhaled["cls"].value_counts()
    v["cls_b2"] = int(counts.get("β2-agonists", 0))
    v["cls_cortico"] = int(counts.get("corticosteroids", 0))
    v["cls_anti"] = int(counts.get("antimuscarinics", 0))
    v["cls_other"] = int(counts.get("other inhaled", 0))
    v["n_extreme_logp"] = int((inhaled.alogp < -2).sum())

    resp = inhaled[inhaled["cls"].isin(("β2-agonists", "antimuscarinics", "corticosteroids"))]
    v["n_respiratory"] = len(resp)
    for prop, key in [("hbd", "resp_hbd"), ("qed_weighted", "resp_qed"), ("aromatic_rings", "resp_arom"),
                      ("hba", "resp_hba"), ("psa", "resp_psa"), ("alogp", "resp_alogp")]:
        res = compare(resp, oral, prop)
        v[key], v[f"{key}_p"] = res["r"], res["p"]

    # PCA on the standardised descriptors, excluding the molecules that are also oral.
    pca_set = data[data.also_oral == 0].dropna(subset=PCA_DESCRIPTORS)
    v["pca_n"] = len(pca_set)
    X = StandardScaler().fit_transform(pca_set[PCA_DESCRIPTORS])
    pca = PCA(n_components=2, random_state=0).fit(X)
    scores = pca.transform(X)
    v["pca_pc1"], v["pca_pc2"] = 100 * pca.explained_variance_ratio_[0], 100 * pca.explained_variance_ratio_[1]
    v["pca_two"] = v["pca_pc1"] + v["pca_pc2"]
    is_inhaled = (pca_set.route_group == "inhaled").to_numpy()
    for i, name in ((0, "pc1"), (1, "pc2")):
        a, b = pd.Series(scores[is_inhaled, i]), pd.Series(scores[~is_inhaled, i])
        v[f"pca_{name}_inhaled"], v[f"pca_{name}_oral"] = a.mean(), b.mean()
        v[f"pca_{name}_r"], _ = rank_biserial(a, b)

    named = data.set_index(data.pref_name.fillna(""))
    v["albuterol_year"] = named.loc["ALBUTEROL", "first_approval"]
    v["albuterol_logp"] = named.loc["ALBUTEROL", "alogp"]
    v["vilanterol_logp"] = named.loc["VILANTEROL", "alogp"]
    v["umeclidinium_logp"] = named.loc["UMECLIDINIUM", "alogp"]

    pde4 = pd.read_sql_query(
        "SELECT pref_name, mw_freebase, alogp FROM molecules WHERE pref_name IN ('TRANIMILAST', 'ROFLUMILAST')",
        con).set_index("pref_name")
    v["tanimilast_mw"], v["tanimilast_alogp"] = pde4.loc["TRANIMILAST", "mw_freebase"], pde4.loc["TRANIMILAST", "alogp"]
    v["roflumilast_mw"], v["roflumilast_alogp"] = pde4.loc["ROFLUMILAST", "mw_freebase"], pde4.loc["ROFLUMILAST", "alogp"]

    # Motif panel, recomputed shares and odds ratio.
    motifs = pd.read_csv(REPORTS / "chemistry_motif_panel.csv").set_index("motif")
    quat = motifs.loc["quaternary_ammonium"]
    v["quat_inhaled"], v["quat_oral"] = 100 * quat.share_inhaled, 100 * quat.share_oral
    v["quat_or"] = quat.odds_ratio
    v["quat_p"] = quat.p_holm

    # The caption quotes the simple per-group regressions that explore_trends.py draws on the
    # figure, which are not the class-adjusted slope quoted in the prose: both are recomputed
    # here so the page cannot drift from the image it shows.
    fig = pd.read_sql_query("SELECT first_approval, alogp, route_group FROM inhalation_dataset "
                            "WHERE first_approval IS NOT NULL AND alogp IS NOT NULL", con)
    for group, key in (("inhaled", "fig_slope_inhaled"), ("oral", "fig_slope_oral")):
        sub = fig[fig.route_group == group]
        v[key] = np.polyfit(sub.first_approval, sub.alogp, 1)[0] * 10

    # Values kept in the report tables: resampling results and the paper-level PDE4 trends.
    trends = pd.read_csv(REPORTS / "exploration_trends.csv").set_index("property")
    v["rho_inhaled"], v["rho_oral"] = trends.loc["alogp", "rho_inhaled"], trends.loc["alogp", "rho_oral"]
    v["logp_decade"] = trends.loc["alogp", "class_adjusted_per_decade"]
    v["rtb_decade"] = trends.loc["rtb", "class_adjusted_per_decade"]
    v["perm_p_logp"] = trends.loc["alogp", "p_permutation"]

    chem = pd.read_csv(REPORTS / "chemistry_trends.csv")
    row = lambda key: chem[chem.trend.str.startswith(key)].iloc[0]
    v["f_p"] = row("T1").p
    f_counts = [int(n) for n in re.findall(r"(\d+)/(\d+)", row("T1").effect)[0]] + \
               [int(n) for n in re.findall(r"(\d+)/(\d+)", row("T1").effect)[1]]
    v["f_inhaled_n"], v["f_inhaled_total"], v["f_oral_n"], v["f_oral_total"] = f_counts
    v["pde4b_p"], v["catechol_p"], v["pde4_t5_p"] = row("T3").p, row("T4").p, row("T5").p
    t5 = row("T5").effect
    v["pde4_mw_inhaled"], v["pde4_mw_other"] = (int(n) for n in re.search(r"MW (\d+) vs (\d+) Da", t5).groups())
    v["pde4_inhaled_papers"] = int(re.search(r"(\d+) inhaled papers", row("T5").test).group(1))
    return v


def main() -> None:
    con = sqlite3.connect(DB)
    print("Recomputing every number on app/case-study.html ...")
    values = recompute(con)

    rows = []
    for key, text, claimed, tol, source in CLAIMS:
        got = float(values[key])
        rows.append({"claim": key, "page_says": text, "claimed": claimed,
                     "recomputed": round(got, 4), "source": source, "pass": abs(got - claimed) <= tol})

    # The table of effect sizes is generated from an array hard-coded in the page: check every cell.
    data = pd.read_sql_query("SELECT * FROM inhalation_dataset", con)
    inhaled, oral = data[data.route_group == "inhaled"], data[data.route_group == "oral"]
    properties = {"H-bond donors": "hbd", "QED drug-likeness": "qed_weighted", "Aromatic rings": "aromatic_rings",
                  "Polar surface area (Å²)": "psa", "H-bond acceptors": "hba", "Rotatable bonds": "rtb",
                  "ALogP": "alogp", "Heavy atoms": "heavy_atoms", "Molecular weight (Da)": "mw_freebase",
                  "Rule-of-five violations": "num_ro5_violations"}
    results = {label: compare(inhaled, oral, col) for label, col in properties.items()}
    adjusted = dict(zip(results, holm(np.array([r["p"] for r in results.values()]))))
    # A median is written at the precision the page prints, so it is checked at that precision
    # (392 must be the correct rounding of 391.51, not equal to it).
    decimals = lambda x: len(str(x).split(".")[1]) if "." in str(x) else 0
    for label, med_in, med_or, r, p_holm in effects_from_page():
        got = results[label]
        for what, claimed, recomputed, ok in [
                ("median inhaled", med_in, got["median_inhaled"], lambda c, g: round(g, decimals(c)) == c),
                ("median oral", med_or, got["median_oral"], lambda c, g: round(g, decimals(c)) == c),
                ("effect size r", r, got["r"], lambda c, g: abs(g - c) <= 0.001),
                ("Holm p", p_holm, adjusted[label], lambda c, g: abs(g - c) <= max(1e-9, 0.02 * c))]:
            rows.append({"claim": f"effects:{properties[label]}:{what.split()[-1]}",
                         "page_says": f"{label}, {what} = {claimed}", "claimed": claimed,
                         "recomputed": round(float(recomputed), 6), "source": "db",
                         "pass": bool(ok(claimed, float(recomputed)))})
    con.close()

    check = pd.DataFrame(rows)
    check.to_csv(REPORTS / "case_study_fact_check.csv", index=False)
    failed = check[~check["pass"]]
    by_source = check.source.str.split(":").str[0].value_counts().to_dict()
    print(f"\nFact check: {len(check) - len(failed)} of {len(check)} claims match "
          f"({by_source.get('db', 0)} recomputed from the database, {by_source.get('report', 0)} from report tables).")
    if len(failed):
        print("\nMISMATCHES (the page must be corrected):")
        with pd.option_context("display.width", 200, "display.max_colwidth", 55):
            print(failed[["claim", "page_says", "claimed", "recomputed"]].to_string(index=False))
    orphans = orphan_claims(CLAIMS)
    if orphans:
        print(f"\nWARNING: {len(orphans)} claim(s) check a number that is no longer on the page "
              f"(remove them or correct the value): {', '.join(orphans)}")

    print("\nSaved reports/case_study_fact_check.csv")
    if len(failed):
        raise SystemExit(f"{len(failed)} claim(s) on the page do not match the data")


if __name__ == "__main__":
    main()
