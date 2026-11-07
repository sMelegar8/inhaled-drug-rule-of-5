"""Step 7 - Audit structural charge and repeat the lipophilicity comparisons.

ALogP is a calculated structural descriptor, not logD at physiological pH.
Permanent cationic centres make a neutral-species interpretation particularly
problematic. This step flags those centres on the stored ChEMBL structures and
tests whether the published route comparison and time trends depend on them.

The five charge classes are inspectable SMARTS rules, NOT measured ionization,
pKa estimates or a speciation prediction. In particular, 'neutral' means no
listed site was found, and 'zwitterion' means both listed acid and base sites;
neither label establishes the predominant microspecies or its net charge.
Basic aliphatic amines and amidines/guanidines both count as bases in R2-R4.
No SMILES is neutralized, desalted or otherwise changed before counting.

Output: molecule_charge in SQLite, reports/charge_{classes,sensitivity,
trend_sensitivity}.csv, and the refreshed app database copy. No HTML is edited.
Usage: py -3.13 src/compute_charge.py (after cleaning.py)
"""

from __future__ import annotations

import base64
import gzip
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from scipy import stats

from compare_inhaled_oral import PROPERTIES, RESPIRATORY, compare, holm
from explore_trends import SEED, bootstrap_rho, class_adjusted_slope, respiratory_class

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "processed" / "pharma_inhalation.sqlite"
REPORTS = ROOT / "reports"
N_PERM = 5000
CLASSES = ("permanent cation", "zwitterion", "cation", "anion", "neutral")

# Each query starts at the site being counted. Symmetric matches are collapsed
# by that first atom, so guanidines count once, not once per amino nitrogen.
# RDKit's ^3 hybridization query is documented in https://rdkit.org/docs/RDKit_Book.html.
SMARTS = {
    # S1: exactly the project's explore_chemistry.MOTIFS quaternary query.
    "quaternary_n": "[NX4+;H0;!$([N+]~[O-])]",
    # S2: sp3, uncharged amines; exclude amides/thioamides/amidines,
    # sulfonamides/phosphoramides, anilines and N-cyano groups. Nitrile N is sp.
    "basic_amine": "[N;^3;X3;+0;!$(N-[#6,#16,#15]=[O,S,N]);!$(N-a);!$(N-C#N)]",
    # S3: central carbon shared by the imino and amino N; exclude acylated
    # amino N, which would extend the strong-base rule to less basic amides.
    "amidine_guanidine": "[C;X3](=[N;X2,X3;+0,+1])-[N;X3,X4;+0,+1;!$(N-C=[O,S])]",
    # S4-S9 include acid and conjugate-base representations, but not esters.
    # S4: carboxyl carbon, rather than each resonance-equivalent oxygen.
    "carboxylic_acid": "[CX3](=O)[OX2H,OX1-]",
    # S5-S6: the two aromatic tetrazole proton locations share one carbon anchor.
    "tetrazole_1h": "[c]1[nH,n-]nnn1",
    "tetrazole_2h": "[c]1n[nH,n-]nn1",
    # S7: the acidic N must have H or negative charge, excluding N-alkyl analogues.
    "acyl_sulfonamide": "[N;H1,-1]([CX3](=O))[SX4](=O)(=O)",
    # S8-S9: count the P/S acid centre once even with several acidic OH groups.
    "phosphorus_acid": "[P](=O)([OX2H,OX1-])",
    "sulfur_acid": "[S](=O)(=O)[OX2H,OX1-]",
    # S10-S12: non-protonated permanent cationic centres beyond S1.
    # Adjacent anions exclude N-oxides and charge-separated sulfoxides;
    # formal positive atoms in nitro/nitrate/azide resonance are not cations.
    "aromatic_cation": "[n+;H0;!$([n+]~[*-])]",  # S10
    "phosphonium": "[P+;X4;H0;!$([P+]~[*-])]",  # S11
    "sulfonium": "[S+;X3;H0;!$([S+]~[*-])]",  # S12
}
PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in SMARTS.items()}
if any(pattern is None for pattern in PATTERNS.values()):
    raise ValueError("Invalid charge SMARTS")

RULES = {
    "R1": "permanent cation: quaternary N or another listed permanent positive centre (takes precedence over acids)",
    "R2": "zwitterion: at least one listed base AND at least one listed acid",
    "R3": "cation: at least one listed base, no listed acid",
    "R4": "anion: at least one listed acid, no listed base",
    "R5": "neutral: no preceding rule matched (not proof of neutrality at pH 7.4)",
}


def charge_record(smiles: str) -> dict[str, int | str]:
    """Count distinct structural sites and record the first matching rule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unparseable canonical SMILES: {smiles}")
    sites = {name: {match[0] for match in mol.GetSubstructMatches(pattern)}
             for name, pattern in PATTERNS.items()}
    # Both tetrazole tautomers count their ring carbon, preventing double counts.
    acids = set().union(*(sites[name] for name in (
        "carboxylic_acid", "tetrazole_1h", "tetrazole_2h", "acyl_sulfonamide",
        "phosphorus_acid", "sulfur_acid")))
    other = set().union(*(sites[name] for name in ("aromatic_cation", "phosphonium", "sulfonium")))
    base = bool(sites["basic_amine"] or sites["amidine_guanidine"])
    if sites["quaternary_n"] or other:
        cls, rule = "permanent cation", "R1"
    elif base and acids:
        cls, rule = "zwitterion", "R2"
    elif base:
        cls, rule = "cation", "R3"
    elif acids:
        cls, rule = "anion", "R4"
    else:
        cls, rule = "neutral", "R5"
    return {"formal_charge": Chem.GetFormalCharge(mol),
            "n_quaternary_n": len(sites["quaternary_n"]),
            "n_basic_amine": len(sites["basic_amine"]),
            "n_amidine_guanidine": len(sites["amidine_guanidine"]),
            "n_acid": len(acids), "n_other_permanent": len(other),
            "charge_class": cls, "rule_applied": rule}


def sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    """Use the published tests, correcting the ten properties within each scenario."""
    subsets = {"All inhaled drugs": data,
               "Without permanently charged": data[data.charge_class != "permanent cation"],
               "Neutral molecules only": data[data.charge_class == "neutral"]}
    rows = []
    for scenario, frame in subsets.items():
        inhaled, oral = frame[frame.route_group == "inhaled"], frame[frame.route_group == "oral"]
        block = pd.DataFrame([{"scenario": scenario, "property": prop, "label": label,
                               **compare(inhaled, oral, prop), "n_oral": int(oral[prop].notna().sum())}
                              for prop, label in PROPERTIES.items()])
        block["p_holm"] = holm(block.p.to_numpy())
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def trend_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    """Repeat the original Spearman and within-class year-permutation analysis."""
    rows = []
    for scenario, frame in {
        "All inhaled drugs": data,
        "Without permanently charged": data[data.charge_class != "permanent cation"],
    }.items():
        frame = frame.dropna(subset=["first_approval", "alogp"]).copy()
        frame["cls"] = frame.apply(respiratory_class, axis=1)
        inhaled, oral = frame[frame.route_group == "inhaled"], frame[frame.route_group == "oral"]
        respiratory = inhaled[inhaled.cls.isin(RESPIRATORY)]
        groups = {"inhaled": inhaled, "oral": oral,
                  **{cls: respiratory[respiratory.cls == cls] for cls in RESPIRATORY}}
        for group, subset in groups.items():
            # One antimuscarinic remains after exclusion: its correlation is undefined.
            valid = len(subset) >= 3 and subset.first_approval.nunique() > 1 and subset.alogp.nunique() > 1
            rho, p = stats.spearmanr(subset.first_approval, subset.alogp) if valid else (np.nan, np.nan)
            rows.append({"scenario": scenario, "analysis": "Spearman", "group": group,
                         "n": len(subset), "rho": rho, "p": p,
                         "per_decade": np.nan, "n_permutations": 0,
                         "status": "ok" if valid else "insufficient observations or variation"})
        # The original script consumes bootstrap draws BEFORE its alogp permutation
        # test. Reuse that sequence to reproduce its published baseline p exactly;
        # each scenario starts at SEED=42 and follows the same procedure.
        rng = np.random.default_rng(SEED)
        for subset in (inhaled, oral):
            bootstrap_rho(subset.first_approval.to_numpy(), subset.alogp.to_numpy(), rng, 4000)
        slope, p = class_adjusted_slope(respiratory, "alogp", rng, N_PERM)
        rows.append({"scenario": scenario, "analysis": "Class-adjusted slope", "group": "respiratory",
                     "n": len(respiratory), "rho": np.nan, "p": p, "per_decade": 10 * slope,
                     "n_permutations": N_PERM, "status": "ok"})
    return pd.DataFrame(rows)


def print_audit(data: pd.DataFrame) -> None:
    """Show denominators and named cases so the classification can be challenged."""
    counts = pd.crosstab(data.charge_class, data.route_group).reindex(index=CLASSES, fill_value=0)
    print("\nCharge classes (n and % of ALL molecules in the route):")
    for route in ("inhaled", "oral"):
        total = int(counts[route].sum())
        print(f"  {route}, n={total}")
        for cls in CLASSES:
            print(f"    {cls}: {counts.loc[cls, route]} ({100 * counts.loc[cls, route] / total:.6f}%)")
        permanent = data[(data.route_group == route) & (data.charge_class == "permanent cation")]
        print("  Permanent centres:", ", ".join(permanent.pref_name.sort_values()))
        n = int(permanent.alogp.notna().sum())
        print(f"  Permanent centres WITH ALogP: {n}/{total} ({100 * n / total:.6f}%)")
    expected = {"IPRATROPIUM", "TIOTROPIUM", "ACLIDINIUM", "UMECLIDINIUM", "GLYCOPYRRONIUM"}
    found = set(data.loc[(data.route_group == "inhaled") &
                         (data.charge_class == "permanent cation"), "pref_name"])
    print("\nFive expected inhaled permanent cations:", "PASS" if found == expected else f"FAIL: {sorted(found)}")
    for name in ("ALBUTEROL", "FORMOTEROL", "SALMETEROL", "AMIKACIN", "PENTETIC ACID"):
        case = data[data.pref_name == name]
        print(case[["pref_name", "formal_charge", "n_basic_amine", "n_acid", "charge_class", "rule_applied"]]
              .to_string(index=False))
        if name in ("ALBUTEROL", "FORMOTEROL", "SALMETEROL"):
            print("  Amine sanity:", "PASS" if len(case) == 1 and case.iloc[0].charge_class == "cation" else "FAIL")


def main() -> None:
    print(f"RDKit {rdBase.rdkitVersion}; permutation seed={SEED}, n={N_PERM}")
    print("Structural rules only; no measured ionization, pKa or logD is inferred.")
    for rule, explanation in RULES.items():
        print(f"  {rule}: {explanation}")
    # Keep the source row order used by explore_trends: seeded permutations
    # depend on it even though the observed correlations and slope do not.
    with sqlite3.connect(DB) as con:
        data = pd.read_sql_query("""
            SELECT d.*, m.canonical_smiles FROM inhalation_dataset d
            LEFT JOIN molecules m ON m.chembl_id = d.chembl_id
        """, con)
    if not data.chembl_id.is_unique or data.canonical_smiles.isna().any():
        raise ValueError("Each dataset molecule must have exactly one canonical SMILES")
    charges = pd.DataFrame([charge_record(smiles) for smiles in data.canonical_smiles])
    charges.insert(0, "chembl_id", data.chembl_id)
    data = data.merge(charges, on="chembl_id", validate="one_to_one")
    if len(charges) != 1154 or data.charge_class.isna().any() or data.rule_applied.isna().any():
        raise ValueError("Expected 1,154 completely classified molecules in this dataset release")

    # Compute reports before replacing the derived table, so calculation errors
    # cannot leave the database with a partial set of classifications.
    sens = sensitivity(data)
    trends = trend_sensitivity(data)
    with sqlite3.connect(DB) as con:
        con.executescript((ROOT / "sql" / "charge_schema.sql").read_text(encoding="utf-8"))
        charges.to_sql("molecule_charge", con, if_exists="append", index=False)
        con.commit()
        con.execute("VACUUM")
    (ROOT / "app" / "pharma_inhalation.sqlite.gz.b64.txt").write_bytes(
        base64.b64encode(gzip.compress(DB.read_bytes(), compresslevel=9)))
    REPORTS.mkdir(exist_ok=True)
    cols = ["chembl_id", "pref_name", "route_group", "alogp", "canonical_smiles", *charges.columns[1:]]
    data[cols].to_csv(REPORTS / "charge_classes.csv", index=False, lineterminator="\n")
    sens.to_csv(REPORTS / "charge_sensitivity.csv", index=False, lineterminator="\n")
    trends.to_csv(REPORTS / "charge_trend_sensitivity.csv", index=False, lineterminator="\n")
    print_audit(data)
    print("\nRoute comparisons (raw p and Holm p across ten properties per scenario):")
    print(sens[["scenario", "property", "n_inhaled", "n_oral", "r", "p", "p_holm"]].to_string(index=False))
    print("\nYear vs ALogP (Spearman p is unadjusted; slope p is two-sided permutation):")
    print(trends.to_string(index=False))
    print(f"\nSaved {len(charges)} charge records, three CSV reports and the app database copy.")


if __name__ == "__main__":
    main()
