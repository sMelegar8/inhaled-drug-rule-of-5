"""Step 8 - Audit the FDA -> ChEMBL name matching.

The whole inhaled group rests on one join that has no shared identifier behind it:
an FDA ingredient name is matched to a ChEMBL name or synonym. If that join is wrong,
every number in the Rule of 5 case study is wrong, so "92.3% matched" is a coverage
figure, not a correctness figure. This script estimates correctness.

It does four things, in order of how much they can prove:

  1. Falsification checks that need no human judgement.
     - Ambiguity: src/02_build_database.py builds its index with `setdefault`, so a name
       known to several molecules silently resolves to whichever was inserted first. This
       counts the names where that happened and whether the FDA actually uses them.
     - Provenance: whether each match came from a ChEMBL preferred name or from a
       synonym, and of which type. An FDA ingredient name is an INN or USAN, so a match
       made only through a TRADE_NAME synonym is a different kind of claim and is flagged.
     - Collapse: several ingredient names mapping to one parent molecule, which is correct
       for salts of one drug and suspicious otherwise.
  2. A full census of the inhaled side rather than a sample. The analysis set has 46
     inhaled molecules, few enough to check every one instead of estimating, and the
     census also lists every ingredient name on an inhalation product that did NOT match,
     because those are drugs potentially missing from the group.
  3. A classification of the unmatched names, so the unmatched share becomes a set of
     understood reasons instead of one opaque number.
  4. A stratified review sheet (a random sample of exact matches plus every salt-stripped
     one) written to reports/, with the evidence a person needs to judge each row.

Output: reports/name_match_audit.csv (the review sheet),
reports/name_match_inhaled_census.csv, reports/name_match_unmatched.csv.
Usage:  py -3.13 src/audit_name_matching.py      (after 02_build_database.py)
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib

build = importlib.import_module("02_build_database")  # reuse the real normalise/strip_salts, never a copy

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "processed" / "pharma_inhalation.sqlite"
REPORTS = ROOT / "reports"
SAMPLE = 50
SEED = 0
# Separators the FDA uses inside a single ingredient field for combination products.
COMBINATION = re.compile(r"\s(\|\||AND|WITH|/)\s|,\s")
PHARMACOPOEIA = re.compile(r",?\s+(USP|NF|BP|EP|JP|USP/NF)$")
VERDICTS = ROOT / "data" / "name_match_verdicts.csv"


def index_provenance(con: sqlite3.Connection) -> tuple[dict[str, list], dict[str, str]]:
    """Rebuild the match index, keeping every candidate instead of only the winner.

    02_build_database.py keeps the first candidate per name. Keeping all of them is what
    makes the silent ambiguity visible.
    """
    candidates: dict[str, list[tuple[str, str]]] = {}
    for cid, name in con.execute(
            "SELECT chembl_id, pref_name FROM molecules WHERE max_phase = 4 AND pref_name IS NOT NULL"):
        candidates.setdefault(build.normalise(name), []).append((cid, "pref_name"))
    for cid, name, syn_type in con.execute("SELECT chembl_id, synonym, syn_type FROM molecule_synonyms"):
        candidates.setdefault(build.normalise(name), []).append((cid, syn_type or "OTHER"))
    # The winner is the first candidate, exactly as the build script resolves it.
    winner = {name: c[0][1] for name, c in candidates.items()}
    return candidates, winner


def classify_unmatched(name: str) -> str:
    # "CARBON DIOXIDE, USP" is one substance with a pharmacopoeia suffix, not a combination.
    name = PHARMACOPOEIA.sub("", build.normalise(name)).strip()
    if COMBINATION.search(name):
        return "combination of several ingredients in one field"
    words = name.split()
    if build.strip_salts(name) != build.normalise(name):
        return "salt form whose parent is also unknown to ChEMBL"
    if any(w in name for w in ("VACCINE", "IMMUNE GLOBULIN", "ANTITOXIN", "TOXOID", "ALLERGENIC")):
        return "biological product"
    if name.endswith(("MAB", "CEPT", "ASE")) or "PEG" in words:
        return "protein, antibody or enzyme"
    if len(words) >= 4:
        return "long descriptive name, probably not a single substance"
    return "single name not found in ChEMBL"


def main() -> None:
    con = sqlite3.connect(DB)
    candidates, winner = index_provenance(con)
    mapping = pd.read_sql_query("SELECT * FROM fda_ingredient_map", con)
    molecules = pd.read_sql_query(
        "SELECT chembl_id, pref_name, max_phase, molecule_type, mw_freebase FROM molecules", con).set_index("chembl_id")
    atc = pd.read_sql_query(
        "SELECT chembl_id, GROUP_CONCAT(level5) AS atc FROM molecule_atc GROUP BY chembl_id", con).set_index("chembl_id")
    in_set = set(pd.read_sql_query("SELECT chembl_id FROM inhalation_dataset", con).chembl_id)

    def describe(row) -> dict:
        """The evidence needed to judge one mapping row."""
        name = row.ingredient_name
        key = name if name in candidates else build.strip_salts(name)
        cands = candidates.get(key, [])
        return {
            "ingredient_name": name,
            "match_method": row.match_method,
            "matched_via": winner.get(key, "-"),
            "n_candidates": len(cands),
            "other_candidates": "; ".join(sorted({c for c, _ in cands[1:]})) or "",
            "chembl_id": row.chembl_id or "",
            "chembl_pref_name": molecules.pref_name.get(row.chembl_id, ""),
            "parent_chembl_id": row.parent_chembl_id or "",
            "parent_pref_name": molecules.pref_name.get(row.parent_chembl_id, ""),
            "parent_mw": molecules.mw_freebase.get(row.parent_chembl_id, ""),
            "atc_codes": atc.atc.get(row.parent_chembl_id, ""),
        }

    # ---- 1. Falsification checks -------------------------------------------------
    used = {n for n in mapping.ingredient_name} | {build.strip_salts(n) for n in mapping.ingredient_name}
    ambiguous = {n: c for n, c in candidates.items() if len({cid for cid, _ in c}) > 1}
    ambiguous_used = {n: c for n, c in ambiguous.items() if n in used}
    print(f"Ambiguity: {len(ambiguous):,} names in the index belong to more than one molecule; "
          f"{len(ambiguous_used)} of them are actually used by an FDA ingredient name.")

    matched = mapping[mapping.match_method != "unmatched"].copy()
    evidence = pd.DataFrame([describe(r) for r in matched.itertuples()])
    by_source = evidence.matched_via.value_counts()
    print("\nWhat each match was made against:")
    for source, n in by_source.items():
        print(f"  {source:<14} {n:5,}  ({100 * n / len(evidence):4.1f}%)")
    trade_only = evidence[evidence.matched_via == "TRADE_NAME"]
    print(f"\nMatched only through a trade-name synonym: {len(trade_only)} "
          f"({100 * len(trade_only) / len(evidence):.1f}%) - these are the ones worth reading.")
    if len(trade_only):
        print(trade_only[["ingredient_name", "chembl_pref_name"]].head(15).to_string(index=False))

    collapse = evidence[evidence.parent_chembl_id != ""].groupby("parent_chembl_id").ingredient_name.agg(list)
    many = collapse[collapse.map(len) >= 4]
    print(f"\nParent molecules reached by 4 or more different ingredient names: {len(many)} "
          f"(expected for salts and esters of one drug):")
    for parent, names in many.head(5).items():
        print(f"  {molecules.pref_name.get(parent, parent)}: {', '.join(names[:6])}")

    # ---- 2. Full census of the inhaled side --------------------------------------
    inhaled_names = pd.read_sql_query("""
        SELECT DISTINCT i.ingredient_name
        FROM fda_products AS p
        JOIN fda_product_ingredients AS i
          ON i.application_number = p.application_number AND i.product_number = p.product_number
        WHERE UPPER(p.route) LIKE '%INHALATION%'""", con)
    census = mapping[mapping.ingredient_name.isin(inhaled_names.ingredient_name)]
    census_rows = pd.DataFrame([describe(r) for r in census.itertuples()])
    census_rows["in_analysis_set"] = census_rows.parent_chembl_id.isin(in_set)
    census_rows = census_rows.sort_values(["match_method", "ingredient_name"])
    census_rows.to_csv(REPORTS / "name_match_inhaled_census.csv", index=False)
    kept = census_rows[census_rows.in_analysis_set]
    print(f"\nInhaled census: {len(census_rows)} ingredient names appear on an inhalation product; "
          f"{(census_rows.match_method != 'unmatched').sum()} matched, "
          f"{(census_rows.match_method == 'unmatched').sum()} did not; "
          f"{kept.parent_chembl_id.nunique()} distinct molecules reached the analysis set.")
    # Every inhaled match was read and judged by hand once; data/name_match_verdicts.csv
    # records that review. Comparing against it turns a one-off audit into a regression
    # test: a new ChEMBL release that silently re-points a match fails here.
    verdicts = pd.read_csv(VERDICTS).set_index("ingredient_name")
    now = census_rows[census_rows.match_method != "unmatched"].set_index("ingredient_name")
    changed = [(n, verdicts.expected_parent[n], now.parent_pref_name[n]) for n in now.index.intersection(verdicts.index)
               if verdicts.expected_parent[n] != now.parent_pref_name[n]]
    new, gone = now.index.difference(verdicts.index), verdicts.index.difference(now.index)
    print(f"\nAgainst the reviewed verdicts ({len(verdicts)} inhaled matches checked by hand): "
          f"{len(changed)} changed, {len(new)} new, {len(gone)} no longer present.")
    for name, was, is_now in changed:
        print(f"  CHANGED {name}: reviewed as {was}, now {is_now} - re-review before trusting the inhaled group")
    for name in new:
        print(f"  NEW {name} -> {now.parent_pref_name[name]} - not covered by the review")
    for name in gone:
        print(f"  GONE {name} - reviewed but no longer on an inhalation product")

    missing = census_rows[census_rows.match_method == "unmatched"]
    # What each unmatched inhaled name actually is, taken from the FDA's own dosage_form
    # field rather than from reading the name: a gas is removed by cleaning rule M3 anyway,
    # so an unmatched gas costs the analysis nothing.
    forms = pd.read_sql_query("""
        SELECT i.ingredient_name, GROUP_CONCAT(DISTINCT p.dosage_form) AS dosage_forms,
               COUNT(DISTINCT p.application_number) AS applications
        FROM fda_product_ingredients AS i
        JOIN fda_products AS p ON p.application_number = i.application_number
                              AND p.product_number = i.product_number
        WHERE UPPER(p.route) LIKE '%INHALATION%' GROUP BY 1""", con).set_index("ingredient_name")
    print(f"\nUnmatched names on inhalation products ({len(missing)}) - each one is a drug that "
          f"could be missing from the inhaled group:")
    for r in missing.itertuples():
        f = forms.loc[r.ingredient_name]
        print(f"  {r.ingredient_name:<22} dosage form {f.dosage_forms:<6} ({f.applications} applications)  "
              f"-> {classify_unmatched(r.ingredient_name)}")
    gas = {forms.dosage_forms[n] for n in missing.ingredient_name}
    print(f"  All of them have dosage form {gas}, which cleaning rule M3 removes in any case."
          if gas == {"GAS"} else f"  WARNING: not all are gases ({gas}) - one may be a real drug missing from the group.")

    # ---- 3. Why the unmatched are unmatched --------------------------------------
    un = mapping[mapping.match_method == "unmatched"].copy()
    un["reason"] = un.ingredient_name.map(classify_unmatched)
    un[["ingredient_name", "reason"]].sort_values(["reason", "ingredient_name"]).to_csv(
        REPORTS / "name_match_unmatched.csv", index=False)
    print(f"\nWhy the {len(un)} unmatched names did not match:")
    for reason, n in un.reason.value_counts().items():
        print(f"  {n:4}  {reason}")

    # ---- 4. Review sheet ---------------------------------------------------------
    salt = evidence[evidence.match_method == "salt_stripped"]
    exact = evidence[evidence.match_method == "exact"].sample(n=SAMPLE, random_state=SEED)
    sheet = pd.concat([salt, exact, census_rows.drop(columns=["in_analysis_set"])]).drop_duplicates("ingredient_name")
    sheet.insert(0, "verdict", "")
    sheet.to_csv(REPORTS / "name_match_audit.csv", index=False)
    print(f"\nReview sheet: reports/name_match_audit.csv - {len(salt)} salt-stripped, "
          f"{len(exact)} random exact matches and the whole inhaled census, "
          f"{len(sheet)} rows in total, with an empty verdict column to fill in.")
    con.close()


if __name__ == "__main__":
    main()
