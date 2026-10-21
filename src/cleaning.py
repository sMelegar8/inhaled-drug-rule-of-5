"""Step 3 - Clean the data and record every exclusion in a ledger.

Two datasets are cleaned with explicit rules, applied in order:
- activities: PDE4 IC50/Ki measurements -> activities_clean (one row per
  measurement, with pIC50/pKi) and activity_potency (median per molecule,
  target and measurement type).
- molecules: approved drugs with FDA oral or inhaled products ->
  inhalation_dataset (lung-targeted inhaled drugs vs oral drugs).

Each exclusion rule only sees the rows left by the previous rules, so every
excluded record has exactly one reason. The ledger is saved as two tables:
cleaning_log (one row per rule, with counts) and cleaning_exclusions (one row
per excluded record). Raw tables are never modified.

Usage:  py -3.13 src/cleaning.py      (or run notebooks/02_data_cleaning.ipynb)
"""

from __future__ import annotations

import base64
import gzip
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "processed" / "pharma_inhalation.sqlite"
APP_DB = ROOT / "app" / "pharma_inhalation.sqlite.gz.b64.txt"
REPORTS = ROOT / "reports"

# ChemAxon descriptors (cx_logp, cx_logd, pKa, molecular_species) are not
# returned by the ChEMBL 37 API, so they are left out of the analysis set.
PROPERTY_COLUMNS = ["mw_freebase", "alogp", "psa", "hbd", "hba", "rtb",
                    "aromatic_rings", "heavy_atoms", "num_ro5_violations", "qed_weighted"]
CORE_PROPERTIES = ["mw_freebase", "alogp", "psa", "hbd", "hba"]
P_RANGE = (3.0, 12.0)          # pX between 1 mM and 1 pM
PCHEMBL_TOLERANCE = 0.05       # log units
INCONSISTENT_RANGE = 1.0       # log units = 10-fold between replicates


@dataclass
class Ledger:
    """Ordered record of what each cleaning rule did to one dataset."""

    dataset: str
    id_column: str
    steps: list[dict] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)

    def _log(self, rule_id, kind, rule, reason, unit, before, removed, after, flagged=0):
        self.steps.append({
            "dataset": self.dataset, "step": len(self.steps), "rule_id": rule_id, "kind": kind,
            "rule": rule, "reason": reason, "unit": unit, "rows_before": before,
            "rows_removed": removed, "rows_after": after, "rows_flagged": flagged,
        })

    def start(self, df: pd.DataFrame, rule: str, unit: str) -> None:
        self._log(f"{self.dataset[0].upper()}0", "input", rule, "Starting data", unit, len(df), 0, len(df))

    def exclude(self, df: pd.DataFrame, rule_id: str, rule: str, reason: str,
                mask: pd.Series, unit: str) -> pd.DataFrame:
        mask = mask.fillna(False).astype(bool)
        ids = df.loc[mask, self.id_column].astype(str)
        self.excluded += [{"dataset": self.dataset, "record_id": i, "rule_id": rule_id} for i in ids]
        self._log(rule_id, "exclude", rule, reason, unit, len(df), int(mask.sum()), int((~mask).sum()))
        return df.loc[~mask].copy()

    def note(self, df: pd.DataFrame, rule_id: str, kind: str, rule: str, reason: str,
             unit: str, rows_before: int | None = None, flagged: int = 0) -> None:
        """Log a step that changes or checks rows without excluding any."""
        before = len(df) if rows_before is None else rows_before
        self._log(rule_id, kind, rule, reason, unit, before, 0, len(df), flagged)

    def log_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def exclusions_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.excluded, columns=["dataset", "record_id", "rule_id"])


# ---------------------------------------------------------------- activities

def load_activities(con: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("""
        SELECT a.*, t.pref_name AS target_pref_name, t.target_type
        FROM activities AS a
        JOIN targets AS t ON t.target_chembl_id = a.target_chembl_id
    """, con)
    df["target_name"] = (df["target_pref_name"]
                         .str.replace("3',5'-cyclic-AMP phosphodiesterase ", "PDE", regex=False)
                         .replace({"Phosphodiesterase 4": "PDE4 family"}))
    return df


def clean_activities(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Ledger]:
    led = Ledger("activities", "activity_id")
    unit = "measurements"
    led.start(raw, "IC50 and Ki measurements on human PDE4 downloaded from ChEMBL", unit)
    df = raw.copy()

    df = led.exclude(df, "A1", "Measurements flagged by ChEMBL curators",
                     "The data_validity_comment field marks a suspect value (units, transcription, out of range).",
                     df["data_validity_comment"].notna(), unit)
    df = led.exclude(df, "A2", "Relation other than '='",
                     "Censored values (<, >) only say that potency is above or below a threshold: they are not a point estimate.",
                     df["standard_relation"].fillna("").ne("="), unit)
    df = led.exclude(df, "A3", "Standard unit other than nM",
                     "Percentages or missing units cannot be converted into a concentration.",
                     df["standard_units"].fillna("").ne("nM"), unit)
    df = led.exclude(df, "A4", "Missing or non-positive value",
                     "The logarithm needs a numeric value greater than zero.",
                     ~(df["standard_value"] > 0), unit)

    df["p_activity"] = 9 - np.log10(df["standard_value"])
    df = led.exclude(df, "A5", f"pIC50/pKi outside {P_RANGE[0]:g}-{P_RANGE[1]:g}",
                     "A potency weaker than 1 mM or stronger than 1 pM is almost always a unit error.",
                     ~df["p_activity"].between(*P_RANGE), unit)

    diff = (df["p_activity"] - df["pchembl_value"]).abs()
    df["pchembl_mismatch"] = (diff > PCHEMBL_TOLERANCE).astype(int)
    led.note(df, "A6", "check", "Comparison with the pChEMBL value calculated by ChEMBL",
             f"Flags measurements where the calculated pX differs from pChEMBL by more than {PCHEMBL_TOLERANCE}.",
             unit, flagged=int(df["pchembl_mismatch"].sum()))

    keys = ["molecule_chembl_id", "target_chembl_id", "target_name", "standard_type"]
    potency = (df.groupby(keys)
                 .agg(p_median=("p_activity", "median"), p_min=("p_activity", "min"),
                      p_max=("p_activity", "max"), n_measurements=("p_activity", "size"),
                      n_documents=("document_chembl_id", "nunique"))
                 .reset_index())
    potency["p_range"] = potency["p_max"] - potency["p_min"]
    potency["inconsistent"] = (potency["p_range"] > INCONSISTENT_RANGE).astype(int)
    led.note(potency, "A7", "aggregate", "Median of repeated measurements",
             "One row per molecule × target × measurement type; the median is robust to outliers.",
             "molecule-target pairs", rows_before=len(df))
    led.note(potency, "A8", "check", f"Replicates more than {INCONSISTENT_RANGE:g} log unit apart",
             "Pairs whose measurements differ more than 10-fold: kept but flagged.",
             "molecule-target pairs", flagged=int(potency["inconsistent"].sum()))

    clean = df[["activity_id", "molecule_chembl_id", "target_chembl_id", "target_name", "target_type",
                "assay_chembl_id", "assay_type", "document_chembl_id", "document_year",
                "standard_type", "standard_value", "p_activity", "pchembl_value", "pchembl_mismatch"]]
    potency = potency[keys + ["p_median", "p_min", "p_max", "p_range", "n_measurements",
                              "n_documents", "inconsistent"]]
    return clean, potency, led


# ----------------------------------------------------------------- molecules

def load_routes(con: sqlite3.Connection) -> pd.DataFrame:
    """One row per FDA product and parent molecule, with the raw route text."""
    return pd.read_sql_query("""
        SELECT m.parent_chembl_id AS chembl_id, p.application_number, p.product_number, p.route
        FROM fda_products AS p
        JOIN fda_product_ingredients AS i
          ON i.application_number = p.application_number AND i.product_number = p.product_number
        JOIN fda_ingredient_map AS m ON m.ingredient_name = i.ingredient_name
        WHERE m.parent_chembl_id IS NOT NULL
    """, con)


def normalise_routes(products: pd.DataFrame) -> pd.DataFrame:
    """Split "INHALATION, ORAL" into single upper-case route tokens."""
    tokens = products.assign(route=products["route"].fillna("").str.upper().str.split(","))
    tokens = tokens.explode("route")
    tokens["route"] = tokens["route"].str.strip()
    tokens = tokens[tokens["route"] != ""]
    return (tokens.drop_duplicates(["chembl_id", "application_number", "product_number", "route"])
                  .groupby(["chembl_id", "route"]).size().rename("n_products").reset_index())


def load_atc(con: sqlite3.Connection) -> pd.Series:
    """ATC codes per parent molecule, including the codes stored on its salt forms."""
    atc = pd.read_sql_query("""
        SELECT m.parent_chembl_id AS chembl_id, ma.level5
        FROM molecule_atc AS ma JOIN molecules AS m ON m.chembl_id = ma.chembl_id
    """, con)
    return atc.groupby("chembl_id")["level5"].agg(lambda s: ",".join(sorted(set(s))))


def _atc_all(codes: pd.Series, prefixes: tuple[str, ...]) -> pd.Series:
    """True when the molecule has ATC codes and every one starts with a prefix."""
    return codes.apply(lambda c: bool(c) and all(x.startswith(prefixes) for x in c.split(",")))


def _atc_any(codes: pd.Series, prefixes: tuple[str, ...]) -> pd.Series:
    return codes.apply(lambda c: any(x.startswith(prefixes) for x in c.split(",")) if c else False)


def clean_molecules(con: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame, Ledger]:
    products = load_routes(con)
    routes = normalise_routes(products)
    raw_inhaled = products.loc[products["route"].eq("INHALATION"), "chembl_id"].nunique()

    pivot = (routes[routes["route"].isin(["INHALATION", "ORAL"])]
             .pivot_table(index="chembl_id", columns="route", values="n_products", fill_value=0)
             .rename(columns={"INHALATION": "n_inhaled_products", "ORAL": "n_oral_products"})
             .reset_index())
    pivot.columns.name = None
    mols = pd.read_sql_query(
        f"SELECT chembl_id, pref_name, molecule_type, first_approval, {', '.join(PROPERTY_COLUMNS)} FROM molecules", con)
    df = pivot.merge(mols, on="chembl_id", how="left")
    df["atc_codes"] = df["chembl_id"].map(load_atc(con)).fillna("")
    df["route_group"] = np.where(df["n_inhaled_products"] > 0, "inhaled", "oral")
    df["also_oral"] = ((df["n_inhaled_products"] > 0) & (df["n_oral_products"] > 0)).astype(int)

    led = Ledger("molecules", "chembl_id")
    unit = "molecules"
    led.start(df, "Parent molecules with at least one oral or inhaled FDA product", unit)
    led.note(df, "M0a", "transform", "Routes of administration split and upper-cased",
             f"\"INHALATION, ORAL\" and \"Inhalation, Oral\" count for both routes: inhaled molecules go "
             f"from {raw_inhaled} (exact text INHALATION only) to {int((df['n_inhaled_products'] > 0).sum())}.", unit)
    led.note(df, "M0b", "transform", "ATC codes inherited from salts",
             "Many ATC codes are stored on the salt (e.g. tiotropium bromide): they are assigned to the parent molecule.", unit)
    led.note(df, "M0c", "transform", "Exclusive groups: inhaled takes precedence over oral",
             "A molecule with inhaled products goes into the inhaled group (also_oral = 1 if it also exists as an oral product); "
             "the oral group contains only molecules that are never inhaled.", unit, flagged=int(df["also_oral"].sum()))

    df = led.exclude(df, "M1", "Not a small molecule",
                     "Proteins and mixtures have no comparable calculated properties.",
                     df["molecule_type"].ne("Small molecule"), unit)
    df = led.exclude(df, "M2", "Core chemical properties missing",
                     f"All are required: {', '.join(CORE_PROPERTIES)}. They are missing for ions and noble gases.",
                     df[CORE_PROPERTIES].isna().any(axis=1), unit)
    df = led.exclude(df, "M3", "Medical gases",
                     "Molecular weight below 100 Da or ATC V03AN (oxygen, nitrogen, nitrous oxide...): not designed drugs.",
                     (df["mw_freebase"] < 100) | _atc_any(df["atc_codes"], ("V03AN",)), unit)
    df = led.exclude(df, "M4", "Diagnostics only",
                     "All ATC codes are V04/V08/V09 (challenge tests, contrast agents, radiopharmaceuticals).",
                     _atc_all(df["atc_codes"], ("V04", "V08", "V09")), unit)
    inhaled = df["route_group"].eq("inhaled")
    df = led.exclude(df, "M5", "Inhaled drugs acting systemically on the nervous system",
                     "First-level ATC N (volatile anaesthetics, loxapine, levodopa): they are inhaled to reach "
                     "the blood, the opposite of a drug meant to stay in the lung.",
                     inhaled & _atc_any(df["atc_codes"], ("N",)), unit)
    oral = df["route_group"].eq("oral")
    df = led.exclude(df, "M6", "Oral drugs acting locally in the gut",
                     "All ATC codes are A06/A07 (laxatives, intestinal anti-infectives and anti-inflammatories): "
                     "they are designed not to be absorbed, so they do not represent systemic oral drugs.",
                     oral & _atc_all(df["atc_codes"], ("A06", "A07")), unit)

    columns = ["chembl_id", "pref_name", "route_group", "also_oral", "n_inhaled_products", "n_oral_products",
               "atc_codes", "first_approval"] + PROPERTY_COLUMNS
    df = df[columns].sort_values(["route_group", "pref_name"]).reset_index(drop=True)
    for group in ("inhaled", "oral"):
        led.note(df[df["route_group"] == group], f"M7{group[0]}", "check", f"Final set: {group} group",
                 "Molecules available for the statistical comparison.", unit,
                 rows_before=int((df["route_group"] == group).sum()))
    return df, routes, led


# -------------------------------------------------------------------- output

def save_outputs(con: sqlite3.Connection, ledgers: list[Ledger], tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Write cleaned tables and the ledger to SQLite, CSV report and the app copy."""
    con.executescript((ROOT / "sql" / "clean_schema.sql").read_text(encoding="utf-8"))
    log = pd.concat([l.log_frame() for l in ledgers], ignore_index=True)
    exclusions = pd.concat([l.exclusions_frame() for l in ledgers], ignore_index=True)
    for name, frame in {"cleaning_log": log, "cleaning_exclusions": exclusions, **tables}.items():
        frame.to_sql(name, con, if_exists="append", index=False)
    con.commit()
    con.execute("VACUUM")

    REPORTS.mkdir(exist_ok=True)
    log.to_csv(REPORTS / "cleaning_log.csv", index=False, encoding="utf-8")
    APP_DB.write_bytes(base64.b64encode(gzip.compress(DB.read_bytes(), compresslevel=9)))
    return log


def main() -> None:
    con = sqlite3.connect(DB)
    activities, potency, led_a = clean_activities(load_activities(con))
    dataset, routes, led_m = clean_molecules(con)
    log = save_outputs(con, [led_a, led_m], {
        "activities_clean": activities, "activity_potency": potency,
        "molecule_routes_clean": routes, "inhalation_dataset": dataset,
    })
    con.close()
    with pd.option_context("display.width", 160, "display.max_colwidth", 60):
        print(log[["rule_id", "kind", "rule", "rows_before", "rows_removed", "rows_after", "rows_flagged"]].to_string(index=False))


if __name__ == "__main__":
    main()
