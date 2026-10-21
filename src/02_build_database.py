"""Step 2 - Load the raw files into one SQLite database.

Reads data/raw/ (written by 01_download_data.py), creates the tables in
sql/schema.sql and loads the values without cleaning them. The only
transformation is the FDA ingredient -> ChEMBL molecule name matching,
which every row records with its match_method so it can be audited.

Output: data/processed/pharma_inhalation.sqlite (rebuilt from scratch)
Usage:  py -3.13 src/02_build_database.py
"""

from __future__ import annotations

import base64
import gzip
import json
import re
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "processed" / "pharma_inhalation.sqlite"
APP_DB = ROOT / "app" / "pharma_inhalation.sqlite.gz.b64.txt"

PROPERTY_COLUMNS = [
    "full_molformula", "mw_freebase", "full_mwt", "alogp", "cx_logp", "cx_logd", "psa",
    "hba", "hbd", "rtb", "aromatic_rings", "heavy_atoms", "num_ro5_violations",
    "qed_weighted", "cx_most_apka", "cx_most_bpka", "molecular_species", "np_likeness_score",
]

# Counter-ions and hydrate words FDA appends to the active moiety name.
SALT_WORDS = {
    "HYDROCHLORIDE", "DIHYDROCHLORIDE", "HCL", "HYDROBROMIDE", "SULFATE", "BISULFATE",
    "SODIUM", "DISODIUM", "POTASSIUM", "CALCIUM", "MAGNESIUM", "BROMIDE", "CHLORIDE",
    "FUMARATE", "MALEATE", "TARTRATE", "BITARTRATE", "MESYLATE", "BESYLATE", "TOSYLATE",
    "ACETATE", "CITRATE", "PHOSPHATE", "SUCCINATE", "XINAFOATE", "TROMETHAMINE",
    "MEGLUMINE", "HYDRATE", "MONOHYDRATE", "DIHYDRATE", "TRIHYDRATE", "ANHYDROUS",
    "HEMIHYDRATE", "SESQUIHYDRATE", "LYSINE", "ARGININE",
}


def load(name: str):
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def normalise(name: str) -> str:
    return re.sub(r"\s+", " ", name.upper()).strip()


def strip_salts(name: str) -> str:
    words = normalise(name).replace(",", " ").split()
    while len(words) > 1 and words[-1] in SALT_WORDS:
        words.pop()
    return " ".join(words)


def molecule_row(mol: dict, source_set: str) -> tuple:
    props = mol.get("molecule_properties") or {}
    structures = mol.get("molecule_structures") or {}
    hierarchy = mol.get("molecule_hierarchy") or {}
    flags = [None if mol.get(k) is None else int(bool(mol[k]))
             for k in ("oral", "parenteral", "topical", "prodrug")]
    natural = mol.get("natural_product")
    return (
        mol["molecule_chembl_id"], mol.get("pref_name"), mol.get("molecule_type"),
        mol.get("max_phase"), mol.get("first_approval"), *flags,
        None if natural is None else int(natural == 1 or natural is True),
        hierarchy.get("parent_chembl_id") or mol["molecule_chembl_id"],
        structures.get("canonical_smiles"), structures.get("standard_inchi_key"),
        *[props.get(c) for c in PROPERTY_COLUMNS], source_set,
    )


def main() -> None:
    DB.parent.mkdir(parents=True, exist_ok=True)
    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    con.executescript((ROOT / "sql" / "schema.sql").read_text(encoding="utf-8"))

    # Molecules: approved drugs first, then PDE4-tested compounds.
    approved = load("chembl_approved_molecules.json")
    tested = load("chembl_pde4_compounds.json")
    tested_ids = {a["molecule_chembl_id"] for a in load("chembl_pde4_activities.json")}
    rows = [molecule_row(m, "approved+pde4_tested" if m["molecule_chembl_id"] in tested_ids else "approved")
            for m in approved]
    rows += [molecule_row(m, "pde4_tested") for m in tested]
    con.executemany(f"INSERT INTO molecules VALUES ({','.join('?' * len(rows[0]))})", rows)

    synonyms, atc_links = [], []
    for mol in approved:
        cid = mol["molecule_chembl_id"]
        synonyms += [(cid, s["molecule_synonym"], s["syn_type"]) for s in mol.get("molecule_synonyms") or []]
        atc_links += [(cid, code) for code in set(mol.get("atc_classifications") or [])]
    con.executemany("INSERT INTO molecule_synonyms VALUES (?,?,?)", synonyms)
    con.executemany("INSERT INTO molecule_atc VALUES (?,?)", atc_links)

    con.executemany("INSERT OR IGNORE INTO atc_classes VALUES (?,?,?,?,?,?,?,?,?,?)", [
        (a["level5"], a.get("who_name"), a.get("level1"), a.get("level1_description"),
         a.get("level2"), a.get("level2_description"), a.get("level3"), a.get("level3_description"),
         a.get("level4"), a.get("level4_description")) for a in load("chembl_atc_classes.json")])

    con.executemany("INSERT INTO targets VALUES (?,?,?,?)", [
        (t["target_chembl_id"], t["pref_name"], t["target_type"], t["organism"])
        for t in load("chembl_pde4_targets.json")])

    con.executemany(f"INSERT INTO activities VALUES ({','.join('?' * 17)})", [
        (a["activity_id"], a["molecule_chembl_id"], a["target_chembl_id"], a["assay_chembl_id"],
         a.get("assay_type"), a.get("document_chembl_id"), a.get("document_year"),
         a.get("standard_type"), a.get("standard_relation"), a.get("standard_value"),
         a.get("standard_units"), a.get("pchembl_value"), a.get("type"), a.get("relation"),
         a.get("value"), a.get("units"), a.get("data_validity_comment"))
        for a in load("chembl_pde4_activities.json")])

    # FDA products and ingredients.
    with zipfile.ZipFile(RAW / "openfda_drugsfda.json.zip") as archive:
        fda_json = json.loads(archive.read(archive.namelist()[0]))
    fda, fda_meta = fda_json["results"], fda_json.get("meta", {})
    products, ingredients = [], []
    for app in fda:
        number = app["application_number"]
        con.execute("INSERT OR IGNORE INTO fda_applications VALUES (?,?)", (number, app.get("sponsor_name")))
        for p in app.get("products") or []:
            products.append((number, p.get("product_number"), p.get("brand_name"), p.get("dosage_form"),
                             p.get("route"), p.get("marketing_status"), p.get("reference_drug")))
            ingredients += [(number, p.get("product_number"), normalise(i["name"]), i.get("strength"))
                            for i in p.get("active_ingredients") or [] if i.get("name")]
    con.executemany("INSERT OR IGNORE INTO fda_products VALUES (?,?,?,?,?,?,?)", products)
    con.executemany("INSERT INTO fda_product_ingredients VALUES (?,?,?,?)", ingredients)

    # Name matching: exact name/synonym first, then without salt words.
    parent_of = {r[0]: r[1] for r in con.execute("SELECT chembl_id, parent_chembl_id FROM molecules")}
    index: dict[str, str] = {}
    for cid, name in con.execute("SELECT chembl_id, pref_name FROM molecules WHERE max_phase = 4 AND pref_name IS NOT NULL"):
        index.setdefault(normalise(name), cid)
    for cid, name in con.execute("SELECT chembl_id, synonym FROM molecule_synonyms"):
        index.setdefault(normalise(name), cid)
    mapping = []
    for (name,) in con.execute("SELECT DISTINCT ingredient_name FROM fda_product_ingredients"):
        if name in index:
            cid, method = index[name], "exact"
        elif strip_salts(name) in index:
            cid, method = index[strip_salts(name)], "salt_stripped"
        else:
            cid, method = None, "unmatched"
        mapping.append((name, cid, parent_of.get(cid, cid), method))
    con.executemany("INSERT INTO fda_ingredient_map VALUES (?,?,?,?)", mapping)
    con.execute("INSERT INTO molecule_routes SELECT * FROM v_molecule_routes")

    manifest = load("manifest.json")
    info = {
        "chembl_release": manifest["chembl_release"],
        "chembl_release_date": manifest["chembl_release_date"],
        "openfda_last_updated": fda_meta.get("last_updated"),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    info.update({f"sha256:{name}": f["sha256"] for name, f in manifest["files"].items()})
    info.update({f"retrieved_at:{name}": f["retrieved_at"] for name, f in manifest["files"].items()})
    con.executemany("INSERT INTO dataset_info VALUES (?,?)", info.items())

    con.commit()
    for table in ("molecules", "molecule_synonyms", "atc_classes", "molecule_atc", "fda_products",
                  "fda_product_ingredients", "fda_ingredient_map", "molecule_routes", "targets", "activities"):
        print(f"{table:24s} {con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]:>8,}")
    print(con.execute("SELECT match_method, COUNT(*) FROM fda_ingredient_map GROUP BY 1").fetchall())
    con.execute("VACUUM")
    con.close()
    print(f"Database: {DB} ({DB.stat().st_size / 1e6:.1f} MB)")

    # Gzipped, base64 copy served next to the web app (app/index.html fetches it).
    # Base64 text because the artifact host serves .txt but not arbitrary binaries.
    APP_DB.write_bytes(base64.b64encode(gzip.compress(DB.read_bytes(), compresslevel=9)))
    print(f"App copy: {APP_DB} ({APP_DB.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
