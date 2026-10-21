"""Step 1 - Download the raw data.

Sources
- ChEMBL web services (EMBL-EBI): approved drugs (max_phase = 4) with
  computed properties, synonyms and ATC codes; PDE4 bioactivities (IC50, Ki)
  and the properties of the tested compounds.
- openFDA Drugs@FDA bulk export: FDA products with dosage form and route
  of administration (used to label inhaled vs oral drugs).

The full ChEMBL SQLite dump (5.4 GB compressed, about 25 GB extracted) does
not fit on this PC, so we download only the slices the project needs.
Every file is written unchanged to data/raw/ and recorded in
data/raw/manifest.json with URL, timestamp, record count and SHA-256.

Usage:  py -3.13 src/01_download_data.py
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
OPENFDA_ZIP = "https://download.open.fda.gov/drug/drugsfda/drug-drugsfda-0001-of-0001.json.zip"

# Human PDE4 isoforms plus the PDE4 protein family record.
PDE4_TARGETS = ["CHEMBL254", "CHEMBL275", "CHEMBL291", "CHEMBL288", "CHEMBL2093863"]

MOLECULE_FIELDS = ",".join([
    "molecule_chembl_id", "pref_name", "molecule_type", "max_phase", "first_approval",
    "oral", "parenteral", "topical", "prodrug", "natural_product",
    "molecule_hierarchy", "molecule_properties", "molecule_structures",
    "molecule_synonyms", "atc_classifications",
])
ACTIVITY_FIELDS = ",".join([
    "activity_id", "molecule_chembl_id", "target_chembl_id", "assay_chembl_id",
    "assay_type", "document_chembl_id", "document_year",
    "standard_type", "standard_relation", "standard_value", "standard_units",
    "pchembl_value", "type", "relation", "value", "units", "data_validity_comment",
])

session = requests.Session()
session.headers["Accept"] = "application/json"


def get_json(url: str, params: dict | None = None, retries: int = 4) -> dict:
    """GET with simple exponential backoff for transient API errors."""
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=180)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as error:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"  retry in {wait}s after: {error}")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def paginate(resource: str, key: str, params: dict) -> list[dict]:
    """Collect every page of a ChEMBL list endpoint (max 1000 rows per page)."""
    url: str | None = f"{CHEMBL}/{resource}.json"
    query: dict | None = {**params, "limit": 1000}
    rows: list[dict] = []
    while url:
        page = get_json(url, query)
        rows.extend(page[key])
        total = page["page_meta"]["total_count"]
        print(f"  {resource}: {len(rows)}/{total}")
        next_page = page["page_meta"]["next"]
        url = f"https://www.ebi.ac.uk{next_page}" if next_page else None
        query = None  # the "next" link already carries the filters
    return rows


def save_json(name: str, payload, source: str, manifest: dict) -> None:
    path = RAW / name
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    path.write_bytes(data)
    manifest["files"][name] = {
        "source": source,
        "records": len(payload) if isinstance(payload, list) else None,
        "sha256": hashlib.sha256(data).hexdigest(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    status = get_json(f"{CHEMBL}/status.json")
    manifest = {"chembl_release": status["chembl_db_version"],
                "chembl_release_date": status["chembl_release_date"], "files": {}}
    print(f"ChEMBL release: {status['chembl_db_version']} ({status['chembl_release_date']})")

    print("1/5 Approved drugs")
    approved = paginate("molecule", "molecules", {"max_phase": 4, "only": MOLECULE_FIELDS})
    save_json("chembl_approved_molecules.json", approved, f"{CHEMBL}/molecule?max_phase=4", manifest)

    print("2/5 ATC classification")
    atc = paginate("atc_class", "atc", {})
    save_json("chembl_atc_classes.json", atc, f"{CHEMBL}/atc_class", manifest)

    print("3/5 PDE4 targets and activities")
    targets = get_json(f"{CHEMBL}/target.json", {"target_chembl_id__in": ",".join(PDE4_TARGETS),
                                                "only": "target_chembl_id,pref_name,target_type,organism"})["targets"]
    save_json("chembl_pde4_targets.json", targets, f"{CHEMBL}/target", manifest)
    activities = paginate("activity", "activities", {
        "target_chembl_id__in": ",".join(PDE4_TARGETS),
        "standard_type__in": "IC50,Ki",
        "only": ACTIVITY_FIELDS,
    })
    save_json("chembl_pde4_activities.json", activities, f"{CHEMBL}/activity", manifest)

    print("4/5 Properties of the tested PDE4 compounds")
    known = {m["molecule_chembl_id"] for m in approved}
    missing = sorted({a["molecule_chembl_id"] for a in activities} - known)
    tested: list[dict] = []
    for start in range(0, len(missing), 100):
        batch = missing[start:start + 100]
        tested.extend(get_json(f"{CHEMBL}/molecule.json", {
            "molecule_chembl_id__in": ",".join(batch), "limit": 1000, "only": MOLECULE_FIELDS,
        })["molecules"])
        print(f"  molecule: {len(tested)}/{len(missing)}")
    save_json("chembl_pde4_compounds.json", tested, f"{CHEMBL}/molecule?molecule_chembl_id__in=...", manifest)

    print("5/5 openFDA Drugs@FDA bulk export")
    response = session.get(OPENFDA_ZIP, timeout=600)
    response.raise_for_status()
    (RAW / "openfda_drugsfda.json.zip").write_bytes(response.content)
    manifest["files"]["openfda_drugsfda.json.zip"] = {
        "source": OPENFDA_ZIP, "records": None,
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Done. Manifest: data/raw/manifest.json")


if __name__ == "__main__":
    main()
