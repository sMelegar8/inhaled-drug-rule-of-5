# Data licence

The source code in this repository is released under the MIT licence (`LICENSE`).
The data files follow the licences of their sources.

## ChEMBL

`app/pharma_inhalation.sqlite.gz.b64.txt` (a gzip-compressed, base64-encoded SQLite
database) and the tables in `reports/` contain data derived from ChEMBL release 37
(EMBL-EBI, 2026-05-01).

ChEMBL is licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported
licence (CC BY-SA 3.0): https://creativecommons.org/licenses/by-sa/3.0/.
These derived files are shared under the same licence.

Reference: Zdrazil B. et al. The ChEMBL Database in 2023. Nucleic Acids Research,
52(D1), D1180-D1192 (2024). https://doi.org/10.1093/nar/gkad1004

## openFDA

Routes of administration come from the openFDA Drugs@FDA bulk export
(https://open.fda.gov/), retrieved on 2026-10-20. openFDA data are provided under the
openFDA terms of service (https://open.fda.gov/terms/). The FDA does not endorse this
project.

## Raw downloads

The raw downloads (`data/raw/`) and the full local database (`data/processed/`) are not
stored in this repository. Run `src/01_download_data.py` and `src/02_build_database.py`
to recreate them. `data/raw/manifest.json` records the source URL, record count,
retrieval time and SHA-256 hash of each file used for the published results.
