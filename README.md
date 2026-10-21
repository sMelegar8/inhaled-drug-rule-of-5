# Inhalation SQL Workbench

Portfolio project (pharma data analyst): **is there a "Rule of 5" for inhaled drugs?**
Compares the computed properties of FDA-approved inhaled and oral small molecules,
then applies the findings to PDE4 inhibitors (respiratory target).

Exploratory analysis on public data, not a discovery: 65 inhaled vs 1,249 oral
molecules are linked in the current build.

## Data
- ChEMBL 37 (EMBL-EBI, 2026-05-01), web services API, licence CC BY-SA 3.0.
- openFDA Drugs@FDA bulk export (2026-10-20): route of administration.

The full ChEMBL dump (5.4 GB compressed, ~25 GB extracted) did not fit on disk,
so only the needed slices are downloaded. Provenance and SHA-256 hashes are in
`data/raw/manifest.json` and in the `dataset_info` table.

## Run
```
py -3.13 -m pip install -r requirements.txt
py -3.13 src/01_download_data.py     # raw files -> data/raw/
py -3.13 src/02_build_database.py    # SQLite -> data/processed/ + app copy
py -3.13 src/cleaning.py             # cleaned tables + exclusion ledger (same as notebook 02)
py -3.13 -m http.server 8000 -d app  # open http://localhost:8000
```

## Cleaning (notebooks/02_pulizia_dati.ipynb)
Ordered rules, one reason per excluded record, raw tables untouched.
The ledger is in `cleaning_log` / `cleaning_exclusions` and `reports/cleaning_log.csv`.
- PDE4 measurements: 11,954 -> 8,793 clean (censored values, curator flags,
  non-nM units removed) -> 6,887 molecule-target medians.
- Analysis set: 1,305 molecules -> 46 lung-targeted inhaled + 1,108 oral
  (gases, diagnostics, CNS-acting inhaled and gut-local oral drugs excluded).

## Statistics in the app
The app's statistics panel works on the current query result (filter with
`WHERE`): descriptive statistics, Pearson/Spearman correlations, PCA (scree,
biplot, loadings, scores saved to `pca_scores`), Mann-Whitney / Kruskal-Wallis
with Holm correction and effect sizes. `app/stats.js` is checked against
numpy/pandas/scipy with `py -3.13 tests/validate_stats.py` (needs Node.js).

## Exploration
`py -3.13 src/explore_trends.py`: among inhaled drugs, year of first approval
correlates with lipophilicity, size and rotatable bonds, within each
respiratory class (class-adjusted +0.43 logP per decade, permutation p = 0.002),
and logP rises faster than for oral drugs. Output in `reports/`.
`py -3.13 src/explore_chemistry.py` (needs RDKit): five structural trends -
fluorination gap, quaternary ammonium enrichment, PDE4B over PDE4D
selectivity over time, decline of the catechol diether in PDE4 papers, and
much larger compounds in inhaled PDE4 programmes. Output `reports/chemistry_trends.csv`.

## Layout
- `src/` Python pipeline (download, build, cleaning)
- `sql/schema.sql`, `sql/clean_schema.sql` tables with column comments
- `notebooks/` numbered analysis notebooks; `reports/` ledger CSV and figures
- `app/index.html` SQL editor, auto-updating charts, searchable schema
- `app/come-funziona.html` step-by-step explanation; `app/backlog.html` next work (pages of the same site)
