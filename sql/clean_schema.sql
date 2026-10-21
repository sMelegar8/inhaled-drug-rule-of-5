-- Tables written by src/cleaning.py (step 3). Raw tables are never modified.

DROP TABLE IF EXISTS cleaning_log;
DROP TABLE IF EXISTS cleaning_exclusions;
DROP TABLE IF EXISTS activities_clean;
DROP TABLE IF EXISTS activity_potency;
DROP TABLE IF EXISTS molecule_routes_clean;
DROP TABLE IF EXISTS inhalation_dataset;

-- Exclusion ledger: one row per cleaning rule, in the order applied.
CREATE TABLE cleaning_log (
    dataset       TEXT,     -- activities or molecules
    step          INTEGER,  -- order of application
    rule_id       TEXT,     -- A1, A2... M1, M2...
    kind          TEXT,     -- input, exclude, transform, check, aggregate
    rule          TEXT,     -- what the rule does
    reason        TEXT,     -- why it is needed
    unit          TEXT,     -- what one row counts (measurements, molecules...)
    rows_before   INTEGER,
    rows_removed  INTEGER,  -- rows excluded by this rule only
    rows_after    INTEGER,
    rows_flagged  INTEGER   -- rows marked but kept (checks)
);

-- Every excluded record and the first rule that excluded it.
CREATE TABLE cleaning_exclusions (
    dataset    TEXT,
    record_id  TEXT,        -- activity_id or chembl_id
    rule_id    TEXT         -- joins cleaning_log.rule_id
);

-- PDE4 measurements that passed every rule, one row per measurement.
CREATE TABLE activities_clean (
    activity_id         INTEGER PRIMARY KEY,
    molecule_chembl_id  TEXT,
    target_chembl_id    TEXT,
    target_name         TEXT,     -- short isoform name: PDE4A..PDE4D or PDE4 family
    target_type         TEXT,     -- SINGLE PROTEIN or PROTEIN FAMILY
    assay_chembl_id     TEXT,
    assay_type          TEXT,     -- B = binding, F = functional
    document_chembl_id  TEXT,
    document_year       INTEGER,
    standard_type       TEXT,     -- IC50 or Ki
    standard_value      REAL,     -- nM
    p_activity          REAL,     -- 9 - log10(nM): pIC50 or pKi
    pchembl_value       REAL,     -- ChEMBL value, for cross-checking
    pchembl_mismatch    INTEGER   -- 1 if p_activity differs from pChEMBL by > 0.05
);

-- Repeated measurements aggregated: one row per molecule, target and type.
CREATE TABLE activity_potency (
    molecule_chembl_id  TEXT,
    target_chembl_id    TEXT,
    target_name         TEXT,
    standard_type       TEXT,     -- IC50 or Ki
    p_median            REAL,     -- median pIC50 / pKi
    p_min               REAL,
    p_max               REAL,
    p_range             REAL,     -- max - min, in log units
    n_measurements      INTEGER,
    n_documents         INTEGER,  -- independent publications
    inconsistent        INTEGER,  -- 1 if p_range > 1 (more than 10-fold apart)
    PRIMARY KEY (molecule_chembl_id, target_chembl_id, standard_type)
);

-- FDA routes split into single tokens ("INHALATION, ORAL" -> two rows).
CREATE TABLE molecule_routes_clean (
    chembl_id   TEXT,         -- parent molecule
    route       TEXT,         -- one route, upper case
    n_products  INTEGER,
    PRIMARY KEY (chembl_id, route)
);

-- Analysis set: lung-targeted inhaled drugs vs oral drugs, one row per molecule.
CREATE TABLE inhalation_dataset (
    chembl_id            TEXT PRIMARY KEY,
    pref_name            TEXT,
    route_group          TEXT,     -- inhaled or oral (mutually exclusive)
    also_oral            INTEGER,  -- inhaled molecule that also has oral products
    n_inhaled_products   INTEGER,
    n_oral_products      INTEGER,
    atc_codes            TEXT,     -- ATC codes of the molecule and its salts
    first_approval       INTEGER,
    mw_freebase          REAL,     -- molecular weight (Da)
    alogp                REAL,     -- Crippen logP
    psa                  REAL,     -- polar surface area (A^2)
    hbd                  INTEGER,  -- H-bond donors
    hba                  INTEGER,  -- H-bond acceptors
    rtb                  INTEGER,  -- rotatable bonds
    aromatic_rings       INTEGER,
    heavy_atoms          INTEGER,
    num_ro5_violations   INTEGER,  -- Lipinski violations
    qed_weighted         REAL      -- drug-likeness 0..1
);
