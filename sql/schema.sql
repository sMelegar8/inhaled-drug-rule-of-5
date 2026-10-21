-- Schema of data/processed/pharma_inhalation.sqlite
-- Tables keep the source values unchanged; cleaning happens in later steps.

-- Provenance: source releases, download dates and file hashes.
CREATE TABLE dataset_info (
    key    TEXT PRIMARY KEY,
    value  TEXT
);

-- One row per ChEMBL molecule (approved drugs + compounds tested on PDE4).
CREATE TABLE molecules (
    chembl_id           TEXT PRIMARY KEY,
    pref_name           TEXT,
    molecule_type       TEXT,      -- Small molecule, Protein, Antibody, ...
    max_phase           REAL,      -- 4 = approved; NULL = preclinical
    first_approval      INTEGER,   -- year
    oral                INTEGER,   -- ChEMBL route flags (0/1)
    parenteral          INTEGER,
    topical             INTEGER,
    prodrug             INTEGER,
    natural_product     INTEGER,
    parent_chembl_id    TEXT,      -- salt forms point to the parent molecule
    canonical_smiles    TEXT,
    standard_inchi_key  TEXT,
    full_molformula     TEXT,
    mw_freebase         REAL,      -- molecular weight of the parent (Da)
    full_mwt            REAL,      -- molecular weight incl. salt
    alogp               REAL,      -- Crippen logP
    cx_logp             REAL,   -- always NULL: not returned by the ChEMBL 37 API
    cx_logd             REAL,   -- always NULL: not returned by the ChEMBL 37 API
    psa                 REAL,      -- polar surface area (A^2)
    hba                 INTEGER,   -- H-bond acceptors
    hbd                 INTEGER,   -- H-bond donors
    rtb                 INTEGER,   -- rotatable bonds
    aromatic_rings      INTEGER,
    heavy_atoms         INTEGER,
    num_ro5_violations  INTEGER,   -- Lipinski rule-of-five violations
    qed_weighted        REAL,      -- drug-likeness 0..1
    cx_most_apka        REAL,   -- always NULL: not returned by the ChEMBL 37 API
    cx_most_bpka        REAL,   -- always NULL: not returned by the ChEMBL 37 API
    molecular_species   TEXT,   -- always NULL: not returned by the ChEMBL 37 API
    np_likeness_score   REAL,
    source_set          TEXT       -- 'approved', 'pde4_tested' or 'approved+pde4_tested'
);

-- Names used to match FDA ingredient names (INN, USAN, BAN, trade names...).
CREATE TABLE molecule_synonyms (
    chembl_id  TEXT REFERENCES molecules(chembl_id),
    synonym    TEXT,
    syn_type   TEXT
);

-- WHO ATC hierarchy and the ATC codes of each molecule.
CREATE TABLE atc_classes (
    level5              TEXT PRIMARY KEY,
    who_name            TEXT,
    level1              TEXT, level1_description TEXT,
    level2              TEXT, level2_description TEXT,
    level3              TEXT, level3_description TEXT,
    level4              TEXT, level4_description TEXT
);
CREATE TABLE molecule_atc (
    chembl_id  TEXT REFERENCES molecules(chembl_id),
    level5     TEXT REFERENCES atc_classes(level5),
    PRIMARY KEY (chembl_id, level5)
);

-- FDA products (openFDA Drugs@FDA): route of administration lives here.
CREATE TABLE fda_applications (
    application_number  TEXT PRIMARY KEY,  -- NDA/ANDA/BLA number
    sponsor_name        TEXT
);
CREATE TABLE fda_products (
    application_number  TEXT REFERENCES fda_applications(application_number),
    product_number      TEXT,
    brand_name          TEXT,
    dosage_form         TEXT,      -- e.g. AEROSOL, METERED; POWDER; TABLET
    route               TEXT,      -- e.g. INHALATION; ORAL; INTRAVENOUS
    marketing_status    TEXT,      -- Prescription, Discontinued, ...
    reference_drug      TEXT,
    PRIMARY KEY (application_number, product_number)
);
CREATE TABLE fda_product_ingredients (
    application_number  TEXT,
    product_number      TEXT,
    ingredient_name     TEXT,      -- as written by FDA, often a salt
    strength            TEXT
);

-- Bridge between FDA ingredient names and ChEMBL molecules.
CREATE TABLE fda_ingredient_map (
    ingredient_name   TEXT PRIMARY KEY,
    chembl_id         TEXT,        -- molecule whose name matched
    parent_chembl_id  TEXT,        -- parent used for the properties
    match_method      TEXT         -- exact, salt_stripped, unmatched
);

-- PDE4 targets and their bioactivity measurements (respiratory use case).
CREATE TABLE targets (
    target_chembl_id  TEXT PRIMARY KEY,
    pref_name         TEXT,
    target_type       TEXT,
    organism          TEXT
);
CREATE TABLE activities (
    activity_id           INTEGER PRIMARY KEY,
    molecule_chembl_id    TEXT REFERENCES molecules(chembl_id),
    target_chembl_id      TEXT REFERENCES targets(target_chembl_id),
    assay_chembl_id       TEXT,
    assay_type            TEXT,    -- B = binding, F = functional
    document_chembl_id    TEXT,
    document_year         INTEGER,
    standard_type         TEXT,    -- IC50, Ki
    standard_relation     TEXT,    -- '=', '>', '<', ...
    standard_value        REAL,
    standard_units        TEXT,    -- usually nM after ChEMBL standardisation
    pchembl_value         REAL,    -- -log10(molar); only for clean '=' values
    published_type        TEXT,    -- values as written in the paper
    published_relation    TEXT,
    published_value       REAL,
    published_units       TEXT,
    data_validity_comment TEXT     -- ChEMBL curator warnings
);

-- Materialised copy of v_molecule_routes (filled by the build script):
-- the browser SQL engine is slower, so the join is computed once.
CREATE TABLE molecule_routes (
    chembl_id   TEXT,          -- parent molecule
    route       TEXT,          -- FDA route of administration
    n_products  INTEGER,       -- FDA products with this molecule and route
    PRIMARY KEY (chembl_id, route)
);

CREATE INDEX idx_syn_name ON molecule_synonyms(synonym);
CREATE INDEX idx_ing_name ON fda_product_ingredients(ingredient_name);
CREATE INDEX idx_ing_product ON fda_product_ingredients(application_number, product_number);
CREATE INDEX idx_routes_route ON molecule_routes(route);
CREATE INDEX idx_act_mol ON activities(molecule_chembl_id);
CREATE INDEX idx_act_target ON activities(target_chembl_id);

-- One row per parent molecule and FDA route, with product counts.
CREATE VIEW v_molecule_routes AS
SELECT m.parent_chembl_id                       AS chembl_id,
       p.route,
       COUNT(DISTINCT p.application_number || '-' || p.product_number) AS n_products
FROM fda_products p
JOIN fda_product_ingredients i
  ON i.application_number = p.application_number AND i.product_number = p.product_number
JOIN fda_ingredient_map m ON m.ingredient_name = i.ingredient_name
WHERE m.parent_chembl_id IS NOT NULL
GROUP BY m.parent_chembl_id, p.route;
