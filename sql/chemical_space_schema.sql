-- Tables written by src/pde4_chemical_space.py (step 4): the PDE4 chemical space.
-- Built from activity_potency and molecules; raw and cleaned tables are not modified.

DROP TABLE IF EXISTS pde4_chemical_space;
DROP TABLE IF EXISTS pde4_clusters;
DROP TABLE IF EXISTS pde4_k_selection;

-- One row per PDE4 compound with a structure and a clean potency.
CREATE TABLE pde4_chemical_space (
    molecule_chembl_id  TEXT PRIMARY KEY,
    pref_name           TEXT,      -- name, empty for most research compounds
    cluster             INTEGER,   -- k-means cluster, 1..k
    cluster_label       TEXT,      -- short chemical description of the cluster
    map_x               REAL,      -- t-SNE map coordinate (no units)
    map_y               REAL,      -- t-SNE map coordinate (no units)
    best_p              REAL,      -- highest median pIC50/pKi over the PDE4 targets
    p_pde4b             REAL,      -- median potency on PDE4B, if measured
    p_pde4d             REAL,      -- median potency on PDE4D, if measured
    selectivity_bd      REAL,      -- p_pde4b - p_pde4d (log units); > 0 favours PDE4B
    mw_freebase         REAL,      -- molecular weight (Da)
    alogp               REAL,      -- Crippen logP
    heavy_atoms         INTEGER,
    ligand_efficiency   REAL,      -- 1.37 * best_p / heavy_atoms (kcal/mol per heavy atom)
    lipophilic_efficiency REAL,    -- best_p - alogp (LLE)
    first_year          INTEGER,   -- year of the first paper or patent reporting it
    n_documents         INTEGER,   -- papers or patents reporting clean measurements
    inhaled_paper       INTEGER,   -- 1 if reported in a paper about inhaled PDE4 inhibitors
    is_medoid           INTEGER    -- 1 for the most central molecule of its cluster
);

-- One row per cluster: size, potency, properties and a representative structure.
CREATE TABLE pde4_clusters (
    cluster             INTEGER PRIMARY KEY,
    cluster_label       TEXT,
    top_motif           TEXT,      -- most enriched substructure in the cluster
    motif_share         REAL,      -- share of the cluster carrying that motif (0-1)
    n_molecules         INTEGER,
    median_p            REAL,      -- median best_p
    share_potent        REAL,      -- share with best_p >= 8 (IC50 <= 10 nM)
    median_mw           REAL,
    median_alogp        REAL,
    median_le           REAL,      -- median ligand efficiency
    median_first_year   REAL,
    n_selectivity       INTEGER,   -- molecules measured on both PDE4B and PDE4D
    median_selectivity_bd REAL,    -- median p_pde4b - p_pde4d
    n_inhaled_paper     INTEGER,   -- molecules from inhaled PDE4 papers
    medoid_chembl_id    TEXT,      -- most central molecule
    medoid_smiles       TEXT,
    named_compounds     TEXT       -- named compounds in the cluster, most potent first
);

-- Model selection: how k-means behaves for each number of clusters k.
CREATE TABLE pde4_k_selection (
    k                   INTEGER PRIMARY KEY,
    inertia             REAL,      -- within-cluster sum of squares (lower = tighter)
    silhouette          REAL,      -- -1..1, higher = better separated clusters
    stability_ari       REAL       -- mean adjusted Rand index over 5 random seeds (1 = identical)
);
