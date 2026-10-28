"""Build the data dictionary shown on app/data-dictionary.html.

Combines hand-written explanations (below) with facts read from the database:
column types, how many values are filled, distinct counts, examples, numeric
ranges, row counts and how well each join key matches its target table.

The script refuses to finish if a column in the database has no explanation,
or if an explanation refers to a column that no longer exists, so the
dictionary cannot silently drift away from the schema.

Output: app/data-dictionary.json
Usage:  py -3.13 src/build_data_dictionary.py      (after cleaning.py)
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "processed" / "pharma_inhalation.sqlite"
OUT = ROOT / "app" / "data-dictionary.json"

LAYERS = [
    ("provenance", "Provenance", "Where the data came from and when it was downloaded."),
    ("raw_chembl", "Raw · ChEMBL", "Loaded unchanged from the ChEMBL 37 web services (step 01-02)."),
    ("raw_fda", "Raw · FDA", "Loaded unchanged from the openFDA Drugs@FDA export (step 01-02)."),
    ("integration", "Integration", "Links FDA products to ChEMBL molecules (step 02)."),
    ("clean", "Clean analysis tables", "Produced by the cleaning rules; use these for analysis (step 03)."),
    ("ledger", "Cleaning ledger", "Records every cleaning rule and every excluded record (step 03)."),
    ("analysis", "Analysis outputs", "Results of the PDE4 chemical space model: fingerprints, k-means and t-SNE (step 04)."),
]

# ------------------------------------------------------------------ tables
TABLES = {
    "dataset_info": dict(
        layer="provenance", built_by="src/02_build_database.py",
        purpose="Key-value list that records the source releases, the download time of every raw file and its SHA-256 fingerprint.",
        grain="one fact about the build", key="key",
        use_for="Citing the data version in a report, or checking that two people analysed the same input files.",
        watch_out="Values are stored as text, including dates.",
        example="SELECT key, value\nFROM dataset_info\nWHERE key NOT LIKE 'sha256:%';"),
    "molecules": dict(
        layer="raw_chembl", built_by="src/02_build_database.py",
        purpose="Every ChEMBL molecule in the project: the 4,225 approved drugs plus the 7,326 compounds tested on PDE4, with structure and computed physicochemical properties.",
        grain="one ChEMBL molecule (a salt and its parent are separate rows)", key="chembl_id",
        use_for="Looking up names, structures and properties; most other tables join here through a ChEMBL ID.",
        watch_out="Salt forms (e.g. albuterol sulfate) are separate rows that point to their parent through parent_chembl_id: compare properties on the parent. 95 research compounds point to a parent that was not downloaded. The ChemAxon columns (cx_*, molecular_species) are empty because the ChEMBL 37 API does not return them.",
        example="SELECT chembl_id, pref_name, mw_freebase, alogp, psa, hbd, hba\nFROM molecules\nWHERE max_phase = 4 AND molecule_type = 'Small molecule'\nORDER BY mw_freebase DESC\nLIMIT 20;"),
    "molecule_synonyms": dict(
        layer="raw_chembl", built_by="src/02_build_database.py",
        purpose="Alternative names of the approved drugs: international (INN), US (USAN) and British (BAN) names, trade names and research codes.",
        grain="one name of one molecule", key="none (a molecule has many names)",
        use_for="Finding a drug by any of its names, and matching FDA ingredient names to ChEMBL.",
        watch_out="Only approved drugs have synonyms here. The same name can appear with several syn_type values.",
        example="SELECT m.pref_name, s.synonym, s.syn_type\nFROM molecule_synonyms AS s\nJOIN molecules AS m ON m.chembl_id = s.chembl_id\nWHERE s.synonym LIKE 'ALBUTEROL%';"),
    "atc_classes": dict(
        layer="raw_chembl", built_by="src/02_build_database.py",
        purpose="The WHO Anatomical Therapeutic Chemical (ATC) classification: a five-level tree from the organ system down to the single substance.",
        grain="one ATC level-5 code (one substance in one therapeutic use)", key="level5",
        use_for="Grouping drugs by therapeutic area, e.g. R03 = drugs for obstructive airway diseases.",
        watch_out="A molecule can have several ATC codes (one per use), so joining through molecule_atc can multiply rows.",
        example="SELECT level2, level2_description, COUNT(*) AS substances\nFROM atc_classes\nWHERE level1 = 'R'\nGROUP BY level2, level2_description;"),
    "molecule_atc": dict(
        layer="raw_chembl", built_by="src/02_build_database.py",
        purpose="Bridge table that assigns ATC codes to approved molecules.",
        grain="one ATC code of one molecule", key="chembl_id + level5",
        use_for="Answering 'which drugs belong to this class?' or 'which classes does this drug belong to?'.",
        watch_out="Codes are often stored on the salt form, not on the parent. The cleaned inhalation_dataset.atc_codes already combines both.",
        example="SELECT m.pref_name, a.level5, a.who_name, a.level4_description\nFROM molecule_atc AS ma\nJOIN atc_classes AS a ON a.level5 = ma.level5\nJOIN molecules AS m ON m.chembl_id = ma.chembl_id\nWHERE a.level3 = 'R03A';"),
    "targets": dict(
        layer="raw_chembl", built_by="src/02_build_database.py",
        purpose="The biological targets of the PDE4 activity data: the four human isoforms PDE4A-D and the PDE4 family record.",
        grain="one ChEMBL target", key="target_chembl_id",
        use_for="Giving readable names to the target IDs in activities.",
        watch_out="The PROTEIN FAMILY record collects assays that did not specify the isoform.",
        example="SELECT * FROM targets;"),
    "activities": dict(
        layer="raw_chembl", built_by="src/02_build_database.py",
        purpose="Raw bioactivity measurements (IC50 and Ki) of compounds on PDE4, both as written in the paper and as standardised by ChEMBL.",
        grain="one measurement reported in one paper", key="activity_id",
        use_for="Auditing the raw data and understanding what the cleaning removed. For analysis, use activities_clean.",
        watch_out="Contains censored values ('>' or '<'), mixed published units and measurements flagged by ChEMBL curators. A lower IC50 means a more potent compound.",
        example="SELECT standard_relation, standard_units, COUNT(*) AS n\nFROM activities\nGROUP BY standard_relation, standard_units\nORDER BY n DESC;"),
    "fda_applications": dict(
        layer="raw_fda", built_by="src/02_build_database.py",
        purpose="FDA marketing applications: new drugs (NDA), generics (ANDA) and biologics (BLA), with the company that filed them.",
        grain="one FDA application", key="application_number",
        use_for="Counting products by sponsor or by application type (the prefix of the number).",
        watch_out="One application can contain many products (strengths, forms).",
        example="SELECT CASE WHEN application_number LIKE 'ANDA%' THEN 'ANDA (generic)'\n            WHEN application_number LIKE 'NDA%' THEN 'NDA (new drug)'\n            ELSE 'BLA (biologic)' END AS type,\n       COUNT(*) AS applications\nFROM fda_applications\nGROUP BY type;"),
    "fda_products": dict(
        layer="raw_fda", built_by="src/02_build_database.py",
        purpose="Individual FDA products with brand name, dosage form, route of administration and marketing status. This is where 'inhaled' comes from.",
        grain="one product inside one application", key="application_number + product_number",
        use_for="Finding inhaled, oral or injectable products.",
        watch_out="The route field is free text: 'INHALATION, ORAL' and 'Inhalation, Oral' mean two routes. Use molecule_routes_clean for a normalised version.",
        example="SELECT route, COUNT(*) AS products\nFROM fda_products\nWHERE UPPER(route) LIKE '%INHAL%'\nGROUP BY route;"),
    "fda_product_ingredients": dict(
        layer="raw_fda", built_by="src/02_build_database.py",
        purpose="Active ingredients of each FDA product, with the name as written by FDA and the strength.",
        grain="one active ingredient of one product", key="none (join on application_number + product_number)",
        use_for="Linking products to molecules through fda_ingredient_map.",
        watch_out="Ingredient names are often salts (ALBUTEROL SULFATE) and some combinations are written in a single field.",
        example="SELECT p.brand_name, i.ingredient_name, i.strength\nFROM fda_product_ingredients AS i\nJOIN fda_products AS p\n  ON p.application_number = i.application_number AND p.product_number = i.product_number\nWHERE p.route = 'INHALATION'\nLIMIT 20;"),
    "fda_ingredient_map": dict(
        layer="integration", built_by="src/02_build_database.py",
        purpose="The bridge between FDA ingredient names and ChEMBL molecules. FDA and ChEMBL share no identifier, so the link is made by name.",
        grain="one distinct FDA ingredient name", key="ingredient_name",
        use_for="Checking how an FDA name was matched, and how reliable the match is.",
        watch_out="7.4% of names stay unmatched (mostly combinations written in one field and recent biologics). match_method tells you how each match was made.",
        example="SELECT match_method, COUNT(*) AS names\nFROM fda_ingredient_map\nGROUP BY match_method;"),
    "v_molecule_routes": dict(
        layer="integration", built_by="sql/schema.sql (view)",
        purpose="A view: a saved query that joins products, ingredients and the name map to count products per molecule and route. It is recalculated every time you query it.",
        grain="one parent molecule and one raw route text", key="chembl_id + route",
        use_for="Reading the join logic. For speed, query molecule_routes instead, which stores the same result.",
        watch_out="Slow in the browser workbench, because it recomputes a join over 60,000 ingredient rows.",
        example="SELECT * FROM v_molecule_routes WHERE route = 'INHALATION' LIMIT 20;"),
    "molecule_routes": dict(
        layer="integration", built_by="src/02_build_database.py",
        purpose="Stored (materialised) copy of v_molecule_routes, with an index, so the browser can query it fast.",
        grain="one parent molecule and one raw route text", key="chembl_id + route",
        use_for="Quick route lookups on the raw route text.",
        watch_out="Uses the raw route text, so 'INHALATION, ORAL' is its own route here. Prefer molecule_routes_clean.",
        example="SELECT m.pref_name, r.route, r.n_products\nFROM molecule_routes AS r\nJOIN molecules AS m ON m.chembl_id = r.chembl_id\nWHERE r.route = 'INHALATION'\nORDER BY r.n_products DESC;"),
    "molecule_routes_clean": dict(
        layer="clean", built_by="src/cleaning.py",
        purpose="Routes of administration after cleaning: combined values are split ('INHALATION, ORAL' becomes two rows) and written in upper case.",
        grain="one parent molecule and one single route", key="chembl_id + route",
        use_for="Any question about which molecules are given by which route.",
        watch_out="A molecule with both inhaled and oral products has two rows here.",
        example="SELECT route, COUNT(*) AS molecules\nFROM molecule_routes_clean\nGROUP BY route\nORDER BY molecules DESC\nLIMIT 10;"),
    "inhalation_dataset": dict(
        layer="clean", built_by="src/cleaning.py",
        purpose="The analysis set of the project: approved small molecules split into lung-targeted inhaled drugs and oral drugs, with their properties.",
        grain="one parent molecule", key="chembl_id",
        use_for="Comparing inhaled and oral drugs: statistics, PCA, trends over time.",
        watch_out="The groups are exclusive: a molecule with inhaled products is 'inhaled' even if it also exists orally (also_oral = 1). Gases, diagnostics, CNS-acting inhaled drugs and gut-local oral drugs were removed; see cleaning_log.",
        example="SELECT route_group, COUNT(*) AS molecules,\n       ROUND(AVG(alogp), 2) AS mean_logp,\n       ROUND(AVG(psa), 1) AS mean_psa\nFROM inhalation_dataset\nGROUP BY route_group;"),
    "activities_clean": dict(
        layer="clean", built_by="src/cleaning.py",
        purpose="PDE4 measurements that passed every cleaning rule, with the potency converted to a logarithmic scale (pIC50 or pKi).",
        grain="one clean measurement", key="activity_id",
        use_for="Potency analysis at the level of single measurements.",
        watch_out="The same molecule can be measured many times; use activity_potency for one value per molecule and target.",
        example="SELECT target_name, standard_type, COUNT(*) AS n,\n       ROUND(AVG(p_activity), 2) AS mean_p\nFROM activities_clean\nGROUP BY target_name, standard_type;"),
    "activity_potency": dict(
        layer="clean", built_by="src/cleaning.py",
        purpose="Repeated measurements collapsed into one median potency per molecule, target and measurement type, with a consistency check.",
        grain="one molecule on one target with one measurement type", key="molecule_chembl_id + target_chembl_id + standard_type",
        use_for="Ranking compounds by potency and comparing isoforms.",
        watch_out="inconsistent = 1 marks pairs whose measurements differ more than 10-fold: treat their median with caution.",
        example="SELECT m.pref_name, p.target_name, p.p_median, p.n_measurements\nFROM activity_potency AS p\nJOIN molecules AS m ON m.chembl_id = p.molecule_chembl_id\nWHERE m.pref_name IS NOT NULL\nORDER BY p.p_median DESC\nLIMIT 20;"),
    "cleaning_log": dict(
        layer="ledger", built_by="src/cleaning.py",
        purpose="The cleaning ledger: every rule applied, in order, with how many rows it removed or flagged and why.",
        grain="one cleaning rule (or check) applied to one dataset", key="rule_id",
        use_for="Explaining the data preparation to a reviewer: from the starting rows to the analysed rows, rule by rule.",
        watch_out="rows_removed counts only the rows removed by that rule; earlier rules already removed others.",
        example="SELECT dataset, rule_id, kind, rule, rows_before, rows_removed, rows_after\nFROM cleaning_log\nORDER BY dataset, step;"),
    "cleaning_exclusions": dict(
        layer="ledger", built_by="src/cleaning.py",
        purpose="Every excluded record with the rule that excluded it: the evidence behind the counts in cleaning_log.",
        grain="one excluded record", key="dataset + record_id",
        use_for="Answering 'why is this molecule not in my analysis?'.",
        watch_out="record_id holds an activity_id for dataset 'activities' and a chembl_id for dataset 'molecules'.",
        example="SELECT e.rule_id, l.rule, m.pref_name\nFROM cleaning_exclusions AS e\nJOIN cleaning_log AS l ON l.rule_id = e.rule_id\nJOIN molecules AS m ON m.chembl_id = e.record_id\nWHERE e.dataset = 'molecules'\nORDER BY e.rule_id;"),
}

TABLES.update({
    "pde4_chemical_space": dict(
        layer="analysis", built_by="src/pde4_chemical_space.py",
        purpose="Every PDE4 compound with a structure and a clean potency, placed on a chemical space map and assigned to one of eight k-means families of similar substructures.",
        grain="one compound", key="molecule_chembl_id",
        use_for="Plotting the chemical space (map_x vs map_y), comparing families, finding efficient or selective compounds.",
        watch_out="map_x and map_y are t-SNE coordinates without units: only neighbourhoods are meaningful, not distances between far-apart islands. Families are model groupings, not natural categories.",
        example="SELECT cluster_label, COUNT(*) AS compounds, ROUND(AVG(best_p), 2) AS mean_p\nFROM pde4_chemical_space\nGROUP BY cluster_label\nORDER BY mean_p DESC;"),
    "pde4_clusters": dict(
        layer="analysis", built_by="src/pde4_chemical_space.py",
        purpose="One summary row per k-means family: size, potency, properties, history, selectivity, a representative molecule and the named compounds.",
        grain="one family (cluster)", key="cluster",
        use_for="Comparing chemical families at a glance; the medoid is the structure to show for each family.",
        watch_out="Names were assigned by reading each family's motifs and medoid; medians hide the spread inside loose families (2 and 3).",
        example="SELECT cluster, cluster_label, n_molecules, ROUND(median_p, 2) AS median_p,\n       ROUND(100 * share_potent, 1) AS pct_potent\nFROM pde4_clusters\nORDER BY cluster;"),
    "pde4_k_selection": dict(
        layer="analysis", built_by="src/pde4_chemical_space.py",
        purpose="How k-means behaved for each number of clusters tried: the evidence for choosing k = 8.",
        grain="one value of k", key="k",
        use_for="Plotting the elbow, silhouette and stability curves.",
        watch_out="Silhouettes are low at every k because chemical families overlap; stability matters as much as separation.",
        example="SELECT k, ROUND(inertia, 0) AS inertia, ROUND(silhouette, 3) AS silhouette,\n       ROUND(stability_ari, 2) AS stability\nFROM pde4_k_selection\nORDER BY k;"),
})

# ----------------------------------------------------------------- columns
# Shared explanations for columns that appear in several tables: (meaning, unit)
COMMON = {
    "chembl_id": ("ChEMBL identifier of the molecule, e.g. CHEMBL714 for albuterol. Stable across ChEMBL releases.", ""),
    "molecule_chembl_id": ("ChEMBL identifier of the tested molecule. Joins to molecules.chembl_id.", ""),
    "target_chembl_id": ("ChEMBL identifier of the biological target. Joins to targets.target_chembl_id.", ""),
    "pref_name": ("Preferred name in ChEMBL, usually the international non-proprietary name (INN) in capitals. Empty for most research compounds.", ""),
    "first_approval": ("Year of the first approval of the molecule, by any route and in any country recorded by ChEMBL.", "year"),
    "mw_freebase": ("Molecular weight of the parent molecule without its salt.", "Da (g/mol)"),
    "alogp": ("Calculated octanol-water partition coefficient (Crippen ALogP): how much the molecule prefers fat to water. Higher = more lipophilic.", "log units"),
    "psa": ("Polar surface area: the surface belonging to oxygen, nitrogen and their hydrogens. A proxy for polarity and for how easily a molecule crosses membranes.", "Å²"),
    "hbd": ("Number of hydrogen-bond donors (OH and NH groups). Lipinski's rule of five expects at most 5.", "count"),
    "hba": ("Number of hydrogen-bond acceptors (mainly O and N atoms). Lipinski's rule of five expects at most 10.", "count"),
    "rtb": ("Number of rotatable bonds: single bonds the molecule can twist around. A measure of flexibility.", "count"),
    "aromatic_rings": ("Number of aromatic rings, such as benzene or pyridine rings.", "count"),
    "heavy_atoms": ("Number of non-hydrogen atoms. A size measure used for ligand efficiency.", "count"),
    "num_ro5_violations": ("How many of Lipinski's four rules the molecule breaks (MW ≤ 500, logP ≤ 5, HBD ≤ 5, HBA ≤ 10).", "0-4"),
    "qed_weighted": ("Quantitative estimate of drug-likeness: one score that combines eight properties. 1 = very typical oral drug, 0 = very unusual.", "0-1"),
    "assay_chembl_id": ("ChEMBL identifier of the experiment (assay) in which the value was measured.", ""),
    "assay_type": ("Kind of experiment: B = binding (does it bind the target?), F = functional (does it change the target's activity in a cell or tissue?).", ""),
    "document_chembl_id": ("ChEMBL identifier of the publication or patent that reported the value.", ""),
    "document_year": ("Year of the publication or patent.", "year"),
    "standard_type": ("Type of measurement: IC50 (concentration that halves the enzyme activity) or Ki (inhibition constant, a measure of binding strength).", ""),
    "standard_value": ("Measured value after ChEMBL standardisation. For IC50 and Ki a lower value means a more potent compound.", "nM"),
    "pchembl_value": ("ChEMBL's potency on a log scale: -log10 of the molar value. 9 = 1 nM, 6 = 1 µM. Only given for clean '=' values.", "log units"),
    "target_name": ("Short target name: PDE4A, PDE4B, PDE4C, PDE4D, or 'PDE4 family' when the isoform was not specified.", ""),
    "target_type": ("SINGLE PROTEIN for one isoform, PROTEIN FAMILY for assays on the family as a whole.", ""),
    "application_number": ("FDA application number. The prefix gives the type: NDA (new drug), ANDA (generic), BLA (biologic).", ""),
    "product_number": ("Number of the product inside its application (one per strength or form).", ""),
    "ingredient_name": ("Active ingredient name as written by FDA, in capitals, often including the salt (e.g. ALBUTEROL SULFATE).", ""),
    "route": ("Route of administration, e.g. INHALATION, ORAL, INTRAVENOUS.", ""),
    "n_products": ("Number of FDA products with this molecule and this route.", "count"),
    "level5": ("ATC level-5 code: one chemical substance in one therapeutic use, e.g. R03AC02 = salbutamol as a selective beta-2 agonist.", ""),
    "dataset": ("Which cleaned dataset the row belongs to: 'activities' (PDE4 measurements) or 'molecules' (inhaled vs oral analysis set).", ""),
    "rule_id": ("Identifier of the cleaning rule: A = activities rules, M = molecules rules, followed by the order. Joins cleaning_log to cleaning_exclusions.", ""),
}

# Table-specific explanations: (meaning, unit). These override COMMON.
COLUMNS = {
    "dataset_info.key": ("Name of the fact, e.g. chembl_release, built_at, sha256:<file>, retrieved_at:<file>.", ""),
    "dataset_info.value": ("Value of the fact, stored as text.", ""),

    "molecules.molecule_type": ("Kind of molecule: Small molecule, Protein, Antibody, Oligonucleotide, Unknown... Empty for many research compounds, where ChEMBL does not set it.", ""),
    "molecules.max_phase": ("Highest clinical phase reached: 4 = approved, 1-3 = clinical trials, empty = research only.", "phase"),
    "molecules.oral": ("ChEMBL flag: 1 if the drug has an oral product. ChEMBL has no inhalation flag, which is why FDA routes are used.", "0/1"),
    "molecules.parenteral": ("ChEMBL flag: 1 if the drug has an injectable (parenteral) product.", "0/1"),
    "molecules.topical": ("ChEMBL flag: 1 if the drug has a topical product (skin, eye...).", "0/1"),
    "molecules.prodrug": ("1 if the molecule is a prodrug: inactive until the body converts it (e.g. ciclesonide).", "0/1"),
    "molecules.natural_product": ("1 if the molecule is a natural product or derived from one.", "0/1"),
    "molecules.parent_chembl_id": ("ChEMBL ID of the parent molecule. For a salt it points to the free base or acid; for a parent it points to itself.", ""),
    "molecules.canonical_smiles": ("Chemical structure written as a SMILES string, one line of text that a program like RDKit can turn back into the molecule.", ""),
    "molecules.standard_inchi_key": ("InChIKey: a 27-character fingerprint of the structure, useful to match the same molecule across databases.", ""),
    "molecules.full_molformula": ("Molecular formula including any salt, e.g. C13H21NO3.", ""),
    "molecules.full_mwt": ("Molecular weight including the salt or counter-ion.", "Da (g/mol)"),
    "molecules.cx_logp": ("ChemAxon logP. Always empty: the ChEMBL 37 API does not return ChemAxon descriptors.", "log units"),
    "molecules.cx_logd": ("ChemAxon logD at pH 7.4. Always empty for the same reason.", "log units"),
    "molecules.cx_most_apka": ("ChemAxon most acidic pKa. Always empty for the same reason.", ""),
    "molecules.cx_most_bpka": ("ChemAxon most basic pKa. Always empty for the same reason.", ""),
    "molecules.molecular_species": ("ChemAxon charge class (ACID, BASE, NEUTRAL, ZWITTERION). Always empty for the same reason.", ""),
    "molecules.np_likeness_score": ("How much the structure resembles natural products. Positive = natural-product-like.", "score"),
    "molecules.source_set": ("Why the molecule is in the project: 'approved', 'pde4_tested', or 'approved+pde4_tested'.", ""),

    "molecule_synonyms.synonym": ("One alternative name of the molecule.", ""),
    "molecule_synonyms.syn_type": ("Source or kind of name: INN (international), USAN (US), BAN (British), JAN (Japanese), FDA, TRADE_NAME, RESEARCH_CODE, OTHER...", ""),

    "atc_classes.who_name": ("WHO name of the substance for this code.", ""),
    "atc_classes.level1": ("ATC level 1 (one letter): anatomical main group, e.g. R = respiratory system.", ""),
    "atc_classes.level1_description": ("Name of the level-1 group.", ""),
    "atc_classes.level2": ("ATC level 2 (letter + 2 digits): therapeutic subgroup, e.g. R03 = drugs for obstructive airway diseases.", ""),
    "atc_classes.level2_description": ("Name of the level-2 group.", ""),
    "atc_classes.level3": ("ATC level 3: pharmacological subgroup, e.g. R03A = adrenergics, inhalants.", ""),
    "atc_classes.level3_description": ("Name of the level-3 group.", ""),
    "atc_classes.level4": ("ATC level 4: chemical subgroup, e.g. R03AC = selective beta-2-adrenoreceptor agonists.", ""),
    "atc_classes.level4_description": ("Name of the level-4 group.", ""),

    "molecule_atc.chembl_id": ("ChEMBL ID of the molecule (often the salt form) that carries the code.", ""),
    "molecule_atc.level5": ("The ATC level-5 code. Joins to atc_classes.level5.", ""),

    "targets.pref_name": ("Full target name, e.g. 3',5'-cyclic-AMP phosphodiesterase 4B.", ""),
    "targets.target_type": ("SINGLE PROTEIN or PROTEIN FAMILY.", ""),
    "targets.organism": ("Species of the target: Homo sapiens for all rows.", ""),

    "activities.activity_id": ("ChEMBL identifier of the single measurement.", ""),
    "activities.standard_relation": ("Relation of the value: '=' exact, '<' or '>' only a limit (a censored value), empty if not given.", ""),
    "activities.standard_units": ("Units after ChEMBL standardisation: nM for almost all rows.", ""),
    "activities.published_type": ("Measurement type as written in the paper, before standardisation.", ""),
    "activities.published_relation": ("Relation as written in the paper.", ""),
    "activities.published_value": ("Value as written in the paper, in the paper's own unit.", "published unit"),
    "activities.published_units": ("Unit as written in the paper: nM, uM, mM, %... This mix is why the standardised columns are used.", ""),
    "activities.data_validity_comment": ("Warning added by ChEMBL curators, e.g. 'Outside typical range' or 'Potential transcription error'. Empty = no warning.", ""),

    "fda_applications.sponsor_name": ("Company that filed the application.", ""),
    "fda_products.brand_name": ("Brand (trade) name of the product, e.g. VENTOLIN HFA.", ""),
    "fda_products.dosage_form": ("Physical form of the product, e.g. AEROSOL, METERED; POWDER; TABLET.", ""),
    "fda_products.route": ("Route of administration as free text. May contain several routes separated by commas.", ""),
    "fda_products.marketing_status": ("Prescription, Over-the-counter, Discontinued, None (tentative approval)...", ""),
    "fda_products.reference_drug": ("Yes if the product is a reference listed drug, the one generics are compared with.", ""),
    "fda_product_ingredients.strength": ("Amount of the ingredient as written by FDA, e.g. 0.09MG/INH (per inhalation).", ""),

    "fda_ingredient_map.chembl_id": ("ChEMBL molecule whose name or synonym matched (may be a salt). Empty if unmatched.", ""),
    "fda_ingredient_map.parent_chembl_id": ("Parent of the matched molecule: the ID used to compare properties.", ""),
    "fda_ingredient_map.match_method": ("How the match was made: exact (same name), salt_stripped (matched after removing salt words), unmatched.", ""),

    "v_molecule_routes.chembl_id": ("Parent molecule.", ""),
    "v_molecule_routes.route": ("Raw FDA route text.", ""),
    "molecule_routes.chembl_id": ("Parent molecule.", ""),
    "molecule_routes.route": ("Raw FDA route text.", ""),
    "molecule_routes_clean.chembl_id": ("Parent molecule.", ""),
    "molecule_routes_clean.route": ("One single route in upper case.", ""),

    "inhalation_dataset.route_group": ("Analysis group: 'inhaled' (has at least one inhaled product) or 'oral' (oral products only).", ""),
    "inhalation_dataset.also_oral": ("1 if an inhaled molecule also has oral products. Useful to repeat an analysis without them.", "0/1"),
    "inhalation_dataset.n_inhaled_products": ("Number of FDA inhaled products containing the molecule.", "count"),
    "inhalation_dataset.n_oral_products": ("Number of FDA oral products containing the molecule.", "count"),
    "inhalation_dataset.atc_codes": ("Comma-separated ATC codes of the molecule and of its salt forms.", ""),

    "activities_clean.activity_id": ("ChEMBL identifier of the measurement. Joins to activities.activity_id to see the raw row.", ""),
    "activities_clean.p_activity": ("Potency on a log scale computed by the project: 9 - log10(value in nM). pIC50 for IC50 rows, pKi for Ki rows.", "log units"),
    "activities_clean.pchembl_value": ("ChEMBL's own log potency, kept to check the project's calculation.", "log units"),
    "activities_clean.pchembl_mismatch": ("1 if p_activity and pchembl_value differ by more than 0.05: a calculation check. 0 for every row in this build.", "0/1"),

    "activity_potency.p_median": ("Median of the log potencies: the summary value to use.", "log units"),
    "activity_potency.p_min": ("Lowest log potency among the repeated measurements.", "log units"),
    "activity_potency.p_max": ("Highest log potency among the repeated measurements.", "log units"),
    "activity_potency.p_range": ("p_max - p_min: how much the repeated measurements disagree. 1 = a 10-fold difference.", "log units"),
    "activity_potency.n_measurements": ("Number of clean measurements combined.", "count"),
    "activity_potency.n_documents": ("Number of different publications behind those measurements.", "count"),
    "activity_potency.inconsistent": ("1 if p_range > 1, i.e. measurements more than 10-fold apart.", "0/1"),

    "cleaning_log.step": ("Order in which the rules were applied within the dataset.", ""),
    "cleaning_log.kind": ("Type of step: input (starting rows), exclude (removes rows), transform (changes values), check (flags rows), aggregate (combines rows).", ""),
    "cleaning_log.rule": ("Short description of what the rule does.", ""),
    "cleaning_log.reason": ("Why the rule is needed.", ""),
    "cleaning_log.unit": ("What one row counts at this step: measurements, molecules or molecule-target pairs.", ""),
    "cleaning_log.rows_before": ("Rows before the rule.", "count"),
    "cleaning_log.rows_removed": ("Rows removed by this rule only.", "count"),
    "cleaning_log.rows_after": ("Rows left after the rule.", "count"),
    "cleaning_log.rows_flagged": ("Rows marked by a check but kept.", "count"),
    "pde4_chemical_space.cluster": ("k-means family number, 1-8, ordered by size (1 = largest).", ""),
    "pde4_chemical_space.cluster_label": ("Chemical name of the family, e.g. 'Roflumilast type (difluoromethoxy catechol)'.", ""),
    "pde4_chemical_space.map_x": ("Horizontal position on the t-SNE map. Has no unit; nearby points are similar structures.", ""),
    "pde4_chemical_space.map_y": ("Vertical position on the t-SNE map.", ""),
    "pde4_chemical_space.best_p": ("Highest median pIC50 or pKi of the compound over the PDE4 targets.", "log units"),
    "pde4_chemical_space.p_pde4b": ("Median potency on PDE4B, when measured.", "log units"),
    "pde4_chemical_space.p_pde4d": ("Median potency on PDE4D, when measured.", "log units"),
    "pde4_chemical_space.selectivity_bd": ("p_pde4b - p_pde4d: +1 means ten-fold more potent on PDE4B (the isoform linked to anti-inflammatory effects), -1 ten-fold on PDE4D (linked to nausea).", "log units"),
    "pde4_chemical_space.ligand_efficiency": ("Potency per heavy atom: 1.37 x best_p / heavy_atoms. About 0.3 or more is often considered good.", "kcal/mol per heavy atom"),
    "pde4_chemical_space.lipophilic_efficiency": ("best_p - alogp (LLE): potency not explained by lipophilicity. 5 or more is often quoted for good candidates.", "log units"),
    "pde4_chemical_space.first_year": ("Year of the first paper or patent reporting a clean measurement of the compound.", "year"),
    "pde4_chemical_space.n_documents": ("Number of papers or patents reporting clean measurements.", "count"),
    "pde4_chemical_space.inhaled_paper": ("1 if the compound appears in a paper whose title is about inhaled PDE4 inhibitors.", "0/1"),
    "pde4_chemical_space.is_medoid": ("1 for the family's medoid: the real compound closest to the k-means centre.", "0/1"),
    "pde4_clusters.cluster": ("Family number, 1-8.", ""),
    "pde4_clusters.cluster_label": ("Chemical name of the family.", ""),
    "pde4_clusters.top_motif": ("Most enriched substructure motif with at least 40% prevalence; empty for mixed families.", ""),
    "pde4_clusters.motif_share": ("Share of the family carrying top_motif.", "0-1"),
    "pde4_clusters.n_molecules": ("Number of compounds in the family.", "count"),
    "pde4_clusters.median_p": ("Median best_p of the family.", "log units"),
    "pde4_clusters.share_potent": ("Share of compounds with best_p >= 8 (IC50 <= 10 nM).", "0-1"),
    "pde4_clusters.median_mw": ("Median molecular weight.", "Da"),
    "pde4_clusters.median_alogp": ("Median ALogP.", "log units"),
    "pde4_clusters.median_le": ("Median ligand efficiency.", "kcal/mol per heavy atom"),
    "pde4_clusters.median_first_year": ("Median year in which the family's compounds were first reported.", "year"),
    "pde4_clusters.n_selectivity": ("Compounds measured on both PDE4B and PDE4D.", "count"),
    "pde4_clusters.median_selectivity_bd": ("Median PDE4B - PDE4D difference.", "log units"),
    "pde4_clusters.n_inhaled_paper": ("Compounds from papers about inhaled PDE4 inhibitors.", "count"),
    "pde4_clusters.medoid_chembl_id": ("ChEMBL ID of the medoid, the most central real compound of the family.", ""),
    "pde4_clusters.medoid_smiles": ("Structure of the medoid as SMILES.", ""),
    "pde4_clusters.named_compounds": ("Named compounds in the family, most potent first (up to 8).", ""),
    "pde4_k_selection.k": ("Number of clusters tried.", ""),
    "pde4_k_selection.inertia": ("Within-cluster sum of squared distances to the centres; always falls as k grows.", ""),
    "pde4_k_selection.silhouette": ("Mean silhouette on a 3,000-compound sample: -1 to 1, higher = better separated.", ""),
    "pde4_k_selection.stability_ari": ("Mean adjusted Rand index between runs with 5 random seeds: 1 = identical groupings.", "0-1"),
    "cleaning_exclusions.record_id":("ID of the excluded record: an activity_id for 'activities', a chembl_id for 'molecules'.", ""),
}

# Join keys: (from table, from column, to table, to column)
JOINS = [
    ("molecules", "parent_chembl_id", "molecules", "chembl_id"),
    ("molecule_synonyms", "chembl_id", "molecules", "chembl_id"),
    ("molecule_atc", "chembl_id", "molecules", "chembl_id"),
    ("molecule_atc", "level5", "atc_classes", "level5"),
    ("activities", "molecule_chembl_id", "molecules", "chembl_id"),
    ("activities", "target_chembl_id", "targets", "target_chembl_id"),
    ("fda_products", "application_number", "fda_applications", "application_number"),
    ("fda_product_ingredients", "application_number", "fda_applications", "application_number"),
    ("fda_product_ingredients", "ingredient_name", "fda_ingredient_map", "ingredient_name"),
    ("fda_ingredient_map", "parent_chembl_id", "molecules", "chembl_id"),
    ("molecule_routes", "chembl_id", "molecules", "chembl_id"),
    ("molecule_routes_clean", "chembl_id", "molecules", "chembl_id"),
    ("inhalation_dataset", "chembl_id", "molecules", "chembl_id"),
    ("activities_clean", "activity_id", "activities", "activity_id"),
    ("activities_clean", "molecule_chembl_id", "molecules", "chembl_id"),
    ("activities_clean", "target_chembl_id", "targets", "target_chembl_id"),
    ("activity_potency", "molecule_chembl_id", "molecules", "chembl_id"),
    ("activity_potency", "target_chembl_id", "targets", "target_chembl_id"),
    ("cleaning_exclusions", "rule_id", "cleaning_log", "rule_id"),
    ("pde4_chemical_space", "molecule_chembl_id", "molecules", "chembl_id"),
    ("pde4_chemical_space", "cluster", "pde4_clusters", "cluster"),
    ("pde4_clusters", "medoid_chembl_id", "molecules", "chembl_id"),
]

GLOSSARY = [
    ("Table", "A grid of rows and columns about one kind of thing (molecules, products, measurements)."),
    ("Grain", "What one row represents. Always check it before counting: 'one row = one molecule' and 'one row = one measurement' give very different totals."),
    ("Primary key", "The column (or columns) that identify a row uniquely, like chembl_id in molecules. No two rows share it."),
    ("Join key", "A column that points to the primary key of another table, so the two tables can be combined with JOIN."),
    ("View", "A saved query that looks like a table but is recalculated each time. Here v_molecule_routes."),
    ("Materialised table", "The stored result of a view, so it does not have to be recalculated. Here molecule_routes."),
    ("NULL", "An empty cell: the value is unknown or does not apply. NULL is not zero, and COUNT(column) skips it."),
    ("ChEMBL ID", "Identifier given by the ChEMBL database to molecules (CHEMBL714), targets, assays and documents."),
    ("Parent and salt", "Many drugs are sold as salts (albuterol sulfate). The parent is the active molecule without the counter-ion; properties are compared on the parent."),
    ("SMILES", "A one-line text code for a chemical structure, readable by chemistry software."),
    ("IC50", "The concentration of a compound that reduces the enzyme activity by half. Lower IC50 = more potent."),
    ("Ki", "The inhibition constant, a measure of how tightly a compound binds its target. Lower Ki = tighter binding."),
    ("pIC50 / pKi", "The same values on a log scale: 9 - log10(value in nM). Higher = more potent; +1 = 10 times more potent."),
    ("Censored value", "A result reported only as a limit, like '> 10,000 nM'. It is not an exact measurement, so it is excluded from potency estimates."),
    ("logP", "How much a molecule prefers an oily phase to water. Lipophilic drugs (high logP) tend to stay longer in membranes and tissues."),
    ("PSA", "Polar surface area: the part of the molecular surface made by oxygen, nitrogen and their hydrogens. High PSA = more polar, crosses membranes less easily."),
    ("Rule of five", "Lipinski's guideline for oral absorption: MW ≤ 500, logP ≤ 5, at most 5 H-bond donors and 10 acceptors."),
    ("ATC code", "WHO classification of drugs by organ system and therapeutic use, in five levels (R → R03 → R03A → R03AC → R03AC02)."),
    ("NDA / ANDA / BLA", "Types of FDA application: new drug, abbreviated (generic) new drug, biologics license."),
    ("Route of administration", "How a product enters the body: inhalation, oral, intravenous..."),
]

NUMERIC = {"INTEGER", "REAL"}


def describe(con: sqlite3.Connection, table: str, column: str, declared: str) -> dict:
    q = f'"{column}"'
    t = f'"{table}"'
    total, filled, distinct = con.execute(f"SELECT COUNT(*), COUNT({q}), COUNT(DISTINCT {q}) FROM {t}").fetchone()
    info = {"filled": filled, "distinct": distinct, "total": total}
    if filled == 0:
        return info
    values = [r[0] for r in con.execute(f"SELECT {q} FROM {t} WHERE {q} IS NOT NULL")]
    numbers = [v for v in values if isinstance(v, (int, float))]
    if declared.upper() in NUMERIC and len(numbers) == len(values) and distinct > 2:
        info.update(min=min(numbers), median=statistics.median(numbers), max=max(numbers))
    elif distinct <= 12:
        top = con.execute(f"SELECT {q}, COUNT(*) FROM {t} WHERE {q} IS NOT NULL GROUP BY {q} ORDER BY 2 DESC, 1 LIMIT 6").fetchall()
        info["top"] = [[v, n] for v, n in top]
    else:
        examples = con.execute(f"SELECT DISTINCT {q} FROM {t} WHERE {q} IS NOT NULL ORDER BY rowid LIMIT 3").fetchall() \
            if table != "v_molecule_routes" else con.execute(f"SELECT DISTINCT {q} FROM {t} WHERE {q} IS NOT NULL LIMIT 3").fetchall()
        info["examples"] = [str(v[0])[:80] for v in examples]
    return info


def main() -> None:
    con = sqlite3.connect(DB)
    objects = con.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') "
                          "AND name NOT LIKE 'sqlite_%' AND name != 'pca_scores'").fetchall()
    missing_tables = sorted({n for n, _ in objects} - set(TABLES))
    stale_tables = sorted(set(TABLES) - {n for n, _ in objects})
    assert not missing_tables and not stale_tables, f"tables without docs {missing_tables}, docs without table {stale_tables}"

    missing, used = [], set()
    tables = []
    for name, kind in objects:
        doc = TABLES[name]
        columns = []
        for _, col, declared, _, _, pk in con.execute(f'PRAGMA table_info("{name}")'):
            key = f"{name}.{col}"
            meaning, unit = COLUMNS.get(key) or COMMON.get(col) or (None, "")
            if meaning is None:
                missing.append(key)
                continue
            used.add(key)
            columns.append({"name": col, "type": declared or "", "pk": pk > 0, "meaning": meaning, "unit": unit,
                            **describe(con, name, col, declared or "")})
        rows = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        tables.append({"name": name, "kind": kind, "rows": rows, **doc, "columns": columns})
    stale = sorted(k for k in COLUMNS if k not in used)
    assert not missing, f"columns without an explanation: {missing}"
    assert not stale, f"explanations for columns that do not exist: {stale}"

    joins = []
    for ft, fc, tt, tc in JOINS:
        filled, matched = con.execute(
            f'SELECT COUNT(f."{fc}"), COUNT(t."{tc}") FROM "{ft}" AS f LEFT JOIN '
            f'(SELECT DISTINCT "{tc}" FROM "{tt}") AS t ON t."{tc}" = f."{fc}"').fetchone()
        joins.append({"from": ft, "from_col": fc, "to": tt, "to_col": tc, "filled": filled, "matched": matched})

    order = [layer for layer, *_ in LAYERS]
    tables.sort(key=lambda t: (order.index(t["layer"]), list(TABLES).index(t["name"])))
    info = dict(con.execute("SELECT key, value FROM dataset_info"))
    OUT.write_text(json.dumps({
        "built_at": info.get("built_at"), "chembl_release": info.get("chembl_release"),
        "chembl_release_date": info.get("chembl_release_date"), "openfda_last_updated": info.get("openfda_last_updated"),
        "layers": [{"id": i, "title": t, "text": x} for i, t, x in LAYERS],
        "glossary": [{"term": t, "text": x} for t, x in GLOSSARY],
        "tables": tables, "joins": joins,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    n_cols = sum(len(t["columns"]) for t in tables)
    print(f"{len(tables)} tables/views, {n_cols} columns documented, {len(joins)} joins -> {OUT}")
    for j in joins:
        if j["filled"] and j["matched"] < j["filled"]:
            print(f"  join {j['from']}.{j['from_col']} -> {j['to']}: {j['matched']}/{j['filled']} matched")


if __name__ == "__main__":
    main()
