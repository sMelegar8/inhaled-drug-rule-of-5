DROP TABLE IF EXISTS molecule_charge;

-- Derived from stored SMILES by src/compute_charge.py; source tables stay intact.
-- Classes are ordered structural rules, not measured ionization or pKa estimates.
CREATE TABLE molecule_charge (
    chembl_id            TEXT PRIMARY KEY,  -- joins inhalation_dataset and molecules
    formal_charge        INTEGER NOT NULL,  -- RDKit total on the stored structure, no neutralization
    n_quaternary_n       INTEGER NOT NULL,  -- distinct permanently charged tetrahedral nitrogens
    n_basic_amine        INTEGER NOT NULL,  -- neutral sp3 amine sites passing the exclusion SMARTS
    n_amidine_guanidine  INTEGER NOT NULL,  -- distinct amidine/guanidine central carbons
    n_acid              INTEGER NOT NULL,  -- distinct carboxyl, tetrazole, acylsulfonamide, P/S acid sites
    n_other_permanent   INTEGER NOT NULL,  -- aromatic cationic N, phosphonium or sulfonium sites
    charge_class        TEXT NOT NULL CHECK (charge_class IN
                        ('permanent cation', 'zwitterion', 'cation', 'anion', 'neutral')), -- rule-based class
    rule_applied        TEXT NOT NULL CHECK (rule_applied IN
                        ('R1', 'R2', 'R3', 'R4', 'R5')) -- first matching rule in compute_charge.RULES
);
