"""Check the charge rules on small chemical counterexamples, without a database.

These checks protect site counting and rule precedence; they do not validate
physiological speciation. Usage: py -3.13 tests/validate_charge.py
"""

from __future__ import annotations

import contextlib
import io
import sqlite3
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import compare_inhaled_oral as comparison
from compute_charge import charge_record


class ChargeRules(unittest.TestCase):
    def test_basic_sites_and_exclusions(self) -> None:
        for smiles, n in {"CCN": 1, "CN(C)C": 1, "NCCN": 2, "CC(=O)N": 0,
                          "Nc1ccccc1": 0, "CC#N": 0, "CS(=O)(=O)N": 0,
                          "NC#N": 0, "NC(=O)N": 0}.items():
            with self.subTest(smiles=smiles):
                self.assertEqual(charge_record(smiles)["n_basic_amine"], n)

    def test_strong_bases_count_once(self) -> None:
        for smiles in ("NC(=N)N", "CC(=N)N", "NC(=[NH2+])N"):
            result = charge_record(smiles)
            self.assertEqual(result["n_amidine_guanidine"], 1)
            self.assertEqual(result["charge_class"], "cation")

    def test_acids_and_esters(self) -> None:
        for smiles, n in {"CC(=O)O": 1, "CC(=O)[O-]": 1, "COC(C)=O": 0,
                          "c1nnn[nH]1": 1, "c1nn[nH]n1": 1, "c1nnn[n-]1": 1,
                          "CC(=O)NS(C)(=O)=O": 1, "OP(=O)(O)O": 1,
                          "COP(=O)(OC)OC": 0, "CS(=O)(=O)O": 1}.items():
            with self.subTest(smiles=smiles):
                self.assertEqual(charge_record(smiles)["n_acid"], n)

    def test_permanent_centres_and_resonance(self) -> None:
        for smiles, cls in {"C[N+](C)(C)C": "permanent cation", "C[n+]1ccccc1": "permanent cation",
                            "C[P+](C)(C)C": "permanent cation", "C[S+](C)C": "permanent cation",
                            "C[N+](C)(C)CC(=O)[O-]": "permanent cation",
                            "[O-][n+]1ccccc1": "neutral", "C[N+](=O)[O-]": "neutral",
                            "C[S+]([O-])C": "neutral", "CN=[N+]=[N-]": "neutral"}.items():
            with self.subTest(smiles=smiles):
                self.assertEqual(charge_record(smiles)["charge_class"], cls)

    def test_rule_precedence_and_stored_charge(self) -> None:
        for smiles, cls, rule, charge in (("NCC(=O)O", "zwitterion", "R2", 0),
                                          ("CCN", "cation", "R3", 0),
                                          ("CC(=O)[O-]", "anion", "R4", -1),
                                          ("CCO", "neutral", "R5", 0)):
            result = charge_record(smiles)
            self.assertEqual((result["charge_class"], result["rule_applied"], result["formal_charge"]),
                             (cls, rule, charge))
        with self.assertRaises(ValueError):
            charge_record("C(C")

    def test_optional_comparison_table(self) -> None:
        with contextlib.closing(sqlite3.connect(":memory:")) as con:
            with patch.object(comparison.sqlite3, "connect", return_value=con), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                inhaled = pd.DataFrame({"chembl_id": ["a", "b"], "cls": ["other inhaled"] * 2,
                                        "also_oral": [0, 0], "alogp": [1, 2]})
                self.assertNotIn("Without permanently charged", comparison.scenarios(inhaled))
                self.assertIn("skipping", output.getvalue())
                with con:
                    con.execute("CREATE TABLE molecule_charge (chembl_id TEXT, charge_class TEXT)")
                    con.execute("INSERT INTO molecule_charge VALUES ('a', 'permanent cation')")
                subset, _ = comparison.scenarios(inhaled)["Without permanently charged"]
                self.assertEqual(subset.chembl_id.tolist(), ["b"])


if __name__ == "__main__":
    unittest.main()
