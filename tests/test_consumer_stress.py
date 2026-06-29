import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stress_engine.stress.consumer import apply_consumer


class ConsumerStressTests(unittest.TestCase):
    def test_fico_to_pd_mapping_drives_consumer_el(self):
        fico_table = pd.DataFrame({"fico_min": [670], "fico_max": [739], "base_pd": [0.025]})
        row = pd.Series({"balance": 1000.0, "fico": 739.0, "current_el_rate": float("nan"), "maturity_formula": "longer_term"})
        result = apply_consumer(
            row,
            {"consumer": {"el_multiplier": 1.5, "value_loss_shock": -0.1}},
            {"el_rate_floor": 0.0, "el_rate_cap": 1.0, "default_consumer_lgd": 0.55},
            fico_table,
        )
        self.assertAlmostEqual(result["base_pd_from_fico"], 0.025)
        self.assertAlmostEqual(result["base_el_rate"], 0.01375)
        self.assertAlmostEqual(result["base_expected_loss"], 13.75)
        self.assertAlmostEqual(result["stressed_expected_loss"], 30.625)


if __name__ == "__main__":
    unittest.main()
