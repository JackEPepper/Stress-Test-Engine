import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stress_engine.stress.consumer import apply_consumer


class ConsumerStressTests(unittest.TestCase):
    def test_fico_mapping_respects_bounds(self):
        fico_table = pd.DataFrame({"fico_min": [670], "fico_max": [739], "base_pd": [0.025]})
        row = pd.Series({"balance": 1000.0, "fico": 739.0, "current_lgd": 0.5, "maturity_formula": "longer_term"})
        result = apply_consumer(row, {"consumer": {"pd_multiplier": 1.5, "value_loss_shock": -0.1}}, {}, fico_table)
        self.assertAlmostEqual(result["base_pd_from_fico"], 0.025)
        self.assertAlmostEqual(result["stressed_pd"], 0.0375)


if __name__ == "__main__":
    unittest.main()
