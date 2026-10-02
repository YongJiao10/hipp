import pathlib
import sys
import unittest

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from clinical_associations_stats import analyze


class AnalyzeTests(unittest.TestCase):
    def test_adjustment_removes_group_only_confounding(self):
        rng = np.random.default_rng(8)
        groups = np.repeat([0.0, 1.0], 100)
        x = groups * 8.0 + rng.normal(0, 0.4, 200)
        y = groups * 9.0 + rng.normal(0, 0.4, 200)
        age = rng.normal(40, 5, 200)
        sex = np.tile([0.0, 1.0], 100)

        raw = analyze(x, y, age, sex, groups, adjusted=False,
                      permutations=199, bootstraps=100, loo=False, seed=4)
        adjusted = analyze(x, y, age, sex, groups, adjusted=True,
                           permutations=199, bootstraps=100, loo=False, seed=4)

        self.assertGreater(raw["rho"], 0.7)
        self.assertLess(abs(adjusted["rho"]), 0.3)

    def test_adjustment_retains_within_group_association(self):
        rng = np.random.default_rng(13)
        groups = np.repeat([0.0, 1.0], 30)
        within = rng.normal(size=60)
        x = within + groups * 5
        y = within * 0.8 + rng.normal(0, 0.2, 60) + groups * 6
        result = analyze(x, y, rng.normal(35, 4, 60), np.tile([0, 1], 30), groups,
                         permutations=199, bootstraps=100, loo=True, seed=3)
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["rho"], 0.8)
        self.assertLessEqual(result["p"], 0.01)
        self.assertFalse(result["loo_sign_flip"])

    def test_raw_permutation_matches_scalar_reference(self):
        x = np.array([7, 1, 6, 2, 5, 3, 4, 8, 10, 9], dtype=float)
        y = np.array([1, 5, 2, 8, 3, 7, 4, 6, 10, 9], dtype=float)
        groups = np.zeros(10)
        seed = 81
        draws = 31
        result = analyze(x, y, np.arange(10), np.zeros(10), groups,
                         adjusted=False, permutations=draws, bootstraps=0,
                         min_n=5, loo=False, seed=seed)
        xr = np.argsort(np.argsort(x)).astype(float) + 1
        yr = np.argsort(np.argsort(y)).astype(float) + 1
        observed = np.corrcoef(xr, yr)[0, 1]
        rng = np.random.default_rng(seed)
        extreme = 0
        for _ in range(draws):
            permuted = rng.permutation(yr - yr.mean()) + yr.mean()
            trial = np.corrcoef(xr, permuted)[0, 1]
            extreme += abs(trial) >= abs(observed) - 1e-12
        self.assertAlmostEqual(result["rho"], observed)
        self.assertEqual(result["p"], (1 + extreme) / (draws + 1))

    def test_missing_records_are_dropped_within_model(self):
        n = 25
        x = np.arange(n, dtype=float)
        y = x.copy()
        age = np.roll(np.arange(n, dtype=float), 7) + 20
        sex = np.tile([0.0, 1.0, 0.0, 1.0, 0.0], 5)
        groups = np.zeros(n)
        x[0] = np.nan
        y[1] = np.inf
        age[2] = np.nan
        result = analyze(x, y, age, sex, groups, permutations=19,
                         bootstraps=20, min_n=20, loo=False)
        self.assertEqual(result["n"], 22)
        self.assertAlmostEqual(result["rho"], 1.0)

    def test_ties_are_valid_but_constant_input_is_explicit(self):
        tied = np.repeat(np.arange(6), 5).astype(float)
        valid = analyze(tied, tied, np.arange(30), np.tile([0, 1], 15),
                        np.zeros(30), permutations=19, bootstraps=20, loo=False)
        constant = analyze(np.ones(30), tied, np.arange(30), np.tile([0, 1], 15),
                           np.zeros(30), permutations=19, bootstraps=20, loo=False)
        self.assertEqual(valid["status"], "ok")
        self.assertAlmostEqual(valid["rho"], 1.0)
        self.assertEqual(constant["status"], "constant_input")
        self.assertTrue(np.isnan(constant["p"]))

    def test_small_sample_is_descriptive_only(self):
        x = np.arange(12, dtype=float)
        result = analyze(x, x[::-1], x + 30, np.tile([0, 1], 6), np.zeros(12),
                         permutations=19, bootstraps=20, min_n=20, loo=True)
        self.assertEqual(result["status"], "insufficient_n")
        self.assertTrue(np.isnan(result["rho"]))
        self.assertTrue(np.isnan(result["p"]))
        self.assertTrue(np.isnan(result["ci_low"]))

    def test_rank_deficient_adjusted_design_fails_closed(self):
        n = 30
        x = np.arange(n, dtype=float)
        result = analyze(x, x + np.sin(x), x + 20, np.zeros(n), np.zeros(n),
                         permutations=19, bootstraps=20, loo=False)
        self.assertEqual(result["status"], "rank_deficient")
        self.assertIn("design", result["reason"])
        self.assertTrue(np.isnan(result["p"]))

    def test_seed_makes_all_resampling_outputs_deterministic(self):
        rng = np.random.default_rng(5)
        arrays = (rng.normal(size=30), rng.normal(size=30), rng.normal(size=30),
                  np.tile([0, 1], 15), np.repeat([0, 1], 15))
        one = analyze(*arrays, permutations=37, bootstraps=40, seed=99)
        two = analyze(*arrays, permutations=37, bootstraps=40, seed=99)
        self.assertEqual(one, two)

    def test_control_reports_incremental_rank_outcome_r_squared(self):
        rng = np.random.default_rng(1)
        n = 50
        control = rng.normal(size=n)
        x = rng.normal(size=n)
        y = 2 * control + 2 * x + rng.normal(0, 0.1, n)
        result = analyze(x, y, rng.normal(size=n), np.tile([0, 1], 25),
                         np.zeros(n), control=control, permutations=39,
                         bootstraps=30, loo=False)
        self.assertGreater(result["delta_r2"], 0.1)


if __name__ == "__main__":
    unittest.main()
