"""Behavior tests for the valuable-only static clinical-association plots."""
import importlib.util
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def _module():
    path = ROOT / "clinical_associations_valuable.py"
    assert path.is_file()
    spec = importlib.util.spec_from_file_location("clinical_valuable", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(field, scale, metric, hemi, system, p, rho=0.3, n=93):
    return dict(branch="standard", population="all_adjusted", metric=metric,
                hemi=hemi, system=system, field=field, scale=scale,
                label=f"{field} label", n=n, rho=rho, ci_low=0.1, ci_high=0.5,
                p=p, status="ok", loo_min=0.2, loo_max=0.4,
                loo_sign_flip=False, delta_r2=0.05)


class ValuableTests(unittest.TestCase):
    def test_keeps_only_fields_with_nominal_hit_and_drops_null_scale(self):
        stats = pd.DataFrame([
            _row("keep_item", "STAI", "segregation", "R", "B", 0.01),
            _row("keep_item", "STAI", "area_percent", "R", "B", 0.4),
            _row("null_item", "GAD-7", "segregation", "L", "A", 0.5),
            _row("null_item", "GAD-7", "area_percent", "L", "A", 0.6),
            # non-primary rows must never create a keep
            dict(_row("margin_only", "STAI", "segregation_given_area", "L", "A", 0.001),
                 branch="standard", population="all_adjusted"),
            dict(_row("ctxreg_only", "STAI", "segregation", "L", "A", 0.001),
                 branch="ctxreg", population="all_adjusted"),
        ])
        module = _module()
        kept, sig, dropped = module.valuable(stats)
        self.assertEqual(set(sig["field"].unique()), {"keep_item"})
        self.assertEqual(set(kept["field"].unique()), {"keep_item"})
        self.assertIn("null_item", set(dropped["field"].unique()))
        # GAD-7 has no nominal primary hit anywhere
        self.assertNotIn("GAD-7", set(sig["scale"].unique()))

    def test_no_nominal_hits_fails_closed(self):
        stats = pd.DataFrame([_row("x", "STAI", "segregation", "L", "A", 0.5)])
        with self.assertRaises(ValueError):
            _module().valuable(stats)

    def test_totals_sort_before_items_within_scale(self):
        def trow(field, scale, level, p):
            row = _row(field, scale, "segregation", "L", "A", p)
            row["level"] = level
            return row
        stats = pd.DataFrame([
            trow("some_item", "PHQ", "item", 0.01),
            trow("the_total", "PHQ", "total", 0.02),
            trow("null_total", "GAD-7", "total", 0.5),
            trow("null_item", "STAI", "item", 0.6),
        ])
        module = _module()
        kept, sig, dropped = module.valuable(stats)
        # null totals are dropped like null items (overview.png keeps them)
        self.assertIn("null_total", set(dropped["field"].unique()))
        self.assertIn("null_item", set(dropped["field"].unique()))
        order = module.ordered_fields(sig)
        # within PHQ the total leads, so bold-vs-regular is unambiguous
        self.assertEqual([r["field"] for r in order],
                         ["the_total", "some_item"])
        self.assertEqual(order[0]["level"], "total")


if __name__ == "__main__":
    unittest.main()
