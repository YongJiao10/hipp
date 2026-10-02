"""Contracts catch reordered ROIs, branch swaps, mutation and false density."""
import hashlib
from pathlib import Path
import unittest
import numpy as np
from scripts.visualize.iowa_hipp_functional_organization import inputs

class Contracts(unittest.TestCase):
    def test_registry_order_is_not_sorted_into_compliance(self):
        rows=[dict(roi_id=str(i),roi_name=f'r{i}',hemi='L' if i<=200 else 'R',canonical7='Default') for i in range(1,401)]
        rows[0],rows[1]=rows[1],rows[0]
        with self.assertRaisesRegex(ValueError,'order'):
            inputs.validate_registry(rows)

    def test_surface_area_conserves_triangle_area(self):
        p=np.array([[0,0,0],[2,0,0],[0,3,0]],float)
        np.testing.assert_allclose(inputs.vertex_areas(p,np.array([[0,1,2]])),[1,1,1])

    def test_anterior_to_posterior_does_not_depend_on_vertex_order(self):
        p=np.array([[181.25,-38.75,1.25],[200,0,1.25],[181.25,-20,1.25]])
        np.testing.assert_allclose(inputs.unfold_axis(p),[1,0,20/38.75])

    def test_temporal_mismatch_is_not_truncated(self):
        with self.assertRaisesRegex(ValueError,'time'):
            inputs.validate_arrays(np.ones((512,30)),np.ones((400,29)),np.ones(512), 'fixture')

    def test_declared_hash_mutation_fails(self):
        p=Path(__file__)
        with self.assertRaisesRegex(ValueError,'hash'):
            inputs.verify_record(dict(path=str(p),sha256='0'*64))

    def test_archive_is_rejected_before_open(self):
        with self.assertRaisesRegex(ValueError,'archive'):
            inputs.active_path('/tmp/archive_fake/never-open')

    def test_unbranched_standard_schema_is_explicit_not_a_branch_fallback(self):
        inputs.check_branch({}, 'standard', 'declared-standard', standard_schema=True)
        with self.assertRaisesRegex(ValueError, 'signal branch'):
            inputs.check_branch({'signal_branch': None}, 'standard', 'declared-standard', standard_schema=True)
        with self.assertRaisesRegex(ValueError, 'signal branch'):
            inputs.check_branch({}, 'ctxreg', 'declared-ctxreg', standard_schema=True)

if __name__=='__main__': unittest.main()
