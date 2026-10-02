"""Numerical regression tests: signed identities, missingness, geometry and inference."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

class MetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name('metrics.py')
        cls.module_path = path
        if path.exists():
            spec = importlib.util.spec_from_file_location('organization_metrics', path)
            cls.m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.m)

    def setUp(self):
        self.assertTrue(self.module_path.exists(), 'numerical implementation is missing')

    def pick(self, rows, system, metric):
        return [r for r in rows if r['system'] == system and r['metric'] == metric][0]

    def test_significance_and_welch_missing(self):
        self.assertEqual([self.m.significance(p) for p in [.1,.05,.01,.001,.0009]], ['', '+','*','**','***'])
        self.assertEqual(self.m.significance(np.nan), 'NA')
        r = self.m.welch([1,2,3,np.nan], [0,1,2])
        self.assertAlmostEqual(r['difference'], 1)
        self.assertAlmostEqual(r['df'], 4)
        self.assertAlmostEqual(r['p'], .2878641347266908)
        self.assertTrue(self.m.welch([1], [2,3])['reason'])
        self.assertTrue(self.m.welch([1,1], [2,2])['reason'])

    def test_signed_identity_area_and_invalid(self):
        rows = self.m.identity(np.array([[-.1,-.2,-.4], [np.nan,1,2], [.6,.2,.1]]), [1,1,0], [1,3,2])
        r = self.pick(rows,'A','smax')
        self.assertAlmostEqual(r['value'], -.1)
        self.assertEqual(r['valid_area'], 1)
        self.assertEqual(r['total_area'], 4)
        self.assertAlmostEqual(self.pick(rows,'whole','smax')['value'], 1.1/3)
        self.assertTrue(self.pick(rows,'B','smax')['reason'])

    def test_spatial_denominator_includes_unassigned_and_fixed_bins(self):
        rows, curves = self.m.spatial([1,0,2], [1,2,1], [0,.5,1])
        self.assertEqual(self.pick(rows,'A','area_fraction')['value'], .25)
        self.assertEqual(self.pick(rows,'unassigned','area_fraction')['value'], .5)
        self.assertEqual(len(curves),80)
        self.assertEqual({c['metric'] for c in curves}, {'area_fraction'})
        self.assertEqual(sum(c['value'] for c in curves if c['system']=='B' and np.isfinite(c['value'])),1)

    def test_boundary_positions_retain_all_edges_without_forcing_one_cut(self):
        labels = np.array([1, 2, 3])
        axis = np.array([0.0, 0.2, 0.8])
        faces = np.array([[0, 1, 2]])
        rows = self.m.boundary_positions(labels, axis, faces, [True, True, True])
        self.assertEqual(
            sorted((row["system_pair"], row["position"]) for row in rows),
            [("A_B", 0.1), ("A_P", 0.4), ("B_P", 0.5)],
        )
        masked = self.m.boundary_positions(labels, axis, faces, [True, True, False])
        self.assertEqual([(row["system_pair"], row["position"]) for row in masked], [("A_B", 0.1)])

    def test_template_slope_and_signed_contrast(self):
        x=np.linspace(0,1,21); scores=np.column_stack((x,2*x,-x))
        rows, curves=self.m.template_axis(scores,np.ones(21),x)
        self.assertAlmostEqual(self.pick(rows,'rP_minus_rA','slope')['value'],-2)
        self.assertAlmostEqual(self.pick(rows,'rA','robust_range')['value'],.9)
        self.assertEqual(len(curves),80)

    def test_regional_mean_before_correlation_and_network_missing(self):
        ts=np.array([[1,2,4,3],[4,2,1,0],[1,3,2,4.]])
        roi=np.tile([1,2,3,4.],(400,1))
        z,reasons=self.m.regional_fc(ts,roi,[1,1,2],[True]*3)
        expected=np.arctanh(np.corrcoef(ts[:2].mean(axis=0),roi[0])[0,1])
        self.assertAlmostEqual(z[0,0],expected)
        self.assertTrue(np.isnan(z[2]).all()); self.assertTrue(reasons[2])
        registry=np.resize(['DMN','SMN','VAN','DAN','CON','LAN','VN'],400)
        net,names=self.m.network_fc(z,registry)
        self.assertEqual(names[0],'DMN')
        z[0,0]=np.nan
        net,_=self.m.network_fc(z,registry)
        self.assertTrue(np.isnan(net[0,0]))
        with self.assertRaises(ValueError): self.m.network_fc(z,['DMN']*400)

    def test_procrustes_preserves_scale_and_aligns_rotations(self):
        rng=np.random.default_rng(1); ref=rng.normal(size=(50,5)); q,_=np.linalg.qr(rng.normal(size=(5,5)))
        aligned,rows=self.m.aligned_gradient_metrics(ref@q,ref,np.arange(50),np.ones(50,bool))
        np.testing.assert_allclose(aligned,ref,atol=1e-12)
        self.assertAlmostEqual(self.pick(rows,'G1','reference_correlation')['value'],1)
        aligned,_=self.m.aligned_gradient_metrics(2*ref,ref,np.arange(50),np.ones(50,bool))
        np.testing.assert_allclose(aligned,2*ref,atol=1e-12)

    def test_kember_excludes_invalid_boundaries_and_respects_radius(self):
        rng=np.random.default_rng(4); ts=rng.normal(size=(6,20))
        pts=np.array([[0,0,0],[1,0,0],[2,0,0],[0,1,0],[1,1,0],[2,1,0.]])
        faces=np.array([[0,1,3],[1,3,4],[1,2,4],[2,4,5]])
        rows=self.m.kember(ts,[1,1,2,1,2,2],np.ones(6,bool),pts,faces,radius=.1)
        self.assertTrue(self.pick(rows,'A','boundary_sharpness')['reason'])
        rows=self.m.kember(ts,[1,1,2,1,2,2],np.zeros(6,bool),pts,faces)
        self.assertEqual(self.pick(rows,'A','boundary_sharpness')['n_valid_boundaries'],0)

    def test_valid_mask_is_not_misclassified_as_unassigned_or_zero(self):
        labels = np.array([1, 0, 2])
        area = np.array([1.0, 2.0, 1.0])
        valid = np.array([True, False, True])
        rows, curves = self.m.spatial(labels, area, [0, .5, 1], valid=valid)
        self.assertEqual(self.pick(rows, "A", "area_fraction")["value"], .25)
        self.assertEqual(self.pick(rows, "B", "area_fraction")["value"], .25)
        self.assertEqual(self.pick(rows, "unassigned", "area_fraction")["value"], 0)
        self.assertEqual(self.pick(rows, "unassigned", "area_mm2")["value"], 0)
        scores = np.array([[.2, .1, 0], [-.9, -.8, -.7], [.3, .2, .1]])
        identity = self.m.identity(scores, labels, area, valid=valid)
        self.assertEqual(self.pick(identity, "whole", "smax")["valid_area"], 2)


if __name__ == '__main__': unittest.main()
