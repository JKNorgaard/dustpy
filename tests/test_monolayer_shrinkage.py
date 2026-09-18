"""Checks for the one-pass monolayer correction; run with unittest discover."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from dustpy import Simulation, constants as c
from dustpy.std import chemistry, volatiles


def fixture():
    sim = Simulation(); sim.verbosity = 0
    sim.ini.grid.Nr = 12; sim.ini.grid.Nmbpd = 4; sim.ini.grid.mmax = 1e-4
    sim.initialize(); volatiles.add_volatiles(sim, ['H2O', 'CO'])
    sim.volatiles.chemistry_variable_radius = True
    sim.dust.Sigma[:] = 1.
    sim.volatiles.H2O.Sigmaice[:] = .2
    sim.volatiles.CO.Sigmaice[:] = chemistry.volatile_quantities(sim, 'CO')[0]
    sim.dust.Sigma.chemdelta[:] = -.5
    sim.dust.Sigma.chemdelta[0] = 0.
    sim.t._prevstepsize = c.year
    return sim


class MonolayerShrinkageTests(unittest.TestCase):
    def test_half_mass_loss_is_one_correction(self):
        sim = fixture()
        before = sim.volatiles.CO.Sigmaice.copy()
        ml = chemistry.volatile_quantities(sim, 'CO')[0]
        rm = sim.volatiles.CO.radius
        a1 = sim.dust.a*np.cbrt(.5)
        expected = ml*(1-((a1+rm)/(sim.dust.a+rm))**2)
        released = chemistry.monolayer_shrinkage(sim)
        np.testing.assert_allclose(released[1, 1], expected[1], rtol=1e-12)
        np.testing.assert_array_equal(sim.volatiles.CO.Sigmaice, before)
        np.testing.assert_array_equal(released[:, 0], 0)
        # Confirm the intentionally accepted ~0.071% residual for smallest grains.
        remaining_mass = .5*sim.dust.Sigma[1, 0]-released[:, 1, 0].sum()
        a2 = sim.dust.a[1, 0]*np.cbrt(remaining_mass/sim.dust.Sigma[1, 0])
        residual = ((a1[1, 0]+rm)/(a2+rm))**2-1
        self.assertAlmostEqual(float(residual), .0007106454765, places=10)

    def test_underfilled_and_thick_bins_unchanged(self):
        for coverage in [.1, 10.]:
            with self.subTest(coverage=coverage):
                sim = fixture(); sim.volatiles.CO.Sigmaice[:] *= coverage
                self.assertFalse(np.any(chemistry.monolayer_shrinkage(sim)[1]))

    def test_no_growth_or_zero_timestep_evaporation(self):
        for change, dt in [(0., c.year), (.2, c.year), (-.5, 0.)]:
            with self.subTest(change=change, dt=dt):
                sim = fixture(); sim.dust.Sigma.chemdelta[:] = change; sim.t._prevstepsize = dt
                self.assertFalse(np.any(chemistry.monolayer_shrinkage(sim)))

    def test_saturation_and_slow_rates(self):
        for mode in ['saturated', 'slow']:
            sim = fixture()
            if mode == 'saturated':
                sim.volatiles.CO.Sigmavap[:] = chemistry.volatile_quantities(sim, 'CO')[1]
            else: sim.dust.H[:] = 1e100
            self.assertFalse(np.any(chemistry.monolayer_shrinkage(sim)[1]))

    def test_updater_conserves_mass_and_remaps_once(self):
        sim = fixture()
        before_ice = np.stack([getattr(sim.volatiles, n).Sigmaice for n in sim.volatiles.names])
        before_vapor = np.stack([getattr(sim.volatiles, n).Sigmavap for n in sim.volatiles.names])
        expected_dust_mass = (sim.dust.Sigma+sim.dust.Sigma.chemdelta).sum(-1)
        before_gas = sim.gas.Sigma.copy()
        released = chemistry.monolayer_shrinkage(sim)
        with patch.object(chemistry, 'remapper', wraps=chemistry.remapper) as remap:
            volatiles.remap_updater(sim)
        self.assertEqual(remap.call_count, 1)
        after_ice = np.stack([getattr(sim.volatiles, n).Sigmaice for n in sim.volatiles.names])
        after_vapor = np.stack([getattr(sim.volatiles, n).Sigmavap for n in sim.volatiles.names])
        np.testing.assert_allclose(after_ice.sum(-1)+after_vapor,
                                   before_ice.sum(-1)+before_vapor, rtol=1e-12)
        np.testing.assert_allclose(after_vapor-before_vapor, released.sum(-1), rtol=1e-12)
        np.testing.assert_allclose(sim.dust.Sigma.sum(-1)+released.sum(axis=(0, 2)),
                                   expected_dust_mass, rtol=1e-12)
        np.testing.assert_allclose(sim.gas.Sigma, before_gas+released.sum(axis=(0, 2)))
        self.assertFalse(np.any(sim.dust.Sigma.chemdelta))


if __name__ == '__main__': unittest.main()
