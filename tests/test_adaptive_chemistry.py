"""Independent reference, local adaptivity, and conservation regressions."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.integrate import solve_ivp
from dustpy import Simulation, constants as c
from dustpy.std import chemistry, volatiles


class AdaptiveChemistryTests(unittest.TestCase):
    def test_wrapper_radius_tolerance_defaults_and_override(self):
        sim = Simulation()
        sim.verbosity = 0
        sim.ini.grid.Nr = 12
        sim.ini.grid.Nmbpd = 4
        sim.ini.grid.mmax = 1e-4
        sim.initialize()
        volatiles.add_volatiles(sim, ['H2O', 'CO'])
        sim.t._prevstepsize = 1.
        for configured, explicit, expected in [
                (0.01, None, 0.01), (0.005, None, 0.005),
                (0.005, 0.002, 0.002), (None, None, 0.01)]:
            with self.subTest(configured=configured, explicit=explicit):
                if configured is None:
                    del sim.volatiles.chemistry_radius_tolerance
                else:
                    sim.volatiles.chemistry_radius_tolerance = configured
                with patch.object(chemistry, '_adaptive_radius',
                                  wraps=chemistry._adaptive_radius) as adaptive:
                    di, dv = chemistry.sublimation_condensation_all(
                        sim, radius_tolerance=explicit)
                self.assertGreater(adaptive.call_count, 0)
                for call in adaptive.call_args_list:
                    self.assertEqual(call.kwargs['radius_tolerance'], expected)
                self.assertTrue(np.all(np.isfinite(di)))
                np.testing.assert_array_equal(dv, -di.sum(axis=-1))

    def test_fixed_radius_opt_out_exchange_and_remapping(self):
        sim = Simulation()
        sim.verbosity = 0
        sim.ini.grid.Nr = 12
        sim.ini.grid.Nmbpd = 4
        sim.ini.grid.mmax = 1e-4
        sim.initialize()
        volatiles.add_volatiles(sim, ['H2O', 'CO'])
        self.assertTrue(sim.volatiles.chemistry_variable_radius)
        sim.volatiles.chemistry_variable_radius = False
        sim.t._prevstepsize = c.year
        sim.gas.T[:] = 300.
        before = np.stack([getattr(sim.volatiles, n).Sigmaice.sum(-1)
                           + getattr(sim.volatiles, n).Sigmavap
                           for n in sim.volatiles.names])
        with patch.object(chemistry, '_adaptive_radius', side_effect=AssertionError('variable radius used')):
            with patch.object(chemistry, 'monolayer_shrinkage', side_effect=AssertionError('shrinkage used')):
                with patch.object(chemistry, '_constant_radius', wraps=chemistry._constant_radius) as fixed:
                    with patch.object(chemistry, 'remapper', wraps=chemistry.remapper) as remap:
                        volatiles.finalize_volatiles(sim)
        self.assertGreater(fixed.call_count, 0)
        self.assertEqual(remap.call_count, 1)
        after = np.stack([getattr(sim.volatiles, n).Sigmaice.sum(-1)
                          + getattr(sim.volatiles, n).Sigmavap
                          for n in sim.volatiles.names])
        np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-90)
        self.assertFalse(np.any(sim.dust.Sigma.chemdelta))

    def setUp(self):
        self.dust = np.array([.05, .8, 2.])
        self.radius = np.array([.1, .7, 2.])
        self.ice = np.array([.2*self.dust, .3*self.dust])
        self.vapor = np.array([1.5, .9])
        self.eq = np.array([.1, .2])
        self.rates = np.array([[1., .4, .2], [.7, .28, .14]])
        self.ml = np.full((2, 3), .0001)
        self.rm = np.array([.0001, .00015])

    def solve(self, dt=.3, **kwargs):
        values = dict(ice=self.ice, vapor=self.vapor, equilibrium=self.eq,
                      dust=self.dust, radius=self.radius, collision=self.rates,
                      monolayer=self.ml, molecular_radius=self.rm)
        values.update(kwargs)
        return chemistry._adaptive_radius(dt, **values)

    def test_reference_convergence_condensation_and_mixed(self):
        for eq in [self.eq, np.array([2., .2])]:
            def rhs(t, y):
                delta = y.reshape(self.ice.shape)
                return (self.rates*(1+delta.sum(0)/self.dust)**(2/3)
                        *(self.vapor-eq-delta.sum(1))[:, None]).ravel()
            reference = solve_ivp(rhs, (0., .01), np.zeros(self.ice.size),
                method='DOP853', rtol=1e-12, atol=1e-14).y[:, -1].reshape(self.ice.shape)
            loose = self.solve(.01, equilibrium=eq, radius_tolerance=.01)
            tight = self.solve(.01, equilibrium=eq, radius_tolerance=.001)
            self.assertLess(np.max(abs(tight-reference)), .2*np.max(abs(loose-reference)))
            np.testing.assert_allclose(tight, reference, rtol=.003, atol=1e-8)

    def test_species_permutation(self):
        for eq in [self.eq, np.array([5., .2]), np.array([5., 5.])]:
            forward = self.solve(1., equilibrium=eq)
            reverse = self.solve(1., ice=self.ice[::-1], vapor=self.vapor[::-1],
                equilibrium=eq[::-1], collision=self.rates[::-1],
                monolayer=self.ml[::-1], molecular_radius=self.rm[::-1])
            np.testing.assert_allclose(forward, reverse[::-1], rtol=1e-12, atol=1e-14)

    def test_stationary_cell_takes_one_trial(self):
        with patch.object(chemistry, '_constant_radius', wraps=chemistry._constant_radius) as step:
            np.testing.assert_array_equal(self.solve(1e8, equilibrium=self.vapor), 0.)
        self.assertEqual(step.call_count, 1)
        np.testing.assert_array_equal(self.solve(0.), 0.)

    def test_large_exchange_adapts_and_conserves(self):
        for eq in [self.eq, np.array([1e15, 1e20])]:
            with patch.object(chemistry, '_constant_radius', wraps=chemistry._constant_radius) as step:
                delta = self.solve(1e8, equilibrium=eq)
            self.assertGreater(step.call_count, 1)
            self.assertTrue(np.all(self.ice+delta >= -1e-15))
            self.assertTrue(np.all(self.dust+delta.sum(0) > 0.))
            vapor = self.vapor-delta.sum(1)
            self.assertTrue(np.all(vapor >= -1e-15))
            np.testing.assert_allclose((self.ice+delta).sum(1)+vapor,
                self.ice.sum(1)+self.vapor, rtol=1e-14)

    def test_fast_exchange_near_equilibrium_takes_one_trial(self):
        with patch.object(chemistry, '_constant_radius', wraps=chemistry._constant_radius) as step:
            delta = self.solve(1e8, equilibrium=self.vapor-1e-12,
                               collision=self.rates*1e20)
        self.assertEqual(step.call_count, 1)
        np.testing.assert_allclose(delta.sum(1), self.vapor-(self.vapor-1e-12), rtol=1e-14)

    def test_canceling_species_still_substep(self):
        dust = np.ones(1)
        with patch.object(chemistry, '_constant_radius', wraps=chemistry._constant_radius) as step:
            delta = chemistry._adaptive_radius(.1, np.array([[.4], [.2]]),
                np.array([0., 1.]), np.array([1., 0.]), dust, dust,
                np.ones((2, 1)), np.zeros((2, 1)), np.zeros(2))
        np.testing.assert_allclose(delta.sum(0), 0., atol=1e-14)
        self.assertGreater(step.call_count, 1)

    def test_underfilled_monolayer_is_not_filled(self):
        ice = np.array([[.001, .3]])
        delta = chemistry._constant_radius(1e20, ice, np.array([0.]),
            np.array([1e20]), np.ones((1, 2)), np.array([[.01, .02]]))
        self.assertEqual(delta[0, 0], 0.)
        np.testing.assert_allclose(ice[0, 1]+delta[0, 1], .02, atol=1e-15)

    def test_constant_radius_exponential(self):
        dt = .3
        result = chemistry._constant_radius(dt, self.ice, self.vapor,
            self.eq, self.rates, np.zeros_like(self.ice))
        total = self.rates.sum(1)
        expected = self.rates/total[:, None]*((self.vapor-self.eq)*(-np.expm1(-total*dt)))[:, None]
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_analytic_depletion_continues_with_remaining_population(self):
        ice = np.array([[.1, 1.]])
        result = chemistry._constant_radius(.1, ice, np.array([0.]),
            np.array([2.]), np.ones((1, 2)), np.full((1, 2), .02))
        event_time = -np.log1p(-.08)/2.
        # Both populations first lose .08; then only the second exchanges
        # with the remaining vapor deficit, 2 - 2*.08 = 1.84.
        expected = np.array([[-.08, -.08-1.84*(-np.expm1(-(.1-event_time)))]])
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_transport_all_species_before_one_chemistry_call(self):
        sim = Simulation()
        sim.verbosity = 0
        sim.ini.grid.Nr = 12
        sim.ini.grid.Nmbpd = 4
        sim.ini.grid.mmax = 1e-4
        sim.initialize()
        volatiles.add_volatiles(sim, ['H2O', 'CO'])
        events = []
        for name in sim.volatiles.names:
            species = getattr(sim.volatiles, name)
            species.Sigmaice.updater.updater = lambda sim, n=name: events.append(('ice', n))
            species.Sigmavap.updater.updater = lambda sim, n=name: events.append(('vapor', n))
        di = np.zeros((2, *sim.dust.Sigma.shape))
        dv = np.zeros((2, sim.grid.Nr))
        def exchange(sim):
            events.append(('chemistry', 'all'))
            return di, dv
        with patch.object(chemistry, 'sublimation_condensation_all', side_effect=exchange) as shared:
            with patch.object(volatiles, 'remap_updater') as remap:
                sim.volatiles.update()
        self.assertEqual(events, [('ice', 'H2O'), ('vapor', 'H2O'),
                                  ('ice', 'CO'), ('vapor', 'CO'), ('chemistry', 'all')])
        self.assertEqual(shared.call_count, 1)
        self.assertEqual(remap.call_count, 1)


    def test_real_heating_cooling_wrapper_and_atomic_failure(self):
        sim = Simulation()
        sim.verbosity = 0
        sim.ini.grid.Nr = 12
        sim.ini.grid.Nmbpd = 4
        sim.ini.grid.mmax = 1e-4
        sim.initialize()
        volatiles.add_volatiles(sim, ['H2O', 'CO'])
        sim.t._prevstepsize = c.year
        sim.volatiles.chemistry_variable_radius = True
        for temperature in [15., 300., 15.]:
            sim.gas.T[:] = temperature
            before = np.stack([getattr(sim.volatiles, n).Sigmaice.sum(-1)
                              +getattr(sim.volatiles, n).Sigmavap for n in sim.volatiles.names])
            volatiles.chemistry_updater(sim)
            after = np.stack([getattr(sim.volatiles, n).Sigmaice.sum(-1)
                             +getattr(sim.volatiles, n).Sigmavap for n in sim.volatiles.names])
            np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-90)
            volatiles.remap_updater(sim)
        before = sim.volatiles.H2O.Sigmaice.copy()
        with patch.object(chemistry, 'sublimation_condensation_all', side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError):
                volatiles.chemistry_updater(sim)
        np.testing.assert_array_equal(before, sim.volatiles.H2O.Sigmaice)
