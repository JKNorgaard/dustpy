import os
import unittest
import numpy as np
from dustpy import Simulation, constants as c
from dustpy.std import volatiles


class ChemistryIntegrationTests(unittest.TestCase):
    def test_full_run_with_transport_collisions_and_floor_crossings(self):
        # Reproduced the reported failure around year 10 before the fix.
        # Include actual integrator/update ordering, not chemistry alone.
        sim = Simulation()
        sim.verbosity = 0
        sim.ini.grid.Nr = 12
        sim.ini.grid.Nmbpd = 4
        sim.ini.grid.mmax = 1e-4
        sim.initialize()
        volatiles.add_volatiles(sim, ['H2O', 'CO'])
        sim.t.snapshots = np.linspace(0., 1e5, 100)*c.year
        sim.writer = None
        sim.run()
        self.assertEqual(float(sim.t), 1e5*c.year)
        self.assertTrue(np.all(np.isfinite(sim.dust.Sigma)))
        self.assertFalse(np.any(sim.dust.Sigma.chemdelta))
        for name in sim.volatiles.names:
            sp = getattr(sim.volatiles, name)
            self.assertTrue(np.all(np.isfinite(sp.Sigmaice)))
            self.assertTrue(np.all(sp.Sigmaice >= 0.))
            self.assertTrue(np.all(sp.Sigmavap >= 0.))


    def test_ice_floor_tracks_absent_carrier_before_repopulation(self):
        from dustpy.std import ice as ice_module
        sim = Simulation()
        sim.verbosity = 0
        sim.ini.grid.Nr = 12
        sim.ini.grid.Nmbpd = 4
        sim.ini.grid.mmax = 1e-4
        sim.initialize()
        volatiles.add_volatiles(sim, ['H2O', 'CO'])
        r, k = 5, 8
        floor = float(sim.dust.SigmaFloor[r, k])
        sim.dust.Sigma[r, k] = .1*floor
        for name in sim.volatiles.names:
            field = getattr(sim.volatiles, name).Sigmaice
            field[r, k] = 1.5*floor  # old independently evolved remnant
            ice_module.enforce_floor_value(field)
            self.assertEqual(field[r, k], .1*field.SigmaFloor[r, k])
        sim.dust.Sigma[r, k] = 1.317*floor  # population returns above floor
        carried = sum(getattr(sim.volatiles, n).Sigmaice[r, k]
                      for n in sim.volatiles.names)
        self.assertLess(carried, sim.dust.Sigma[r, k])
        # The floor operation must not hide an invalid populated-bin state.
        field = sim.volatiles.H2O.Sigmaice
        field[r, k] = 2.*floor
        ice_module.enforce_floor_value(field)
        self.assertEqual(field[r, k], 2.*floor)


    @unittest.skipUnless(os.environ.get('DUSTPY_LONG_TESTS') == '1',
                         'Set DUSTPY_LONG_TESTS=1 for the full notebook run')
    def test_default_grid_notebook_snapshot_schedule(self):
        # Snapshot endpoints change the timestep sequence. A single endpoint
        # at 1000 yr missed the reported failure at 1713.658669631845 yr.
        sim = Simulation()
        sim.verbosity = 0
        sim.initialize()
        sim.t.snapshots = np.linspace(0., 1e5, 100)*c.year
        volatiles.add_volatiles(sim, ['H2O', 'CO'])
        sim.writer = None
        sim.run()
        self.assertEqual(float(sim.t), 1e5*c.year)
        self.assertTrue(np.all(np.isfinite(sim.dust.Sigma)))
        for name in sim.volatiles.names:
            sp = getattr(sim.volatiles, name)
            self.assertTrue(np.all(np.isfinite(sp.Sigmaice)))
            self.assertTrue(np.all(np.isfinite(sp.Sigmavap)))
            self.assertTrue(np.all(sp.Sigmaice >= 0.))
            self.assertTrue(np.all(sp.Sigmavap >= 0.))

