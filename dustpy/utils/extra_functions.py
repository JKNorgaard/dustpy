"""Optional helpers for setting up simulations."""

import numpy as np
from dustpy import constants as c
from chemcomp.disks._disk_class import solve_viscous_heating_globally


def set_abundances(sim):
    """Set the initial fiducial composition from number abundances per H nucleus.
    """
    # Abundances of molecules 
    abundances = {
        "H2O": 2.6753631e-4,
        "CO": 5.38e-5,
        "CO2": 2.69e-5,
        "CH4": 2.69e-5,
        "C_decomposable": 1.2912e-4,
        "C_persistent": 3.228e-5,
        "He": 0.085,
    }
    # Subtract hydrogen bound in water and methane before forming H2.
    abundances["H2"] = (1 - 2 * abundances["H2O"] - 4 * abundances["CH4"]) / 2

    molecular_mass = {
        "H2O": 18.015, "CO": 28.010, "CO2": 44.009, "CH4": 16.043,
        "C_decomposable": 12.011, "C_persistent": 12.011,
        "H2": 2.016, "He": 4.002602,
    }
    
    rock_mass_fraction = 0.004177587431498567
    mass_per_H = {name: abundance * molecular_mass[name]
                  for name, abundance in abundances.items()}
    
    nonrock_mass_per_H = sum(mass_per_H.values())

    mass_fractions = {
        name: (1 - rock_mass_fraction) * mass / nonrock_mass_per_H
        for name, mass in mass_per_H.items()
    }

    mass_fractions["rock"] = rock_mass_fraction
    volatile_names = ("H2O", "CO", "CO2", "CH4")
    solid_names = (*volatile_names, "C_decomposable", "C_persistent", "rock")
    dust_fraction = sum(mass_fractions[name] for name in solid_names)
    gas_fraction = mass_fractions["H2"] + mass_fractions["He"]

    names = tuple(sim.volatiles.names)
    dust = np.asarray(sim.dust.Sigma)
    gas = np.asarray(sim.gas.Sigma)
    dust_sum = dust.sum(axis=-1)
    weights = np.divide(dust, dust_sum[:, None], out=np.zeros_like(dust),
                        where=dust_sum[:, None] > 0)
    total = gas + dust_sum
    # Unit component distribution: multiply by a disc mass fraction below.
    distribution = weights * total[:, None]
    sim.gas.Sigma[...] = gas_fraction * total
    sim.dust.Sigma[...] = dust_fraction * distribution
    for name in volatile_names:
        species = getattr(sim.volatiles, name)
        species.Sigmaice[...] = mass_fractions[name] * distribution
        species.Sigmaice.SigmaFloor = (
            mass_fractions[name] / dust_fraction * np.array(sim.dust.SigmaFloor))
        outer = species.Sigmaice.boundary.outer
        if outer.condition == "val":
            outer.value = 0.1 * species.Sigmaice.SigmaFloor[-1]
    sim.refractory_carbon.Sigma[...] = mass_fractions["C_decomposable"] * distribution
    sim.gas.update()
    sim.dust.update()


def add_viscous_heating(sim, Z=3.e-4):
    """Set and freeze the initial irradiation + viscous temperature.

    Call after ``sim.initialize()`` and before initializing volatiles.
    Requires Chemcomp; uses its opacity, abundance scaling and smoothing.

    Parameters
    ----------
    sim : Simulation
        Initialized simulation, with the desired initial gas surface density.
    Z : float or array, optional
        Small-grain dust-to-gas mass ratio, passed directly to Chemcomp.
        The default approximates the measured abundance of grains <= 3 microns
        near 1 au after 1,000 yr in the 10 L_sun profiling run (3.04e-4).
        It represents evolved inner-disk dust, not the initial MRN abundance
        (0.01), and is only a rough approximation at other radii. Chemcomp's
        existing double abundance scaling is retained. A radial array may
        be supplied instead.
    """


    if sim.gas.T is None:
        raise ValueError("Call sim.initialize() before add_viscous_heating().")

    r = np.asarray(sim.grid.r)
    Z = np.broadcast_to(np.asarray(Z, dtype=float), r.shape)

    T_irr = np.asarray(sim.gas.T)

    T = solve_viscous_heating_globally(
        sigma_g=np.asarray(sim.gas.Sigma),
        tirr=T_irr,
        T=T_irr.copy(),
        omega_k=np.sqrt(c.G * float(sim.star.M) / r**3),
        alpha=np.asarray(sim.gas.alpha),
        Z=Z,
        mu=np.asarray(sim.gas.mu) / c.m_p,
        r=r,
    )

    sim.gas.T[...] = T
    sim.gas.T.updater = None
    sim.gas.update()
    sim.dust.update()
