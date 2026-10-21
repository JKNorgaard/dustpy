"""Optional helpers for setting up simulations."""

import numpy as np
from dustpy import constants as c
from chemcomp.disks._disk_class import solve_viscous_heating_globally


def set_abundances(sim, Mdisk=None):
    """Set the initial fiducial composition from number abundances per H nucleus.

    Parameters
    ----------
    sim : Simulation
        Initialized simulation to modify.
    Mdisk : float, optional
        Total gas plus dust mass on the simulation grid, in solar masses.
        Rescale the existing radial profile while retaining the fiducial
        component mass fractions. If omitted, retain the existing mass.
        Numerical surface density floors can add a negligible mass.
    """
    from dustpy import std

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

    # Empty bins contain positive placeholders proportional to grain mass.
    # Normalizing those values would seed the largest grains in empty annuli.
    dust = np.where(sim.dust.Sigma > sim.dust.SigmaFloor,
                    np.asarray(sim.dust.Sigma), 0.)
    gas = np.asarray(sim.gas.Sigma)
    dust_sum = dust.sum(axis=-1)
    populated = dust_sum > 0
    weights = np.divide(dust, dust_sum[:, None], out=np.zeros_like(dust),
                        where=dust_sum[:, None] > 0)
    total = gas + dust_sum
    if Mdisk is not None:
        if not np.isscalar(Mdisk) or not np.isfinite(Mdisk) or Mdisk <= 0:
            raise ValueError("Mdisk must be a finite, positive mass in solar masses.")
        current_mass = np.sum(total * np.asarray(sim.grid.A))
        if not np.isfinite(current_mass) or current_mass <= 0:
            raise ValueError("The existing disk must have a finite, positive mass.")
        # Use the disk's grain distribution where an annulus has no dust.
        # This preserves the composition without normalizing floor placeholders.
        if np.any(~populated & (total > 0)):
            grain_mass = np.sum(dust * np.asarray(sim.grid.A)[:, None], axis=0)
            if grain_mass.sum() <= 0:
                raise ValueError("The disk must contain populated dust bins to set abundances.")
            weights[~populated] = grain_mass / grain_mass.sum()
        total = total * (Mdisk * c.M_sun / current_mass)
        populated = np.ones_like(populated, dtype=bool)
    # Unit component distribution: multiply by a disc mass fraction below.
    distribution = weights * total[:, None]
    sim.gas.Sigma[...] = np.where(populated, gas_fraction * total, gas)
    sim.dust.Sigma[...] = dust_fraction * distribution
    std.dust.enforce_floor_value(sim)
    for name in volatile_names:
        species = getattr(sim.volatiles, name)
        species.Sigmaice[...] = mass_fractions[name] * distribution
        species.Sigmaice.SigmaFloor = (
            mass_fractions[name] / dust_fraction * np.array(sim.dust.SigmaFloor))
        outer = species.Sigmaice.boundary.outer
        if outer.condition == "val":
            outer.value = 0.1 * species.Sigmaice.SigmaFloor[-1]
        std.ice.enforce_floor_value(species.Sigmaice)
    sim.refractory_carbon.Sigma[...] = mass_fractions["C_decomposable"] * distribution
    std.ice.enforce_floor_value(sim.refractory_carbon.Sigma)
    sim.gas.update()
    sim.dust.update()


def add_viscous_heating(sim, a_cut=3.e-4, update=True):
    """Add viscous heating to the gas temperature.

    Calculate the initial heated temperature and refresh gas and dust quantities.
    By default, subsequent temperature updates recalculate viscous heating.

    Parameters
    ----------
    sim : Simulation
        Initialized simulation to modify.
    a_cut : float, optional
        Maximum grain radius contributing to the small-grain abundance [cm].
    update : bool, optional
        If False, disable temperature updates after the initial calculation,
        keeping the total gas temperature (irradiation plus viscous heating)
        fixed. Defaults to True.
    """

    T_irr = np.array(sim.gas.T, copy=True)

    def temperature_updater(sim):

        r = np.asarray(sim.grid.r)
        Sigma_g = np.asarray(sim.gas.Sigma)

        # Calculate radii directly: dust.a is refreshed after gas.T.
        small = (sim.dust.a <= a_cut)
        populated = sim.dust.Sigma > sim.dust.SigmaFloor
        Sigma_small = np.where(small & populated, sim.dust.Sigma, 0.).sum(axis=-1)
        abundance = np.divide(
            Sigma_small, Sigma_g, out=np.zeros_like(Sigma_g),
            where=Sigma_g > sim.gas.SigmaFloor,
        )

        return solve_viscous_heating_globally(
            sigma_g=Sigma_g,
            tirr=T_irr,
            T=np.array(sim.gas.T, copy=True),
            omega_k=sim.grid.OmegaK,
            alpha=np.asarray(sim.gas.alpha),
            Z=abundance,
            mu=np.asarray(sim.gas.mu) / c.m_p,
            r=r,
        )

    sim.gas.T.updater.updater = temperature_updater
    sim.gas.update()
    if not update:
        sim.gas.T.updater.updater = None
    sim.dust.update()
