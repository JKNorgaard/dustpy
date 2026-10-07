"""Optional helpers for setting up simulations."""

import numpy as np
from dustpy import constants as c
from chemcomp.disks._disk_class import solve_viscous_heating_globally


def set_abundances(sim):
    """Set the initial fiducial composition from number abundances per H nucleus.
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


def add_viscous_heating(sim, a_cut=3.e-4):
    """Add viscous heating to the gas temperature.
    Sets the updater for the gas temperature to include viscous heating, and updates the gas and dust quantities.
    """

    T_irr = np.array(sim.gas.T, copy=True)


    def temperature_updater(sim):
        from dustpy import std

        r = np.asarray(sim.grid.r)
        Sigma_g = np.asarray(sim.gas.Sigma)

        # Calculate radii directly: dust.a is refreshed after gas.T.
        small = (std.dust.a(sim) <= a_cut)
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
            omega_k=np.sqrt(c.G * float(sim.star.M) / r**3),
            alpha=np.asarray(sim.gas.alpha),
            Z=abundance,
            mu=np.asarray(sim.gas.mu) / c.m_p,
            r=r,
        )

    sim.gas.T.updater.updater = temperature_updater
    sim.gas.update()
    sim.dust.update()
