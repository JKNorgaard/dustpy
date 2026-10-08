'''Module containing standard functions for setting up volatile species.'''
import numpy as np
import scipy.sparse as sp
import time
from astropy import constants as ac

from dustpy import std
from dustpy.std import ice_f, sim
from dustpy.utils.boundary import Boundary
from simframe.frame import Group
from dustpy.std.volatile_constants import VOLATILE_PROPERTIES

def chemistry_updater(sim):
    """Exchange phases with variable radius unless explicitly disabled."""

    # Run the sublimation and condensation chemistry for all volatile species
    has_carbon = hasattr(sim, 'refractory_carbon')
    if has_carbon:
        delta_ice, delta_vapor, delta_carbon = std.chemistry.solid_chemistry_all(sim)
    else:
        delta_ice, delta_vapor = std.chemistry.sublimation_condensation_all(sim)

    # Update the surface densities of each volatile species based on the changes in ice and vapor surface densities
    for x, name in enumerate(sim.volatiles.names):
        species = getattr(sim.volatiles, name)
        species.Sigmaice += delta_ice[x]
        species.Sigmavap += delta_vapor[x]
        # Enforce floor values for the updated surface densities of ice and vapor
        std.vapor.enforce_floor_value(species.Sigmavap)

    # Update the dust chemistry delta and gas surface density based on the total changes in ice and vapor
    sim.dust.Sigma.chemdelta += delta_ice.sum(axis=0)
    if has_carbon:
        sim.refractory_carbon.Sigma += delta_carbon
        sim.dust.Sigma.chemdelta += delta_carbon
        sim.gas.Sigma -= delta_ice.sum(axis=(0, 2))+delta_carbon.sum(axis=-1)
    else:
        sim.gas.Sigma += delta_vapor.sum(axis=0)

def finalize_volatiles(sim):
    """Finish solid transport, then apply chemistry and remapping."""
    try:
        if hasattr(sim, "refractory_carbon"):
            sim.refractory_carbon.Sigma.update()
    finally:
        # Carbon is the last solid tracer to use this solver.
        # Release it before chemistry and simulation output.
        sim.dust._ice_lu = None

    chemistry_updater(sim)
    remap_updater(sim)
    refresh_after_chemistry(sim)

def refresh_after_chemistry(sim):
    """Refresh coefficients and interior sources once, in dependency order.

    Uses the current fixed material properties, mixing parameters, molecular
    mass, alpha, fragmentation threshold, backreaction, torque and external
    sources. Evolving prescriptions for these require extending this order.
    """

    gas, dust = sim.gas, sim.dust


    def update_preserving_boundaries(field):
        boundary = field[[0, -1]].copy()
        try:
            field.update()
        finally:
            field[[0, -1]] = boundary

    gas.update()
    dust.update()

    # Flux divergence supplies the hydrodynamic sources used by the timestep.
    update_preserving_boundaries(gas.Fi)
    for name in ("adv", "diff", "tot"):
        update_preserving_boundaries(getattr(dust.Fi, name))
    update_preserving_boundaries(gas.S.hyd)
    update_preserving_boundaries(dust.S.hyd)
    update_preserving_boundaries(dust.S.coag)
    update_preserving_boundaries(gas.S.tot)
    update_preserving_boundaries(dust.S.tot)


def remap_updater(sim):
    """Apply solid chemistry mass changes, optionally redistributing mass bins.

       Set sim.volatiles.remapping = False to keep solids in their current
       mass bins. Chemistry, including the monolayer correction controlled
       by chemistry_variable_radius, still updates dust, ice, and gas.
    
        Parameters
        ----------
        sim : Frame
            Parent simulation frame
        """
    if getattr(sim.volatiles, 'chemistry_variable_radius', True):
        # Correct monolayers once using the radius after ordinary chemistry.
        # All species see the same geometry for this correction.
        released = std.chemistry.monolayer_shrinkage(sim)
        for i, volatile_name in enumerate(sim.volatiles.names):
            volatile = getattr(sim.volatiles, volatile_name)
            volatile.Sigmaice -= released[i]
            volatile.Sigmavap += released[i].sum(axis=-1)
        sim.dust.Sigma.chemdelta -= released.sum(axis=0)
        sim.gas.Sigma += released.sum(axis=(0, 2))

    # Apply chemistry mass exchange even when redistribution is disabled.
    has_carbon = hasattr(sim, 'refractory_carbon')
    if not getattr(sim.volatiles, 'remapping', True):
        dust_new = np.array(sim.dust.Sigma, copy=True)
        # Preserve the inner boundary, as in the remapper.
        dust_new[1:] += sim.dust.Sigma.chemdelta[1:]
        ice_new = np.stack([
            getattr(sim.volatiles, name).Sigmaice for name in sim.volatiles.names
        ])
        if has_carbon:
            carbon_new = np.array(sim.refractory_carbon.Sigma, copy=True)
    elif has_carbon:
        dust_new, ice_new, carbon_new = std.chemistry.remapper(sim, include_carbon=True)
    else:
        dust_new, ice_new = std.chemistry.remapper(sim)

    # Update the dust surface density and enforce floor value
    sim.dust.Sigma = dust_new
    std.dust.enforce_floor_value(sim)

    # Update the ice surface densities for each volatile species and enforce floor values
    for i, volatile_name in enumerate(sim.volatiles.names):
        getattr(sim.volatiles, volatile_name).Sigmaice = ice_new[i]
        std.ice.enforce_floor_value(getattr(sim.volatiles, volatile_name).Sigmaice)

    if has_carbon:
        sim.refractory_carbon.Sigma = carbon_new
        std.ice.enforce_floor_value(sim.refractory_carbon.Sigma)

    # Reset the chemistry delta for the next timestep
    sim.dust.Sigma.chemdelta *= 0


def add_refractory_carbon(sim, initial_distribution, prefactor=4.e13,
                          activation_temperature=24500.):
    """Label existing dust as refractory-organic carbon, after add_volatiles.

    The finite, nonnegative initial distribution has shape (Nr, Nm) and
    counts carbon mass [g/cm²], not total CHON mass. CH4 must be a volatile.
    Carbon plus ice must leave a positive nonvolatile grain remainder.
    Prefactor [s^-1] and activation temperature [K] are the adopted organic
    pyrolysis coefficients of Gail & Trieloff (2017), Appendix A, Table A.2.
    All released carbon forms CH4 using untracked background gas hydrogen.
    """

    if not hasattr(sim, 'volatiles') or 'CH4' not in sim.volatiles.names:
        raise ValueError('Add volatile CH4 before adding refractory carbon.')
    
    initial = np.array(initial_distribution, dtype=float, copy=True)
    dust = np.asarray(sim.dust.Sigma)

    # Validate coefficients before changing the simulation.
    std.chemistry.refractory_carbon_rate(500., prefactor, activation_temperature)
    ice = sum(np.asarray(getattr(sim.volatiles, n).Sigmaice) for n in sim.volatiles.names)

    if np.any((dust > sim.dust.SigmaFloor) & (initial+ice >= dust)):
        raise ValueError('Carbon plus ice must leave a positive nonvolatile grain remainder.')

    sim.addgroup('refractory_carbon', description='Carbon in refractory organic solids')
    carbon = sim.refractory_carbon
    carbon.addfield('mass', 12.011*ac.u.cgs.value, description='Carbon atomic mass [g]')
    carbon.addfield('prefactor', float(prefactor), description='Pyrolysis prefactor [1/s]')
    carbon.addfield('activation_temperature', float(activation_temperature), description='Pyrolysis activation temperature [K]')
    carbon.addfield('Sigma', initial.copy(), description='Solid carbon surface density [g/cm²]')
    field = carbon.Sigma
    field.boundary = Group(sim, description='Boundary conditions')
    field.SigmaFloor = np.zeros_like(dust)
    std.ice.set_initial_boundary(sim, field)
    std.ice.finalize_implicit(field)

    def carbon_updater(sim):
        carbon.Sigma = std.ice.evolve_implicit(sim, field)
        std.ice.finalize_implicit(field)

    # Called once by finalize_volatiles, after ordinary volatile transport.
    field.updater.updater = carbon_updater


def add_volatile(sim, volatile_name, initial_distribution):
    """Adds a single volatile species to the simulation frame.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame.
    volatile_name : str
        Name of the volatile species to be added.
    initial_distribution : ndarray
        Initial solid surface density distribution of the volatile.
    """

    ###################### GENERAL VOLATILE SPECIES SETUP ######################
    # Check that the requested volatile is implemented
    if volatile_name not in VOLATILE_PROPERTIES:
        raise ValueError(
            f"Volatile '{volatile_name}' is not implemented. "
            f"Available volatiles are: {list(VOLATILE_PROPERTIES)}"
        )

    # Species-specific properties
    properties = VOLATILE_PROPERTIES[volatile_name]

    # Create group for volatile species
    sim.volatiles.addgroup(
        volatile_name,
        description=f"{volatile_name} volatile species",
    )

    # Reference to volatile group
    volatile = getattr(sim.volatiles, volatile_name)

    # Add constants for the volatile species
    volatile.addfield(
        "radius",
        properties["radius"],
        description=f"Molecular radius of {volatile_name} [cm]",
    )

    volatile.addfield(
        "mass",
        properties["mass"],
        description=f"Molecular mass of {volatile_name} [g]",
    )

    # Saturation vapor pressure prescription
    volatile.vapor_pressure = properties["vapor_pressure"]


    ####################### SOLID VOLATILE SPECIES (ICE) ######################
    # Add field for the surface density of the volatile in the solid phase (ice)
    volatile.addfield(
        "Sigmaice",
        initial_distribution,
        description=(
            f"Surface density of {volatile_name} "
            "in the solid phase [g/cm^2]"
        ),
    )

    # Reference to the ice surface density field
    ice_field = volatile.Sigmaice

    # Initialize ice species and enforce floor / boundary conditions
    std.ice.initialize_ice(sim, ice_field)

    # Define and set updater for dynamic/collisional evolution of the ice species
    def ice_updater(sim):
        """Evolve the solid component of the volatile through transport and collisional processes."""

        volatile.Sigmaice = std.ice.evolve_implicit(
            sim,
            ice_field,
        )
        std.ice.finalize_implicit(ice_field)

    ice_field.updater.updater = ice_updater


    ######################## GASEOUS VOLATILE SPECIES (VAPOR) ##################
    # Add field for the surface density of the volatile in the vapor phase
    volatile.addfield(
        "Sigmavap",
        np.zeros_like(sim.grid.r),
        description=(
            f"Surface density of {volatile_name} "
            "in the vapor phase [g/cm^2]"
        ),
    )

    # Reference to the vapor surface density field
    vapor_field = volatile.Sigmavap

    # Initialize vapor species and enforce floor / boundary conditions
    std.vapor.initialize_vapor(sim, vapor_field)

    # Define and set updater for dynamic evolution of the vapor species
    def vapor_updater(sim):
        """Evolve the gaseous component of the volatile through transport processes."""

        volatile.Sigmavap = std.vapor.evolve_implicit(sim, vapor_field)

        std.vapor.finalize_implicit(vapor_field)

    vapor_field.updater.updater = vapor_updater



def add_volatiles(sim, volatiles, initial_distributions=None):
    """Function adds volatile species to the simulation frame

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    volatiles : list of str
        List of volatile species to be added to the simulation frame
    initial_distributions : list of float, optional
        List of initial distributions for each volatile species. 
        If not provided, the initial distribution will be set to a uniform distribution based on the dust surface density.

    Notes
    -----
    Remapping is enabled by default. After adding volatiles, set
    ``sim.volatiles.remapping = False`` to apply chemistry surface-density
    changes in the existing mass bins without redistributing solids.
    """

    # Create a new group for volatile species in the simulation frame
    sim.addgroup("volatiles", description="Volatile species")

    sim.volatiles.names = tuple(volatiles)
    sim.volatiles.remapping = True
    sim.volatiles.chemistry_variable_radius = True
    sim.volatiles.chemistry_radius_tolerance = 0.01

    # Constants and bookkeeping arrays for ice coagulation are initialized
    cstick_ice, cstick_ind_ice, Xi = ice_f.coagulation_parameters_ice(sim.grid.m, sim.ini.dust.erosionMassRatio, sim.grid.Nm)
    sim.dust.coagulation.stick_ice = cstick_ice
    sim.dust.coagulation.stick_ind_ice = cstick_ind_ice
    sim.dust.coagulation.Xi_ice = Xi
    sim.grid.B = np.mean(sim.grid.m[1:] / sim.grid.m[:-1])

    # Initialize bookkeeping array for chemistry contributions to the dust surface density
    sim.dust.Sigma.chemdelta = np.zeros_like(sim.dust.Sigma)   
    
    # If initial distributions for the ice are not provided, set them to a uniform distribution based on the dust surface density
    # Such that half of the dust surface density is distributed among the solid volatile species
    if initial_distributions is None:
        initial_distributions = np.array([sim.dust.Sigma / (2 * len(volatiles)) for _ in volatiles])
    
    # Add each volatile species to the simulation frame
    for i, volatile_name in enumerate(volatiles):
        add_volatile(sim, volatile_name, initial_distributions[i])
        getattr(sim.volatiles, volatile_name).updater = ['Sigmaice', 'Sigmavap']

    sim.updater = ['star', 'grid', 'gas', 'dust', 'volatiles']
    sim.volatiles.updater = volatiles
    sim.volatiles.updater.diastole = finalize_volatiles
