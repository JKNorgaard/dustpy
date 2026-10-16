'''Module containing standard functions for setting up volatile species.'''

import numpy as np
from astropy import constants as ac
from dustpy import std
from dustpy.std import ice_f
from simframe.frame import Group
from dustpy.std.volatile_constants import VOLATILE_PROPERTIES

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


    ####################### SOLID COMPONENT (ICE) ######################
    # Add field for the surface density of the solid components of the volatile species (ice)
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

    # Initialize ice species
    # Enforces floor / boundary conditions
    std.ice.initialize_ice(sim, ice_field)

    # Define and set updater for transport/collisional evolution of the ice species
    def ice_updater(sim):
        """Evolve the solid component of the volatile through transport and collisional processes."""

        volatile.Sigmaice = std.ice.evolve_implicit(
            sim,
            ice_field,
        )
        std.ice.finalize_implicit(ice_field)

    ice_field.updater.updater = ice_updater


    ######################## GASEOUS COMPONENT (VAPOR) ##################
    # Add field for the surface density of the gaseous components of the volatile species (vapor)
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

    # Initialize vapor species
    # Enforces floor / boundary conditions
    std.vapor.initialize_vapor(sim, vapor_field)

    # Define and set updater for transport evolution of the vapor species
    def vapor_updater(sim):
        """Evolve the gaseous component of the volatile through transport processes."""

        volatile.Sigmavap = std.vapor.evolve_implicit(sim, vapor_field)

        std.vapor.finalize_implicit(vapor_field)

    vapor_field.updater.updater = vapor_updater


def add_volatiles(sim, volatiles, initial_distributions=None):
    """Function adds list of volatile species to the simulation frame

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    volatiles : list of str
        List of volatile species to be added to the simulation frame
    initial_distributions : list of float, optional
        List of initial distributions for each volatile species. Each distribution should be of size (Nr,Nm).
        If not provided, the initial distribution will be set to a uniform distribution based on the dust surface density.
        Such that half of the dust surface density is distributed among the solid volatile species

    Notes
    -----
    Remapping is enabled by default. Set sim.volatiles.remapping = False to disable
    """

    # Create a new group for the volatile species in the simulation frame
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
    
    # Add each volatile species to the simulation frame and set their updaters for transport
    for i, volatile_name in enumerate(volatiles):
        add_volatile(sim, volatile_name, initial_distributions[i])
        getattr(sim.volatiles, volatile_name).updater = ['Sigmaice', 'Sigmavap']

    # Add volatile species to the updateorder
    sim.updater = ['star', 'grid', 'gas', 'dust', 'volatiles']
    # Set updateorder of volatile species 
    sim.volatiles.updater = volatiles
    # Set the diastole updater for the volatile species to finalize_volatiles, which applies chemistry and remapping after transport
    sim.volatiles.updater.diastole = finalize_volatiles


def add_refractory_carbon(sim, initial_distribution=None, prefactor=4.e13,
                          activation_temperature=24500.):
    """Adds a refractory carbon species to the simulation frame. This species undergoes pyrolysis, releasing carbon into the gas phase as CH4.
       The simulation frame must already contain the volatile species CH4 before adding refractory carbon. 
       The initial distribution of the refractory carbon is by default set to 20% of the dust surface density.
    """

    # Check if CH4 is present in the simulation
    if not hasattr(sim, 'volatiles') or 'CH4' not in sim.volatiles.names:
        raise ValueError('Add volatile CH4 before adding refractory carbon.')

    # Check if the initial distribution is valid
    dust = np.asarray(sim.dust.Sigma)
    initial_distribution = np.array(
        0.2*dust if initial_distribution is None else initial_distribution,
        dtype=float, copy=True)
    if (initial_distribution.shape != dust.shape
            or not np.all(np.isfinite(initial_distribution))
            or np.any(initial_distribution < 0)):
        raise ValueError('Carbon distribution must be finite, nonnegative and match dust.Sigma.')

    # Check if the initial distribution of refractory carbon and ice is valid
    ice = sum(np.asarray(getattr(sim.volatiles, n).Sigmaice) for n in sim.volatiles.names)
    if np.any((dust > sim.dust.SigmaFloor) & (initial_distribution+ice >= dust)):
        raise ValueError('Carbon plus ice must leave a positive nonvolatile grain remainder.')

    # Add refractory carbon group to the simulation frame
    sim.addgroup('refractory_carbon', description='Carbon in refractory organic solids')
    carbon = sim.refractory_carbon
    carbon.addfield('mass', 12.011*ac.u.cgs.value, description='Carbon atomic mass [g]')
    carbon.addfield('prefactor', float(prefactor), description='Pyrolysis prefactor [1/s]')
    carbon.addfield('activation_temperature', float(activation_temperature), description='Pyrolysis activation temperature [K]')
    carbon.addfield('Sigma', initial_distribution.copy(), description='Solid carbon surface density [g/cm²]')

    # Define and enforce boundary and floor conditions
    field = carbon.Sigma
    field.boundary = Group(sim, description='Boundary conditions')
    field.SigmaFloor = np.zeros_like(dust)
    std.ice.set_initial_boundary(sim, field)
    std.ice.finalize_implicit(field)

    # Define and set updater for transport/collisional evolution of the refractory carbon species
    def carbon_updater(sim):
        carbon.Sigma = std.ice.evolve_implicit(sim, field)
        std.ice.finalize_implicit(field)

    field.updater.updater = carbon_updater


def finalize_volatiles(sim):
    """Apply chemistry and remapping to the transported volatiles, and refractory carbon if it is present
       then update the dust and gas properties changed by chemistry and remapping."""
    try:
        # If refractory carbon is present, calculate its transport
        if hasattr(sim, "refractory_carbon"):
            sim.refractory_carbon.Sigma.update()
    finally:    
        # Carbon is the last solid tracer to use this solver.
        # Release it before chemistry and simulation output.
        sim.dust._ice_lu = None

    # Apply chemistry, remapping, and update the dust and gas properties changed by chemistry and remapping
    chemistry_updater(sim)
    remap_updater(sim)
    refresh_after_chemistry(sim)


def chemistry_updater(sim):
    """Runs sumlimation and condenstation for all volatile species. If refractory carbon is present
       this component goes through pyrolysis. Volatile, carbon and bulk gas surface densitites are updated to account for the changes caused by chemistry.
       The dust surface density is not updated here as it also requires remapping."""

    # Run the chemistry for all volatile species, and carbon if it is present.
    has_carbon = hasattr(sim, 'refractory_carbon')

    # Chemistry step, if carbon is not present, delta_carbon is just zeros
    delta_ice, delta_vapor, delta_carbon = std.chemistry.solid_chemistry_all(sim)

    # Update the surface densities of each volatile species based on the changes in ice and vapor surface densities
    for x, name in enumerate(sim.volatiles.names):
        species = getattr(sim.volatiles, name)
        species.Sigmaice += delta_ice[x]
        species.Sigmavap += delta_vapor[x]
        # Enforce floor values for the updated surface densities of vapor
        std.vapor.enforce_floor_value(species.Sigmavap)

    # The total change in dust surface density is found, this is used in the remapper
    sim.dust.Sigma.chemdelta += delta_ice.sum(axis=0) + delta_carbon

    # If carbon is present, update surface density and dust chemdelta
    if has_carbon:
        sim.refractory_carbon.Sigma += delta_carbon

    # Update the bulk gas surface density to account for the changes in ice and carbon surface densities
    sim.gas.Sigma -= (delta_ice.sum(axis=(0, 2)) + delta_carbon.sum(axis=-1))

def remap_updater(sim):
    """Updater for correcting monolayers and remapping the solid surface densities of volatiles, carbon and dust.
       After the chemistry step, bins that have become bare, have been reduced in radius allowing smaller monolayers.
       This is corrected, by allowing the excess ice above the monolayer to be released into the gas phase.
       The remapper then calculates the shift along the mass grid and remaps surface densities among their nearest neighbors."""

    # Apply chemistry mass exchange even when redistribution is disabled.
    has_carbon = hasattr(sim, 'refractory_carbon')
    if not getattr(sim.volatiles, 'remapping', True):
        # If remapping is disabled, we still need to update the surface densities.
        dust_new = np.array(sim.dust.Sigma, copy=True)

        # Preserve the inner and outer boundaries, as in the remapper.
        dust_new[1:-1] += sim.dust.Sigma.chemdelta[1:-1]
        ice_new = np.stack([getattr(sim.volatiles, name).Sigmaice for name in sim.volatiles.names])
        if has_carbon:
            carbon_new = np.array(sim.refractory_carbon.Sigma, copy=True)

    else:
        tracers = [
            np.asarray(getattr(sim.volatiles, name).Sigmaice)
            for name in sim.volatiles.names
        ]
        if has_carbon:
            tracers.append(np.asarray(sim.refractory_carbon.Sigma))

        dust_new, tracers_new = std.chemistry._remap_arrays(
            np.asarray(sim.dust.Sigma), np.stack(tracers),
            np.asarray(sim.dust.Sigma.chemdelta), float(sim.grid.B))

        if has_carbon:
            ice_new = tracers_new[:-1]
            carbon_new = tracers_new[-1]
        else:
            ice_new = tracers_new

    # Update the dust surface density and enforce floor value
    sim.dust.Sigma = dust_new
    std.dust.enforce_floor_value(sim)

    # Update the ice surface densities for each volatile species and enforce floor values
    for i, volatile_name in enumerate(sim.volatiles.names):
        getattr(sim.volatiles, volatile_name).Sigmaice = ice_new[i]
        std.ice.enforce_floor_value(getattr(sim.volatiles, volatile_name).Sigmaice)

    # Update the refractory carbon surface density and enforce floor value if it is present
    if has_carbon:
        sim.refractory_carbon.Sigma = carbon_new
        std.ice.enforce_floor_value(sim.refractory_carbon.Sigma)

    # Reset the chemistry delta for the next timestep
    sim.dust.Sigma.chemdelta *= 0

def update_preserving_boundaries(field):
    boundary = field[[0, -1]].copy()
    try:
        field.update()
    finally:
        field[[0, -1]] = boundary

def refresh_after_chemistry(sim):
    """Refresh dependencies of the gas and dust surface densities after remapping.

    Mass-bin radii, material properties, mixing parameters, fragmentation
    thresholds and external sources are fixed here. Custom chemistry-dependent
    prescriptions for those quantities must extend this update order.
    """
    gas, dust = sim.gas, sim.dust

    # Passive irradiation is unchanged, but viscous heating can respond to
    # the new gas and small-grain abundances. Refresh thermal dependencies
    # only if the temperature actually changes.
    temperature = gas.T.copy()
    gas.T.update()
    if not np.array_equal(gas.T, temperature):
        for field in (gas.cs, gas.Hp, gas.nu, dust.v.rel.brown):
            field.update()

    for field in (gas.rho, gas.n, gas.mfp, gas.P, gas.eta,
                  dust.St, dust.H, dust.rho, dust.eps):
        field.update()

    gas.v.update()
    dust.v.driftmax.update()
    dust.v.rad.update()
    dust.v.rel.update()

    # Preserve the completed transport step's boundary fluxes and sources.
    # Restore each flux before taking its divergence in the next update.
    for field in (gas.Fi, dust.Fi.adv, dust.Fi.diff, dust.Fi.tot,
                  gas.S.hyd, dust.S.hyd, dust.S.coag, gas.S.tot, dust.S.tot):
        update_preserving_boundaries(field)
