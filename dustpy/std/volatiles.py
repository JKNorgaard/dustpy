'''Module containing standard functions for setting up volatile species.'''
import numpy as np
import scipy.sparse as sp
import time

from dustpy import std
from dustpy.std import ice_f, sim
from dustpy.utils.boundary import Boundary
from simframe.frame import Group
from dustpy.std.volatile_constants import VOLATILE_PROPERTIES

def chemistry_updater(sim):
    """Exchange phases with variable radius unless explicitly disabled."""

    # Run the sublimation and condensation chemistry for all volatile species
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
    sim.gas.Sigma += delta_vapor.sum(axis=0)


def finalize_volatiles(sim):
    """Transport all species, exchange phases together, then remap once."""
    chemistry_updater(sim)
    remap_updater(sim)


def remap_updater(sim):
    """Function defines the updater function for remapping surface densities back onto the mass grid.
       This function is called at each time step to update the dust surface density based on the change in ice species surface densities.
    
        Parameters
        ----------
        sim : Frame
            Parent simulation frame
        volatiles : list of str
            List of volatile species in the simulation frame
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

    # Remap once to account for the chemistry mass exchange.
    dust_new, ice_new = std.chemistry.remapper(sim)

    # Update the dust surface density and enforce floor value
    sim.dust.Sigma = dust_new
    std.dust.enforce_floor_value(sim)

    # Update the ice surface densities for each volatile species and enforce floor values
    for i, volatile_name in enumerate(sim.volatiles.names):
        getattr(sim.volatiles, volatile_name).Sigmaice = ice_new[i]
        std.ice.enforce_floor_value(getattr(sim.volatiles, volatile_name).Sigmaice)

    # Reset the chemistry delta for the next timestep
    sim.dust.Sigma.chemdelta *= 0


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
        """Evolve the solid component of the volatile."""

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
        """Evolve the gaseous component of the volatile."""

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
    """

    # Create a new group for volatile species in the simulation frame
    sim.addgroup("volatiles", description="Volatile species")

    sim.volatiles.names = tuple(volatiles)
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
