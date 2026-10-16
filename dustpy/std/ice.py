'''Module containing standard functions for ice species.'''

from dataclasses import field

import numpy as np
import scipy.sparse as sp
import time

from dustpy.std import ice_f, sim
from dustpy.utils.boundary import Boundary
from simframe.frame import Group

def initialize_ice(sim, field):
    """Function initializes volatile ice specie
    and enforces boundary conditions

    Parameters
    ----------
    sim : Frame, Parent simulation frame
    field : Field, Parent ice field
    """
    
    #The boundary conditions group is created
    field.boundary = Group(sim, description="Boundary conditions")

    # Floor value is dust floor scaled by the summed volatile ice fraction (Eq. 23)
    mass_frac = field.sum(-1).sum(-1) / sim.dust.Sigma.copy().sum(-1).sum(-1)
    field.SigmaFloor = mass_frac * sim.dust.SigmaFloor.copy() 

    # Boundary conditions are set and enforced    
    set_initial_boundary(sim, field) # Boundary conditions are set
    boundary(field)                  # Boundary conditions are enforced
    enforce_floor_value(field)       # Floor values are enforced


def set_initial_boundary(sim, field):
    """Function sets the initial boundary condition of ice specie surface density.
    Uses the same conditions as dust.
    Parameters
    ----------
    sim : Frame, Parent simulation frame
    field : Field, Parent field
    """
    #Inner boundary
    field.boundary.inner = Boundary(
        sim.grid.r,
        sim.grid.ri,
        field,
        condition="const_grad"
    )

    #Outer boundary
    field.boundary.outer = Boundary(
        sim.grid.r[::-1],
        sim.grid.ri[::-1],
        field[::-1],
        condition="val",
        value=0.1*field.SigmaFloor[-1] 
    )


def boundary(field):
    """Function sets the boundary condition for ice surface density

    Parameters
    ----------
    field : Field, Parent field
    """

    field.boundary.inner.setboundary()
    field.boundary.outer.setboundary()


def enforce_floor_value(field):
    """
    Function enforces floor value onto ice surface density

    Parameters
    ----------
    field : Field, Parent field
    """

    # An ice tracer cannot retain a resolved population after its carrier
    # dust has been reset to the numerical floor. Otherwise those remnants
    # accumulate independently and can exceed the grain mass when the bin
    # becomes populated again. Apply the existing ice floor to the same
    # absent populations; do not clip inventories in populated dust bins.
    sim = field._owner
    populated = sim.dust.Sigma > sim.dust.SigmaFloor
    field[...] = np.where(
        (field > field.SigmaFloor) & populated,
        field,
        0.1*field.SigmaFloor)


def finalize_implicit(field):
    """
    Function finalizes implicit integration step
    By enforcing boundary conditions and floor values

    Parameters
    ----------
    field : Field, Parent field
    """
    boundary(field)
    enforce_floor_value(field)


def build_J_coag(sim):
    """
    Builds the coagulation part of the ice jacobian for interior radii 0 < r < Nm-1

    Parameters
    ----------
    sim : Frame, Parent simulation frame

    Returns
    -------
    J_coag : Coagulation part of ice jacobian
    """

    # Required quantities are defined
    A       = sim.dust.coagulation.A
    cstick  = sim.dust.coagulation.stick_ice
    eps     = sim.dust.coagulation.eps
    ilf     = sim.dust.coagulation.lf_ind
    istick  = sim.dust.coagulation.stick_ind_ice
    irm     = sim.dust.coagulation.rm_ind
    m       = sim.grid.m
    phi     = sim.dust.coagulation.phi
    Rf = sim.dust._ice_Rf
    Rs = sim.dust._ice_Rs
    SigD = np.asarray(sim.dust._sigmaOld)
    SigDfl  = sim.dust.SigmaFloor
    Xi      = sim.dust.coagulation.Xi_ice
    Nr      = sim.grid.Nr
    Nm      = sim.grid.Nm
    Ntot    = Nr * Nm

    # Coagulation jacobian is calculated and constructed
    dat, row, col = ice_f.jacobian_coagulation_generator(
        A, cstick, eps, ilf, irm, istick, m, phi, Rf, Rs, SigD, SigDfl, Xi, Nr, Nm
    )
    J_coag = sp.csc_matrix((dat, (row, col)), shape=(Ntot, Ntot))

    return J_coag


def rhs_boundary(sim, field):
    """
    Initializes and enforces boundary conditions on field._rhs

    Parameters
    ----------
    sim : Frame, Parent simulation frame
    field : Field, Parent field
    """

    # Required quantities are defined
    Nm = int(sim.grid.Nm)
    r = sim.grid.r
    ri = sim.grid.ri

    # Initialize field._rhs
    field._rhs = np.zeros_like(field).ravel()

    # Filling data vector depending on boundary condition
    if field.boundary.inner is not None:
        # Given value
        if field.boundary.inner.condition == "val":
            field._rhs[:Nm] = field.boundary.inner.value
        # Constant value
        elif field.boundary.inner.condition == "const_val":
            field._rhs[:Nm] = 0.
        # Given gradient
        elif field.boundary.inner.condition == "grad":
            field._rhs[:Nm] = - ri[1]/r[0] * \
                (r[1]-r[0])*field.boundary.inner.value
        # Constant gradient
        elif field.boundary.inner.condition == "const_grad":
            field._rhs[:Nm] = 0.
        # Given power law
        elif field.boundary.inner.condition == "pow":
            p = field.boundary.inner.value
            field._rhs[:Nm] = field[1] * (r[0]/r[1])**p
        # Constant power law
        elif field.boundary.inner.condition == "const_pow":
            field._rhs[:Nm] = 0.


    # Filling data vector depending on boundary condition
    if field.boundary.outer is not None:
        # Given value
        if field.boundary.outer.condition == "val":
            field._rhs[-Nm:] = field.boundary.outer.value
        # Constant value
        elif field.boundary.outer.condition == "const_val":
            field._rhs[-Nm:] = 0.
        # Given gradient
        elif field.boundary.outer.condition == "grad":
            field._rhs[-Nm:] = ri[-2]/r[-1] * \
                (r[-1]-r[-2])*field.boundary.outer.value
        # Constant gradient
        elif field.boundary.outer.condition == "const_grad":
            field._rhs[-Nm:] = 0.
        # Given power law
        elif field.boundary.outer.condition == "pow":
            p = field.boundary.outer.value
            field._rhs[-Nm:] = field[-2] * (r[-1]/r[-2])**p
        # Constant power law
        elif field.boundary.outer.condition == "const_pow":
            field._rhs[-Nm:] = 0.


# Accumulated wall-clock seconds across all assemble_rhs calls and ice species.
# Clear this dictionary before each benchmark to reset the counters.
assemble_rhs_timings = {}


def assemble_rhs(sim, field):
    """
    Assembles the RHS = Σ_ice^i + (M * Σ_dust^(t+Δt) / m) * Δt for ice specie

    Parameters
    ----------
    sim : Frame
    field : Field, Parent field

    Returns
    -------
    rhs : Right hand side of the matrix equation for ice specie

    Notes
    -----
    Wall-clock timings accumulate in ``assemble_rhs_timings``. The
    ``calls`` entry counts completed calls; all other entries are seconds.
    The total excludes the counter bookkeeping and includes timer overhead.
    """

    t0 = time.perf_counter()

    # Time step size
    dt = sim.t.prevstepsize

    # Define the first term of the RHS
    rhs1 = field.copy().ravel()
    t1 = time.perf_counter()

    # Old dust sets the active collision bins; updated dust weights the RHS.
    Nr, Nm = sim.grid.Nr, sim.grid.Nm
    coag = sim.dust.coagulation
    Sigma_dust = np.asarray(sim.dust._sigmaOld)
    Sigma_ice = np.asarray(field)
    t2 = time.perf_counter()

    dust_number = sim.dust.Sigma / sim.grid.m[None, :]
    t3 = time.perf_counter()
    rhs2 = ice_f.rhs_generator(
        coag.stick_ice, coag.lf_ind, coag.stick_ind_ice,
        sim.grid.m, coag.phi, sim.dust._ice_Rf, sim.dust._ice_Rs,
        Sigma_dust, Sigma_ice, sim.dust.SigmaFloor, dust_number, Nr, Nm,
    )
    t4 = time.perf_counter()
    rhs2 = (rhs2 * dt).ravel()
    t5 = time.perf_counter()

    # Assemble total RHS
    # Load RHS boundary conditions
    rhs = field._rhs.copy()

    # Add the two RHS terms to the interior
    rhs[Nm:-Nm] += rhs1[Nm:-Nm] + rhs2[Nm:-Nm]
    t6 = time.perf_counter()

    timings = assemble_rhs_timings
    timings['calls'] = timings.get('calls', 0) + 1
    for name, elapsed in (
        ('rhs1_copy', t1 - t0),
        ('prepare_inputs', t2 - t1),
        ('dust_number', t3 - t2),
        ('rhs_generator', t4 - t3),
        ('scale_and_flatten', t5 - t4),
        ('assemble_total', t6 - t5),
        ('total', t6 - t0),
    ):
        timings[name] = timings.get(name, 0.0) + elapsed
    return rhs 

def _get_ice_solver(sim):
    """Build once and reuse across solid tracers in this transport step."""
    solver = getattr(sim.dust, "_ice_lu", None)
    if solver is None:
        dt = float(sim.t.prevstepsize)

        J_coag = build_J_coag(sim)
        J_ice = (sim.dust._J_hyd + J_coag + sim.dust._J_boundary)

        A = sp.identity(J_ice.shape[0], format="csc") - dt * J_ice

        solver = sp.linalg.splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
        
        sim.dust._ice_lu = solver

    return solver

def evolve_implicit(sim, field):
    """Advance one solid tracer using the shared transport matrix."""
    solver = _get_ice_solver(sim)

    # These remain specific to each tracer.
    rhs_boundary(sim, field)
    rhs = assemble_rhs(sim, field)

    return solver.solve(np.asarray(rhs)).reshape(field.shape)
