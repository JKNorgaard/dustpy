'''Module containing standard functions for ice species.'''

from dataclasses import field

import numpy as np
from numba import njit, prange
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

rhs_time = []


@njit(cache=True, parallel=True)
def rhs_generator(
    stick_ice,
    lf_ind,
    stick_ind_ice,
    m,
    phi,
    Rf,
    Rs,
    Sigma_dust,
    Sigma_ice,
    SigmaFloor,
    dust_number,
    Nr,
    Nm,
    rhs,
):
    """
    Compute the RHS for ice species.

    Parameters
    ----------
    stick_ice : array-like, sticking probabilities for ice species
    lf_ind : array-like, indices for the left factor in coagulation
    stick_ind_ice : array-like, indices for sticking in ice coagulation
    m : array-like, mass grid
    phi : array-like, porosity factors
    Rf : array-like, fragmentation radii for ice species
    Rs : array-like, sticking radii for ice species
    Sigma_dust : array-like, dust surface density
    Sigma_ice : array-like, ice surface density
    SigmaFloor : float, floor value for surface density
    dust_number : array-like, number density of dust grains
    Nr : int, number of radial grid points
    Nm : int, number of mass grid points

    Returns
    -------
    rhs2 : array-like, computed RHS for ice species

    Notes
    -----
    The index arrays ``lf_ind`` and ``stick_ind_ice`` use Python indexing;
    ``-1`` in ``stick_ind_ice`` denotes an empty collision-product slot.
    """

    # b is defined as m(i+1) = m(i) * 10**b.
    b = np.log10(m[0] / m[-1]) / (1.0 - Nm)

    # ce from Brauer et al. (2008), equation (A.6).
    ce = int(-1.0 / b * np.log10(1.0 - 10.0 ** (-b))) + 1

    # Fortran's interior range 2..Nr-1 becomes 1..Nr-2 in Python.
    for ir in prange(1, Nr - 1):
        imax = 0
        for mass_bin in range(Sigma_ice.shape[1]):
            if Sigma_ice[ir, mass_bin] > SigmaFloor[ir, mass_bin]:
                imax = mass_bin + 1

        for i in range(1, imax + 1):
            ii = i - 1
            ni = dust_number[ir, ii]
            if i == Nm:
                jmax = 0
            else:
                jmax = min(
                    Nm - 1,
                    int(np.log10(10.0 ** (b * Nm) - 10.0 ** (b * i)) / b),
                )

            for j in range(1, i + 1):
                jj = j - 1
                sticking_rate = Rs[ir, jj, ii]

                if sticking_rate > 0.0:
                    if i <= j + 1 - ce:
                        d_term = -m[jj + 1] * m[ii] / (
                            (m[jj + 1] - m[jj]) * (m[jj] + m[ii])
                        )
                    else:
                        d_term = -1.0

                    if j <= i + 1 - ce:
                        f_term = m[ii] * (
                            m[ii + 1] - (m[ii] + m[jj])
                        ) / (
                            (m[ii + 1] - m[ii]) * (m[ii] + m[jj])
                        )
                    else:
                        f_term = 0.0

                    for l in range(4):
                        k = stick_ind_ice[l, jj, ii]
                        if k < 0:
                            continue
                        rhs[ir, k] += (
                            stick_ice[l, jj, ii]
                            * m[k]
                            * sticking_rate
                            * Sigma_ice[ir, jj]
                            / (m[ii] + m[jj])
                            * ni
                        )

                    if j <= jmax:
                        rhs[ir, jj] += (
                            sticking_rate * Sigma_ice[ir, jj] * d_term * ni
                        )
                        rhs[ir, ii] += (
                            Rs[ir, ii, jj] * Sigma_ice[ir, jj] * f_term * ni
                        )

                fragmentation_rate = Rf[ir, jj, ii]
                if fragmentation_rate > 0.0:
                    ratef = fragmentation_rate * Sigma_ice[ir, jj] * ni
                    rhs[ir, :] += ratef * phi[lf_ind[jj, ii], :]
                    rhs[ir, jj] -= ratef

    return rhs


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
    Nr, Nm = int(sim.grid.Nr), int(sim.grid.Nm)
    coag = sim.dust.coagulation
    Sigma_dust = np.asarray(sim.dust._sigmaOld)
    Sigma_ice = np.asarray(field)
    t2 = time.perf_counter()

    dust_number = np.asarray(sim.dust.Sigma / sim.grid.m[None, :])
    t3 = time.perf_counter()
    #rhs2_f = ice_f.rhs_generator(coag.stick_ice, coag.lf_ind, coag.stick_ind_ice,
    #    sim.grid.m, coag.phi, sim.dust._ice_Rf, sim.dust._ice_Rs,
    #    Sigma_dust, Sigma_ice, sim.dust.SigmaFloor, dust_number, Nr, Nm,
    #)

    rhs2 = np.zeros((Nr, Nm), dtype=np.float64)
    rhs_generator(
        np.asarray(coag.stick_ice),
        np.asarray(coag.lf_ind),
        np.asarray(coag.stick_ind_ice),
        np.asarray(sim.grid.m),
        np.asarray(coag.phi),
        np.asarray(sim.dust._ice_Rf),
        np.asarray(sim.dust._ice_Rs),
        Sigma_dust,
        Sigma_ice,
        np.asarray(sim.dust.SigmaFloor),
        dust_number,
        Nr,
        Nm,
        rhs2,
    )
    t4 = time.perf_counter()
    #print(f"rhs check: {np.abs(rhs2 - rhs2_f).max():.6e} (max abs difference)")
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
    rhs_time.append(t4-t3)
    #print(f"rhs: {t4-t3:.6f} s, total: {np.sum(rhs_time):.6f} s, calls: {timings['calls']}, avg: {np.sum(rhs_time)/timings['calls']:.6f} s")
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
