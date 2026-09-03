'''Module containing standard functions for ice species.'''

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

    field[...] = np.where(
        field > field.SigmaFloor,
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
    Rf      = sim.dust.kernel_old * sim.dust.p.frag_old
    Rs      = sim.dust.kernel_old * sim.dust.p.stick_old 
    SigD    = sim.dust._sigmaOld.copy()
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
    """

     # Time step size
    dt = sim.t.prevstepsize

    # Define the first term of the RHS
    rhs1 = field.copy().ravel()

    # Required quantities are defined
    Nr, Nm      = sim.grid.Nr, sim.grid.Nm
    dt          = sim.t.prevstepsize
    m           = sim.grid.m
    A           = sim.dust.coagulation.A
    cStick      = sim.dust.coagulation.stick_ice
    eps         = sim.dust.coagulation.eps
    iRM         = sim.dust.coagulation.rm_ind
    iLF         = sim.dust.coagulation.lf_ind
    iStick      = sim.dust.coagulation.stick_ind_ice
    phi         = sim.dust.coagulation.phi
    Rf          = sim.dust.kernel_old * sim.dust.p.frag_old 
    Rs          = sim.dust.kernel_old * sim.dust.p.stick_old 
    Sigma_dust  = sim.dust._sigmaOld.copy()
    SigmaFloor  = sim.dust.SigmaFloor
    Sigma_ice   = field.copy()
    Xi       = sim.dust.coagulation.Xi_ice

    # M matrix is calculated
    M = ice_f.m_generator(A, cStick, eps, iLF, iRM, iStick, m, phi, Rf, Rs, Sigma_dust, Sigma_ice, SigmaFloor, Xi, Nr, Nm)

    # Calculate second term of RHS
    rhs2 = (np.einsum('rkj,rj->rk', M, sim.dust.Sigma/sim.grid.m[None, :]) * dt).ravel()

    # Assemble total RHS
    # Load RHS boundary conditions
    rhs = field._rhs.copy()

    # Add the two RHS terms to the interior
    rhs[Nm:-Nm] += rhs1[Nm:-Nm] + rhs2[Nm:-Nm]
    return rhs 


def evolve_implicit(sim, field):
    """
    Advance Σ_ice implicitly for one step.
    Solves the matrix equation: ΔΣ_ice * A = RHS for ΔΣ_ice
    Where RHS = Σ_ice^i + (M * Σ_dust^(t+Δt) / m) * Δt, and A = (1 - Δt*J_ice)

    Parameters
    ----------
    sim : Frame
    field : Field, Parent field

    Returns
    -------
    Y1  : Ice surface density Σ_ice at t+dt 
    """

    # Ice surface density we wish to evolve
    Y0 = field.copy()

    # Time step size used in dust routine
    dt = sim.t.prevstepsize

    # Retrieve hydrodynamics and boundary jacobians from dust routine
    J_hyd = sim.dust._J_hyd 
    J_boundary = sim.dust._J_boundary
    #bf = time.perf_counter()
    # Build coagulation jacobian for ice
    J_coag = build_J_coag(sim)
    #af = time.perf_counter()
    #print(f"Ice jacobian time = {af - bf:.3f}")
    # Build total jacobian J_ice = J_hyd + J_coag_ice + J_boundary
    J_ice = (J_hyd + J_coag + J_boundary)
    #bf = time.perf_counter()
    # Enforce boundary conditions on RHS
    rhs_boundary(sim, field)
    #af = time.perf_counter()
    #print(f"Ice jacobian boundary time = {af - bf:.3f}")

    #bf = time.perf_counter()
    # Assemble RHS 
    rhs = assemble_rhs(sim, field)
    #af = time.perf_counter() 
    #print(f"Assembling rhs time = {af - bf:.3f}")

    #bf = time.perf_counter()
    # Identity matrix
    N = J_ice.shape[0]
    I = sp.identity(N, format="csc")

    # Calculate A
    A = I - dt * J_ice

    # LU factorize 
    A_LU = sp.linalg.splu(A,
                        permc_spec="MMD_AT_PLUS_A", 
                        diag_pivot_thresh=0.0,
                        options=dict(SymmetricMode=True)
    )

    # Solve the matrix equation and reshape
    Y1_ravel = A_LU.solve(rhs)
    #af = time.perf_counter()
    #print(f"Solving ice evo time = {af - bf:.3f}")
    Y1 = Y1_ravel.reshape(Y0.shape)

    return Y1

