import dustpy.constants as c
from dustpy.std import gas_f
from dustpy.utils.boundary import Boundary
from simframe.frame import Group

import numpy as np
import scipy.sparse as sp

from simframe.integration import Scheme
import scipy.sparse.linalg as splinalg

def initialize_vapor(sim, field):
    """Function initializes and enforces vapor specie boundary conditions

    Parameters
    ----------
    sim : Frame, Parent simulation frame
    field : Field, Parent field
    """
    
    #The boundary conditions group is created
    field.boundary = Group(sim, description="Boundary conditions")

    # Floor values and boundary conditions are set and enforced
    field.SigmaFloor = sim.gas.SigmaFloor.copy()

    set_initial_boundary(sim, field)
    boundary(field)
    enforce_floor_value(field)

def set_initial_boundary(sim, field):
    field.boundary.inner = Boundary(
                    sim.grid.r, sim.grid.ri, field)
    field.boundary.inner.setcondition("const_grad")

    field.boundary.outer = Boundary(
                    sim.grid.r[::-1], sim.grid.ri[::-1], field[::-1])
    field.boundary.outer.setcondition("val", field.SigmaFloor[-1]/100)

def boundary(field):
    """Function sets the boundary condition of dust surface density.
    Not implemented, yet.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    field.boundary.inner.setboundary()
    field.boundary.outer.setboundary()

def enforce_floor_value(field):
    """Function enforces floor value onto dust surface density.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""

    field[...] = np.where(
        field > field.SigmaFloor,
        field,
        0.1*field.SigmaFloor)
    
def finalize_implicit(field):
    """Function finalizes the implicit step by enforcing boundary conditions
    and floor values.

    Parameters
    ----------
    field : Field
        Parent field"""
    boundary(field)
    enforce_floor_value(field)

def evolve_implicit(sim, field):
    """Function evolves vapor surface density implicitly for one step.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    field : Field
        Parent field"""
    
    Y0 = np.copy(field)
    
    boundary = field.boundary
    
    Sext = sim.gas.S.ext

    jac = sim.gas._jac
    dt = sim.t.prevstepsize

    rhs = Y0

    # Setting boundary values in jac and rhs
    # Inner boundary
    if boundary.inner is not None:
        # Given value
        if boundary.inner.condition == "val":
            rhs[0] = boundary.inner.value
        # Constant value
        elif boundary.inner.condition == "const_val":
            rhs[0] = 0.
        # Given gradient
        elif boundary.inner.condition == "grad":
            rhs[0] = - boundary.inner._ri[1]/boundary.inner._r[0] * \
                (boundary.inner._r[1]-boundary.inner._r[0]) * \
                boundary.inner.value
        # Constant gradient
        elif boundary.inner.condition == "const_grad":
            rhs[0] = 0.
        # Given power law
        elif boundary.inner.condition == "pow":
            p = boundary.inner.value
            rhs[0] = Y0[1] * (boundary.inner._r[0]/boundary.inner._r[1])**p
        # Constant power law
        elif boundary.inner.condition == "const_pow":
            rhs[0] = 0.

    # Outer boundary
    if boundary.outer is not None:
        # Given value
        if boundary.outer.condition == "val":
            rhs[-1] = boundary.outer.value
        # Constant value
        elif boundary.outer.condition == "const_val":
            rhs[-1] = 0.
        # Given gradient
        elif boundary.outer.condition == "grad":
            rhs[-1] = boundary.outer._ri[1]/boundary.outer._r[0] * \
                (boundary.outer._r[0]-boundary.outer._r[1]) * \
                boundary.outer.value
        # Constant gradient
        elif boundary.outer.condition == "const_grad":
            rhs[-1] = 0.
        # Given power law
        elif boundary.outer.condition == "pow":
            p = boundary.outer.value
            rhs[-1] = Y0[-2] * (boundary.outer._r[-0]/boundary.outer._r[1])**p
        # Constant power law
        elif boundary.outer.condition == "const_pow":
            rhs[-1] = 0.

    # Add external source terms to right-hand side
    rhs[:] = gas_f.modified_rhs(
        dt,
        rhs,
        Sext
    )
    
    # Solve for field variables at next timestep
    A_LU = sp.linalg.splu(jac,
                          permc_spec="MMD_AT_PLUS_A",
                          diag_pivot_thresh=0.0,
                          options=dict(SymmetricMode=True))
    
    Y1 = A_LU.solve(rhs)
    
    return Y1