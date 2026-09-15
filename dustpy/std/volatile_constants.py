import numpy as np
from astropy import constants as ac

# In this file, we define the properties of volatile species that are used in the chemistry calculations. 
# Each volatile species is represented as a dictionary containing its physical properties, such as molecular radius, mass, and vapor pressure function.

def vapor_pressure_H2O(T):
    """Saturation vapor pressure of H2O."""
    return np.exp(28.9074 - 6143.7 / T)

def vapor_pressure_CO(T):
    """Saturation vapor pressure of CO."""
    return np.exp(27.37 - 1030 / T)

def vapor_pressure_CH4(T):
    """Saturation vapor pressure of CH4."""
    return np.exp(23.49 - 1110 / T)

VOLATILE_PROPERTIES = {
    "H2O": {
        "radius": 1.38e-8,                         # cm
        "mass": 18.01528 * ac.u.cgs.value,         # g
        "vapor_pressure": vapor_pressure_H2O,
    },

    "CO": {
        "radius": 1.88e-8,                         # cm
        "mass": 28.01 * ac.u.cgs.value,            # g
        "vapor_pressure": vapor_pressure_CO,
    },

    "CH4": {
        "radius": 3.8e-8,                         # cm
        "mass": 16.043 * ac.u.cgs.value,            # g
        "vapor_pressure": vapor_pressure_CH4,
    },
}