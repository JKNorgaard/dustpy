import numpy as np
from numba import njit
import astropy.constants as ac

def refractory_carbon_rate(temperature, prefactor=4.e13, activation_temperature=24500.):
    """Organic-carbon decay rate [s^-1]; Gail & Trieloff (2017), Table A.2.
    Temperature is in K.
    """
    temperature = np.asarray(temperature, dtype=float)
    return prefactor*np.exp(-activation_temperature/temperature)

def volatile_quantities(sim, volatile_name, calculate_ML=True):
    """
    Calculate the monolayer ice surface density, equilibrium vapor
    surface density, and thermal velocity for a volatile species.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame.

    volatile_name : str
        Name of the volatile species, e.g. "H2O".
    
    calculate_ML : bool, optional
        Whether to calculate the monolayer surface density. Default is True.
    Returns
    -------
    Sigma_ML : ndarray or None
        Surface density corresponding to one monolayer of the volatile
        on the dust grains. Returns None if calculate_ML is False.

    Sigma_eq_vap : ndarray
        Equilibrium vapor surface density.

    v_therm : ndarray
        Thermal velocity of the volatile molecules.
    """
    # Get volatile species object
    volatile = getattr(sim.volatiles, volatile_name)

    # Species-specific constants
    r_mol = volatile.radius
    m_mol = volatile.mass
    vapor_pressure = volatile.vapor_pressure

    # Simulation quantities
    m = sim.grid.m
    a = sim.dust.a
    T = sim.gas.T
    H_P = sim.gas.Hp

    #################  Monolayer surface density ####################
    # Total monolayer surface density
    if calculate_ML:
        # Cross-sectional area of a molecule
        sigma_mol = np.pi * r_mol**2
        
        # Monolayer surface density for one grain
        ML_per_grain = (4.0 * np.pi * (a + r_mol)**2 / sigma_mol * m_mol)

        # Monolayer surface density for the entire dust population
        Sigma_ML = (sim.dust.Sigma / m) * ML_per_grain
    else:
        Sigma_ML = None

    #################################################################
    ############## Equilibrium vapor surface density ################

    # Calculate vapor pressure distribution
    P_eq = vapor_pressure(T)

    # Calculate equilibrium vapor surface density
    Sigma_eq_vap = np.sqrt(2.0 * np.pi) * H_P * m_mol * P_eq / (ac.k_B.cgs.value * T)

    # Clip numerically problematic values
    Sigma_eq_vap = np.clip(Sigma_eq_vap, 1e-300, None)

    #################################################################
    ###################### Thermal velocity #########################

    # Calculate thermal velocity
    v_therm = np.sqrt(8.0 * ac.k_B.cgs.value * T / (np.pi * m_mol))

    # Return the calculated quantities
    return Sigma_ML, Sigma_eq_vap, v_therm



def solid_chemistry_all(sim, radius_tolerance=None, max_substeps=10000):
    """Calculate and return the changes to volatile and carbon surface densities caused by chemistry.
       If sim.volatiles.chemistry_variable_radius=True, substepping is enabled in radius.
    """

    # Get the current timestep size
    dt = float(sim.t.prevstepsize)

    # Get the names of all volatile species and their corresponding objects
    names = tuple(sim.volatiles.names)
    species = [getattr(sim.volatiles, name) for name in names]
    ice = np.stack([np.asarray(specie.Sigmaice) for specie in species])
    vapor = np.stack([np.asarray(specie.Sigmavap) for specie in species])
    dust = np.asarray(sim.dust.Sigma)

    # Initialize arrays to hold the changes in ice and carbon surface densities
    ice_change = np.zeros_like(ice)
    carbon_change = np.zeros_like(dust)

    # Define radius of mass bins
    radius = np.asarray(sim.dust.a)

    # Mask out bins where dust surface density is below the floor value
    resolved = dust > sim.dust.SigmaFloor

    # Calculate the monolayer surface density, equilibrium vapor surface density, and thermal velocity for each volatile species
    quantities = [volatile_quantities(sim, name) for name in names]
    monolayer = np.stack([q[0] for q in quantities])
    equilibrium = np.stack([q[1] for q in quantities])
    velocity = np.stack([q[2] for q in quantities])

    # Get the molecular radii for each volatile species
    r_mol = np.array([float(specie.radius) for specie in species])

    # Calculate the radius independent monolayer surface density for each species and bin
    mono = monolayer / (radius[None, :, :]+r_mol[:, None, None])**2

    # Calculate the midplane number density and collision rates for each species and bin
    n_midplane = np.asarray(sim.dust.rho)/np.asarray(sim.grid.m)
    collision = np.pi*radius[None, :, :]**2*n_midplane[None, :, :]*velocity[:, :, None]

    # Get the refractory carbon species object, if it exists
    carbon_species = getattr(sim, 'refractory_carbon', None)

    # Initialize arrays for carbon surface density and its decay rate
    carbon = np.zeros_like(dust)
    rate = np.zeros(dust.shape[0])

    methane_index, methane_ratio = None, 1.
    if carbon_species is not None:
        carbon = np.asarray(carbon_species.Sigma)
        rate = refractory_carbon_rate(sim.gas.T, float(carbon_species.prefactor),
                                      float(carbon_species.activation_temperature))
        methane_index = names.index('CH4')
        methane_ratio = float(sim.volatiles.CH4.mass/carbon_species.mass)

    # Determine whether to use variable radius for chemistry calculations and set the radius tolerance
    variable_radius = getattr(sim.volatiles, 'chemistry_variable_radius', True)
    radius_tolerance = getattr(sim.volatiles, 'chemistry_radius_tolerance', 0.01)

    # Loop over radial bins, and perform chemistry 
    for r in range(1, dust.shape[0]-1):
        use = resolved[r]
        if not np.any(use):
            continue

        # Create dictionary with arguments for carbon chemistry calculations
        carbon_args = dict(carbon=carbon[r, use], carbon_rate=rate[r])

        # Calculate change in ice and carbon surface densitites with or without adaptive radius substepping
        if variable_radius:
            result = _adaptive_radius(
                dt, ice[:, r, use], vapor[:, r], equilibrium[:, r],
                dust[r, use], radius[r, use], collision[:, r, use],
                mono[:, r, use], r_mol, radius_tolerance=radius_tolerance,
                max_substeps=max_substeps, **carbon_args)
        else:
            result = _constant_radius(
                dt, ice[:, r, use], vapor[:, r], equilibrium[:, r],
                collision[:, r, use], monolayer[:, r, use], **carbon_args)
            
        ice_change[:, r, use], carbon_change[r, use] = result

    # Calculate the change in vapor surface density based on chemistry
    vapor_change = -ice_change.sum(axis=-1)

    # Include the final geometric correction before committing any field updates.
    released = monolayer_shrinkage(
        sim, ice=ice+ice_change, vapor=vapor+vapor_change,
        solid_delta=ice_change.sum(axis=0)+carbon_change, quantities=quantities)
    ice_change -= released
    vapor_change += released.sum(axis=-1)

    # Deposit pyrolysis methane after ice chemistry, including its final
    # monolayer correction. Same-step phase coupling is neglected.
    if methane_index is not None:
        vapor_change[methane_index] -= methane_ratio*carbon_change.sum(axis=-1)
    return ice_change, vapor_change, carbon_change


def _constant_radius(dt, ice, vapor, equilibrium, collision, monolayer,
                     carbon=None, carbon_rate=0.):
    """Analytic phase exchange at fixed grain radius, with depletion substepping.
    """

    if carbon is None:
        carbon = np.zeros(ice.shape[1])
    ice_change = _ice_exchange(dt, ice, vapor, equilibrium, collision, monolayer)
    carbon_change = carbon*np.expm1(-carbon_rate*dt)
    return ice_change, carbon_change


@njit(cache=True, error_model="numpy")
def _ice_exchange(dt, ice, vapor, equilibrium, collision, monolayer):
    """Analytic phase exchange at fixed grain radius, with depletion substepping.

    Arrays have shape (species, bins), except vapor/equilibrium (species,).
    Compiled on first use; subsequent calls reuse the cached Numba kernel.
    """


    change = np.zeros_like(ice)
    nbins = ice.shape[1]
    # Reuse scratch storage across species and substeps. Keep IEEE arithmetic
    # (no fastmath): depletion decisions depend on small residual capacities.
    active = np.empty(nbins, dtype=np.bool_)
    weights = np.empty(nbins, dtype=np.float64)
    capacity = np.empty(nbins, dtype=np.float64)
    required = np.empty(nbins, dtype=np.float64)

    # Loop over volatile species
    for x in range(len(vapor)):

        # Calculate the the deficit of the current species
        deficit = float(vapor[x] - equilibrium[x])
        # If the deficit is zero, skip to the next species
        if deficit == 0.0:
            continue

        # Mask bins that are condensing
        condensing = deficit > 0.0

        # Sublimating or condensing bins are called active
        # They must have positive collision rates and either be condensing or have ice above the monolayer
        # To be considered active
        for j in range(nbins):
            active[j] = collision[x, j] > 0.0 and (condensing or ice[x, j] > monolayer[x, j])

        # Define the remaining time for the current timestep
        remaining = float(dt)

        # Loop over substeps with a maximum of nbins+1 iterations
        for _ in range(nbins + 1):

            # 
            total = 0.0
            changed = 0.0
            for j in range(nbins):
                if active[j]:
                    total += collision[x, j]
                changed += change[x, j]
            if total == 0.0 or remaining <= 0.0:
                break
            available = max(0.0, np.sign(deficit) * (deficit - changed))
            if available == 0.0:
                break

            first_required = np.inf
            for j in range(nbins):
                weights[j] = collision[x, j] / total if active[j] else 0.0
                if not condensing:
                    capacity[j] = max(ice[x, j] + change[x, j] - monolayer[x, j], 0.0)
                    required[j] = capacity[j] / weights[j] if active[j] else np.inf
                    first_required = min(first_required, required[j])

            event_time = np.inf
            if not condensing:
                fraction = first_required / available
                if fraction < 1.0:
                    event_time = -np.log1p(-fraction) / total

            step = min(remaining, event_time)
            transferred = available * (-np.expm1(-total * step))
            for j in range(nbins):
                delta = weights[j] * transferred
                if not condensing:
                    delta = -min(delta, capacity[j])
                change[x, j] += delta

            finished = step == remaining
            remaining = max(0.0, remaining - step)
            if finished:
                break
            if event_time <= step:
                # Remove all bins tied at the first depletion threshold.
                for j in range(nbins):
                    active[j] = active[j] and required[j] > first_required
            else:
                break
    return change

def _adaptive_radius(dt, ice, vapor, equilibrium, dust, radius, collision,
                   monolayer, molecular_radius, radius_tolerance=0.01,
                   max_substeps=10000, carbon=None, carbon_rate=0.):
    """ Calculate the change in ice and carbon surface densitites caused by chemistry, with adaptive substepping in radius. 
        Rates of phase change and pyrolysis are calculated based on the current grain radius, which is updated after each substep. 
        The substepping threshold is set by radius_tolerance, which is the maximum allowed fractional change in grain radius per substep.
    """

    # Initialize bookkeeping array for change in ice surface density
    ice_change = np.zeros_like(ice)
    carbon_change = np.zeros_like(dust)
    carbon = np.zeros_like(dust) if carbon is None else carbon

    # Initialize the remaining time for the current timestep
    remaining = float(dt)

    # Convert fractional radius tolerance to mass tolerance 
    mass_tolerance = 1.0-(1.0-radius_tolerance)**3

    # Loop over substeps with a maximum of max_substeps iterations
    for _ in range(max_substeps):

        # If there is no remaining time, return the changes
        if remaining <= 0.0:
            return ice_change, carbon_change

        # Calculate the total mass of dust and volatile ices in each bin
        mass = dust + ice_change.sum(axis=0) + carbon_change

        # Calculate the new grain radius 
        a = radius*np.cbrt(mass/dust)

        # Calculate the collision rates and monolayer surface densities for new grain radius
        rates = collision*(a/radius)[None, :]**2
        ml = monolayer*(a[None, :]+molecular_radius[:, None])**2

        # Update ice and vapor for change in surface density
        current_ice = ice+ice_change
        current_vapor = vapor-ice_change.sum(axis=1)
        current_carbon = np.maximum(carbon+carbon_change, 0.)


        h = remaining
        if carbon_rate > 0:
            # An exact bound avoids repeatedly halving long trials when
            # destruction is very fast. Exhaustible small reservoirs do not
            # restrict the timestep once their entire mass fits the budget.
            fraction = np.divide(mass_tolerance*mass, current_carbon,
                out=np.full_like(mass, np.inf), where=current_carbon > 0)
            limiting = fraction < 1
            if np.any(limiting):
                h = min(h, float(np.min(-np.log1p(-fraction[limiting])/carbon_rate)))
        for _ in range(100):
            # Ice exchange and carbon loss are independent at fixed geometry.
            # Carbon still affects the radii and the accepted substep size.
            trial, trial_carbon = _constant_radius(
                h, current_ice, current_vapor, equilibrium, rates, ml,
                carbon=current_carbon, carbon_rate=carbon_rate)

            # Calculate the gross change in ice surface density for the current substep
            gross = np.abs(trial).sum(axis=0)-trial_carbon

            # If the gross change is within the mass tolerance, break the loop and accept the trial change
            if np.all(gross <= mass_tolerance*mass):
                break

            # If the gross change exceeds the mass tolerance, find the maximum allowed timestep for the current substep
            # Calculate the rate of phase change for each species and bin
            flux = rates*(current_vapor-equilibrium)[:, None]

            # Prevent a bare species from dictating the timestep
            flux = np.where((flux < 0.0) & (current_ice <= ml), 0.0, flux)

            # A nearly bare species can have an enormous instantaneous
            # sublimation rate but almost no ice left to transfer. Its
            # depletion is handled analytically; it must not dictate the
            # timestep for a slower species with a substantial mantle.
            significant = np.abs(trial) > mass_tolerance*mass/(4*len(vapor))

            # Only use bins that are significant for determining the timestep
            # and set flux to zero for bins with insignificant change
            flux = np.where(significant, flux, 0.0)

            # Calculate the gross rate of significant bins
            gross_rate = np.abs(flux).sum(axis=0)
            gross_rate += carbon_rate*current_carbon

            # Calculate a new timestep based on the mass tolerance and gross rate, ensuring it is positive and finite
            bound = np.divide(mass_tolerance*mass, gross_rate,
                              out=np.full_like(mass, np.inf), where=gross_rate > 0)

            # The new timestep is min of either half of the current timestep, ensuring that the new timestep becomes smaller,
            # or the calculated timestep candidate, 
            h = min(0.5*h, bound.min())
            if h <= 0.0:
                raise RuntimeError('Adaptive chemistry timestep underflow.')
        else:
            raise RuntimeError('Adaptive chemistry could not bound the radius change.')

        # Update the change in ice surface density and remaining time for the current timestep
        ice_change += trial
        carbon_change += trial_carbon
        # A tiny substep can change the ice or carbon even when subtracting
        # it does not change remaining at floating-point precision.
        # Allow that chemical progress; max_substeps bounds the iteration.
        remaining = max(0.0, remaining-h)

        # If there is no remaining time, return the change in ice surface density
        if remaining == 0.0:
            return ice_change, carbon_change
    raise RuntimeError('Adaptive chemistry exceeded max_substeps.')


def monolayer_shrinkage(sim, ice=None, vapor=None, solid_delta=None, quantities=None):
    """If dust bins have become bare, they have also shrunk in radius.
       This function calculates the amount of ice that must be released to
       maintain the same monolayer coverage on the new dust radius. 
    """

    # Define the ratio of the new dust surface density to the old dust surface density based on the chemistry delta
    dust = np.asarray(sim.dust.Sigma)
    if solid_delta is None:
        solid_delta = sim.dust.Sigma.chemdelta
    ratio = 1.0 + np.divide(solid_delta, dust, out=np.zeros_like(dust), where=dust > 0)

    # Preserve inner and outer boundary, this is set by boundary conditions and not chemistry.
    ratio[0] = 1.0
    ratio[-1] = 1.0

    # Calculate the new dust radius based on the ratio
    radius = sim.dust.a * np.cbrt(ratio)

    # Calculate the midplane number density
    number_density = dust / (np.sqrt(2*np.pi) * sim.dust.H * sim.grid.m)

    # The sublimation process can take a maximum of 1% of the previous timestep
    # ensuring that the proposed release is fast enough to be considered instantaneous
    budget = 0.01 * max(float(sim.t.prevstepsize), 0.0)

    # Define bookkeeping array
    released = []

    # Loop over volatile species
    for x, name in enumerate(sim.volatiles.names):
        volatile = getattr(sim.volatiles, name)
        current_ice = volatile.Sigmaice if ice is None else ice[x]
        current_vapor = volatile.Sigmavap if vapor is None else vapor[x]
        ml, equilibrium, thermal_speed = (
            volatile_quantities(sim, name) if quantities is None else quantities[x])

        # Calculate the new monolayer surface density based on the new dust radius
        ml_new = ml * ((radius + volatile.radius) / (sim.dust.a + volatile.radius))**2

        # Only consider bins at the monolayer limit
        candidate = np.where(current_ice <= ml*(1.0 + 1e-10), np.maximum(current_ice - ml_new, 0.0), 0.0)

        # Inner and outer boundaries are not considered
        candidate[0] = 0.0
        candidate[-1] = 0.0

        # Calculate the deficit of the current species, including the proposed release
        deficit = np.maximum(equilibrium - current_vapor - candidate.sum(axis=-1), 0.0)

        # Calculate the rate of sublimation based on the new dust radius and deficit
        rate = (np.pi * radius**2 * thermal_speed[:, None] * number_density * deficit[:, None])

        # Determine the amount of ice to release based on the budget, rate, and candidate values
        released.append(np.where((budget > 0) & (rate > 0) & (candidate <= budget*rate), candidate, 0.0))

    # Return the released ice surface densities for all volatile species, stacked into an array
    return np.stack(released)


@njit(cache=True, error_model="numpy")
def _remap_arrays(dust, tracers, solid_delta_total, B):
    """Remap dust and tracers back on the discrete logarithmic mass grid 
       based on the change in solid material caused by chemistry. 
    """

    # Initialize new arrays for dust and tracers with the same shape as the input arrays
    dust_new = np.zeros_like(dust)
    tracers_new = np.zeros_like(tracers)

    # Preserve inner and outer boundaries
    dust_new[0] = dust[0]
    dust_new[-1] = dust[-1]
    tracers_new[:, 0, :] = tracers[:, 0, :]
    tracers_new[:, -1, :] = tracers[:, -1, :]

    # Precompute log(B) and the denominator for weight calculations
    log_B = np.log(B)
    denom = 1.0 / (B - 1.0)

    # Get the number of radial bins and mass bins
    nr, nm = dust.shape

    # Loop over radial bins (excluding boundaries) and mass bins to remap dust and tracers
    for r in range(1, nr-1):
        for j in range(nm):
            ratio = (dust[r, j] + solid_delta_total[r, j]) / dust[r, j]
            mass = dust[r, j] * ratio
            displacement = np.log(ratio) / log_B
            offset = int(np.floor(displacement))
            fraction = displacement - offset
            power = B ** (1.0 - fraction)
            weight_left = (power - 1.0) * denom
            weight_right = (B - power) * denom
            left = min(max(j + offset, 0), nm - 1)
            right = min(max(j + offset + 1, 0), nm - 1)
            dust_new[r, left] += mass * weight_left
            dust_new[r, right] += mass * weight_right
            for x in range(tracers.shape[0]):
                tracers_new[x, r, left] += tracers[x, r, j] * weight_left
                tracers_new[x, r, right] += tracers[x, r, j] * weight_right
    return dust_new, tracers_new
