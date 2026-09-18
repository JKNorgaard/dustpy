import numpy as np
import astropy.constants as ac
from dustpy import constants as c  # unused here, kept as in your original

def volatile_quantities(sim, volatile_name):
    """
    Calculate the monolayer ice surface density, equilibrium vapor
    surface density, and thermal velocity for a volatile species.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame.

    volatile_name : str
        Name of the volatile species, e.g. "H2O".

    Returns
    -------
    Sigma_ML : ndarray
        Surface density corresponding to one monolayer of the volatile
        on the dust grains.

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
    # Cross-sectional area of a molecule
    sigma_mol = np.pi * r_mol**2

    # Monolayer surface density for one grain
    ML_per_grain = (4.0 * np.pi * (a + r_mol)**2 / sigma_mol * m_mol)

    # Total monolayer surface density
    Sigma_ML = (sim.dust.Sigma / m) * ML_per_grain

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

def _constant_radius(dt, ice, vapor, equilibrium, collision, monolayer):
    """Analytic phase exchange for one radial bin at fixed grain radius, with depletion substepping.

    Arrays have shape (species, bins), except vapor/equilibrium (species,).
    """

    # Initialize bookkeeping array for change in ice surface density
    change = np.zeros_like(ice)

    # Loop over each species 
    for x in range(len(vapor)):

        # Calculate the deficit of the current species
        deficit = float(vapor[x]-equilibrium[x])
        if deficit == 0.0:
            continue

        # Determine if the species is condensing or sublimating
        condensing = deficit > 0.0

        # Only consider bins that have positive collision rates
        active = collision[x] > 0.0

        # For sublimation, only consider non-bare bins
        if not condensing:
            active &= ice[x] > monolayer[x]

        # Initialize the remaining time for the current timestep
        remaining = float(dt)

        # Loop over substeps with a maximum of ice.shape[1]+1 iterations to ensure convergence
        for _ in range(ice.shape[1]+1):

            # Calculate the summed collision rates for the active bins
            rates = np.where(active, collision[x], 0.0)
            total = rates.sum()

            # If there are no active bins or no remaining time, break the loop
            if total == 0.0 or remaining <= 0.0:
                break


            weights = rates/total

            # Calculate the mass available for sublimation or condensation
            available = max(0.0, np.sign(deficit)*(deficit-change[x].sum()))
            if available == 0.0:
                break

            # Calculate the time until the next event (a bin becomes bare) for sublimation
            event_time = np.inf
            if not condensing:
                # Calculate the remaining ice capacity for sublimation
                capacity = np.maximum(ice[x]+change[x]-monolayer[x], 0.0)

                # Calculate the total sublimation across all bins required for making each bin bare
                required = np.divide(capacity, weights, out=np.full_like(weights, np.inf), where=active)

                # The lowest required sublimation determines the bin which will become bare first
                first = int(np.argmin(required))

                # Calculate the fraction of the vapor deficit that is used for sublimation in this substep
                fraction = required[first]/available

                # If the fraction is less than 1, a bin becomes bare and the time until that happens is calculated
                if fraction < 1.0:
                    event_time = -np.log1p(-fraction)/total

            # If the remaining time is less than the event time, the full remaining interval is integrated
            step = min(remaining, event_time)

            # Calculate the transferred mass for this substep
            with np.errstate(over='ignore'):
                transferred = available*(-np.expm1(-total*step))

            # The transferred mass is distributed among the mass bins
            delta = weights*transferred

            # For sublimation, ensure that the change in ice surface density does not exceed the available ice
            if not condensing:
                delta = -np.minimum(delta, capacity)

            # Update the change in ice surface density for the current species
            change[x] += delta

            # Check if the substep is the last one
            finished = step == remaining

            # Update the remaining time for the current timestep
            remaining = max(0.0, remaining-step)
            if finished:
                # The full remaining interval has been integrated.
                break

            # If a bin becomes bare, remove it from the active set for the next substep
            if event_time <= step:
                active[first] = False
            else:
                break
    return change

def _adaptive_radius(dt, ice, vapor, equilibrium, dust, radius, collision,
                   monolayer, molecular_radius, radius_tolerance=0.01,
                   max_substeps=10000):
    """ Adaptive phase exchange for one radial bin with variable grain radius, with depletion substepping.
    
    Arrays have shape (species, bins), except vapor/equilibrium (species,).
    """

    # Initialize bookkeeping array for change in ice surface density
    change = np.zeros_like(ice)

    # Initialize the remaining time for the current timestep
    remaining = float(dt)

    # Convert fractional radius tolerance to mass tolerance 
    mass_tolerance = 1.0-(1.0-radius_tolerance)**3

    # Loop over substeps with a maximum of max_substeps iterations to ensure convergence
    for _ in range(max_substeps):

        # If there is no remaining time or no ice to transfer, return the change in ice surface density
        if remaining <= 0.0 or ice.size == 0:
            return change

        # Calculate the total mass of dust and volatile ices in each bin
        mass = dust+change.sum(axis=0)
        if np.any(mass <= 0.0):
            raise RuntimeError('Chemistry exhausted the grain mass.')

        # Calculate the new grain radius 
        a = radius*np.cbrt(mass/dust)

        # Calculate the collision rates and monolayer surface densities for new grain radius
        rates = collision*(a/radius)[None, :]**2
        ml = monolayer*(a[None, :]+molecular_radius[:, None])**2

        # Update ice and vapor for change in surface density
        current_ice = ice+change
        current_vapor = vapor-change.sum(axis=1)


        h = remaining
        for _ in range(100):
            # Calculate the trial change in ice surface density for the current substep
            trial = _constant_radius(h, current_ice, current_vapor, equilibrium, rates, ml)

            # Calculate the gross change in ice surface density for the current substep
            gross = np.abs(trial).sum(axis=0)

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

            # Only use bins that are significant for determining the timestep, and set flux to zero for bins with insignificant change
            flux = np.where(significant, flux, 0.0)

            # Calculate the gross rate of significant bins
            gross_rate = np.abs(flux).sum(axis=0)

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
        change += trial
        remaining = max(0.0, remaining-h)

        # If there is no remaining time, return the change in ice surface density
        if remaining == 0.0:
            return change
    raise RuntimeError('Adaptive chemistry exceeded max_substeps.')


def sublimation_condensation_all(sim, radius_tolerance=None, max_substeps=10000):
    """Return phase changes, shapes (Ns,Nr,Nm), (Ns,Nr).

    Call after transport of ALL species and before any chemistry/remapping.
    T, H_d, N, solid density and equilibrium vapor pressures remain fixed.
    Geometry varies unless sim.volatiles.chemistry_variable_radius is
    disabled. In variable-radius mode radius_tolerance bounds excursions within
    each substep, defaulting to chemistry_radius_tolerance (0.01).
    """

    # Get the names of all volatile species in the simulation
    names = tuple(sim.volatiles.names)

    # Stack the ice and vapor surface densities for all species into arrays
    ice = np.stack([np.asarray(getattr(sim.volatiles, n).Sigmaice) for n in names])
    vapor = np.stack([np.asarray(getattr(sim.volatiles, n).Sigmavap) for n in names])

    # Get the timestep size
    dt = float(sim.t.prevstepsize)

    # Check if variable radius chemistry is enabled in the simulation, defaulting to True if not specified
    variable_radius = getattr(sim.volatiles, 'chemistry_variable_radius', True)

    if radius_tolerance is None:
        radius_tolerance = getattr(sim.volatiles, 'chemistry_radius_tolerance', 0.01)

    # Initialize bookkeeping array for change in ice surface density
    change = np.zeros_like(ice)

    # Define simulation quantities for the adaptive chemistry calculations
    dust = np.asarray(sim.dust.Sigma)
    radius = np.asarray(sim.dust.a)
    mass = np.asarray(sim.grid.m)
    n_d = dust / mass
    n_midplane = np.asarray(sim.dust.rho) / mass
    floor = np.asarray(sim.dust.SigmaFloor)

    # Check for invalid conditions in the dust and ice surface densities
    resolved = dust > floor
    invalid = (dust < 0) | np.any(ice < 0, axis=0) | (
        resolved & (ice.sum(axis=0) > dust*(1+1e-10)))
    if np.any(invalid):
        r, k = np.argwhere(invalid)[0]
        raise ValueError(f'Inconsistent total ice at radial bin {r}, mass bin {k}: '
                         f'ice={ice[:,r,k].sum():.17g}, dust={dust[r,k]:.17g}.')
    
    # Calculate the monolayer surface density, equilibrium vapor surface density, and thermal velocity for each volatile species
    quantities = [volatile_quantities(sim, n) for n in names]

    # Stack the calculated quantities into arrays for further calculations
    equilibrium = np.stack([q[1] for q in quantities])
    velocity = np.stack([q[2] for q in quantities])

    # Get the molecular radius and mass for each volatile species
    r_mol = np.array([float(getattr(sim.volatiles, n).radius) for n in names])
    m_mol = np.array([float(getattr(sim.volatiles, n).mass) for n in names])

    # Calculate the monolayer surface density for each volatile species
    mono = (4*m_mol/r_mol**2)[:, None, None] * n_d[None, :, :]

    # Calculate the collision rates for each volatile species
    collision = np.pi*radius[None, :, :]**2*n_midplane[None, :, :]*velocity[:, :, None]

    # Loop over radial bins and calculate the change in ice surface density for each volatile species
    for r in range(dust.shape[0]):

        # Only consider radial bins that have more dust than the floor value
        use = resolved[r]
        if not np.any(use):
            continue

        # Calculate change in ice surface density using adaptive chemistry or constant-radius chemistry based on the variable_radius flag
        try:
            if variable_radius:
                # If variable_radius is True, use the adaptive radius chemistry method
                change[:, r, use] = _adaptive_radius(
                    dt, ice[:, r, use], vapor[:, r], equilibrium[:, r],
                    dust[r, use], radius[r, use], collision[:, r, use],
                    mono[:, r, use], r_mol, radius_tolerance=radius_tolerance,
                    max_substeps=max_substeps)
            else:
                # If variable_radius is False, use the constant-radius chemistry method
                monolayer = mono[:, r, use] * (radius[r, use][None, :] + r_mol[:, None])**2
                change[:, r, use] = _constant_radius(
                    dt, ice[:, r, use], vapor[:, r], equilibrium[:, r],
                    collision[:, r, use], monolayer)
        except RuntimeError as error:
            raise RuntimeError(f'Chemistry at radial bin {r}: {error}') from error

        # Return the change in ice surface density and the negative sum of the change equal to the change in vapor surface density
    return change, -change.sum(axis=-1)


def monolayer_shrinkage(sim):
    """Return ice to release once, before remapping (Nspecies, Nr, Nm).

    Ordinary chemistry has already set dust.Sigma.chemdelta. Use that mass
    change to estimate the new radius. Sublimate ice that is now above the 
    monolayer limit, and return it to the gas.
    This assumes fast sublimation: the proposed release must take <= 1% of
    the previous timestep, using the vapor deficit AFTER the whole proposed
    transfer. Saturated and slow bins are left unchanged.
    """

    # Define the ratio of the new dust surface density to the old dust surface density based on the chemistry delta
    dust = np.asarray(sim.dust.Sigma)
    ratio = 1.0 + np.divide(sim.dust.Sigma.chemdelta, dust, out=np.zeros_like(dust), where=dust > 0)
    ratio[0] = 1.0  # Preserve inner boundary, this is set by boundary conditions and not chemistry.
    if np.any(ratio <= 0):
        raise ValueError("Chemistry must leave positive grain mass.")

    # Calculate the new dust radius based on the ratio
    radius = sim.dust.a * np.cbrt(ratio)

    # Calculate the midplane number density
    number_density = dust / (np.sqrt(2*np.pi) * sim.dust.H * sim.grid.m)

    # The sublimation process can take a maximum of 1% of the previous timestep, ensuring that the proposed release is fast enough to be considered instantaneous
    budget = 0.01 * max(float(sim.t.prevstepsize), 0.0)

    # Define bookkeeping array
    released = []

    # Loop over volatile species
    for name in sim.volatiles.names:
        volatile = getattr(sim.volatiles, name)
        ml, equilibrium, thermal_speed = volatile_quantities(sim, name)

        # Calculate the new monolayer surface density based on the new dust radius
        ml_new = ml * ((radius + volatile.radius) / (sim.dust.a + volatile.radius))**2

        # Only consider bins at the monolayer limit
        candidate = np.where(volatile.Sigmaice <= ml*(1.0 + 1e-10), np.maximum(volatile.Sigmaice - ml_new, 0.0), 0.0)

        # Inner boundary is not considered, this is set by boundary conditions and should not be included
        candidate[0] = 0.0

        # Calculate the deficit of the current species, including the proposed release
        deficit = np.maximum(equilibrium - volatile.Sigmavap - candidate.sum(axis=-1), 0.0)

        # Calculate the rate of sublimation based on the new dust radius and deficit
        rate = (np.pi * radius**2 * thermal_speed[:, None] * number_density * deficit[:, None])

        # Determine the amount of ice to release based on the budget, rate, and candidate values
        released.append(np.where((budget > 0) & (rate > 0) & (candidate <= budget*rate), candidate, 0.0))

    # Return the released ice surface densities for all volatile species, stacked into an array
    return np.stack(released)


def remapper(sim):
    """
    Remap the dust and all volatile ice surface densities, keeping
    particle radii and Stokes numbers consistent with ice gained/lost
    during the chemistry step.

    The total change in ice mass across all volatile species determines
    the movement of the dust grains in mass space. Each individual ice
    species is then remapped using the same movement.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame.

    Returns
    -------
    dust_new : ndarray
        Remapped dust surface density with shape (Nr, Nm).

    ice_new : ndarray
        Remapped ice surface densities with shape
        (Nvolatiles, Nr, Nm). The first dimension follows the order in
        sim.volatiles.species.
    """

    volatiles_names = sim.volatiles.names
    n_volatiles = len(volatiles_names)

    # Current dust distribution
    dust = np.copy(sim.dust.Sigma)

    # Ice distributions for all species
    # Shape: (Nvolatiles, Nr, Nm)
    ice = np.stack([
        np.copy(getattr(sim.volatiles, volatile_name).Sigmaice)
        for volatile_name in volatiles_names
    ])

    # Total change in grain mass due to all volatile chemistry
    # Shape: (Nr, Nm)
    ice_delta_total = sim.dust.Sigma.chemdelta.copy()

    # Bookkeeping arrays
    dust_new = np.zeros_like(dust)
    ice_new = np.zeros_like(ice)

    # Preserve inner radial cell since this is set by the boundary condition
    dust_new[0] = dust[0]
    ice_new[:, 0, :] = ice[:, 0, :]

    # Constants
    B = sim.grid.B
    log_B = np.log(B)
    n_r, n_sigma = dust.shape
    denom = 1.0 / (B - 1.0)

    # Apply Podolak remapping independently at each radius
    for r in range(1, n_r):

        dust_r = dust[r, :]
        ice_r = ice[:, r, :]
        ice_delta_r = ice_delta_total[r, :]

        # Total fractional change in grain mass caused by chemistry
        delta_frac_dust = (dust_r + ice_delta_r) / dust_r

        # Dust surface density after chemistry
        dust_chemistry = dust_r * delta_frac_dust

        # Number of mass-grid cells by which the grains move
        k = np.log(delta_frac_dust) / log_B

        # Left destination cell
        i = np.floor(k).astype(int)

        # Fractional displacement between the two destination cells
        delta = k - i

        # Weighting factors for the left and right destination cells
        B_delta_term = B**(1.0 - delta)

        weight_minus = (B_delta_term - 1.0) * denom
        weight_plus = (B - B_delta_term) * denom

        # Dust distributed between neighboring mass bins
        sigma_minus = dust_chemistry * weight_minus
        sigma_plus = dust_chemistry * weight_plus

        # Every volatile species follows the same grain displacement
        ice_minus = ice_r * weight_minus[None, :]
        ice_plus = ice_r * weight_plus[None, :]

        # Destination indices
        new_idxL = np.arange(n_sigma) + i
        new_idxR = new_idxL + 1

        valid_maskL = (new_idxL >= 0) & (new_idxL < n_sigma)
        valid_maskR = (new_idxR >= 0) & (new_idxR < n_sigma)

        valid_idxL = new_idxL[valid_maskL]
        valid_idxR = new_idxR[valid_maskR]

        #################################################
        ################ Dust remapping #################

        dust_r_new = np.zeros_like(dust_r)

        np.add.at(dust_r_new, valid_idxL, sigma_minus[valid_maskL])

        np.add.at(dust_r_new, valid_idxR, sigma_plus[valid_maskR])

        # Material moving outside the grid is placed in the edge bins
        dust_r_new[0] += (sigma_minus[new_idxL < 0].sum() + sigma_plus[new_idxR < 0].sum())

        dust_r_new[-1] += (sigma_minus[new_idxL >= n_sigma].sum() + sigma_plus[new_idxR >= n_sigma].sum())

        ###################################################
        ################### Ice remapping #################


        ice_r_new = np.zeros_like(ice_r)

        for j in range(n_volatiles):

            np.add.at(ice_r_new[j], valid_idxL, ice_minus[j, valid_maskL])

            np.add.at(ice_r_new[j],valid_idxR, ice_plus[j, valid_maskR])

            # Material moving outside the grid is placed in the edge bins
            ice_r_new[j, 0] += (ice_minus[j, new_idxL < 0].sum() + ice_plus[j, new_idxR < 0].sum())

            ice_r_new[j, -1] += (ice_minus[j, new_idxL >= n_sigma].sum() + ice_plus[j, new_idxR >= n_sigma].sum())

        dust_new[r, :] = dust_r_new
        ice_new[:, r, :] = ice_r_new

    return dust_new, ice_new
