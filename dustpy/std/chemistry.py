import numpy as np
from numba import njit
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

@njit(cache=True, error_model="numpy")
def _constant_radius(dt, ice, vapor, equilibrium, collision, monolayer):
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

    for x in range(len(vapor)):
        deficit = float(vapor[x] - equilibrium[x])
        if deficit == 0.0:
            continue
        condensing = deficit > 0.0
        for j in range(nbins):
            active[j] = collision[x, j] > 0.0 and (
                condensing or ice[x, j] > monolayer[x, j])
        remaining = float(dt)

        for _ in range(nbins + 1):
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
                   max_substeps=10000, carbon=None, carbon_rate=0.,
                   methane_index=None, methane_mass_ratio=1., variable_radius=True):
    """ Adaptive phase exchange for one radial bin with variable grain radius, with depletion substepping.
    
    Arrays have shape (species, bins), except vapor/equilibrium (species,).
    """

    # Initialize bookkeeping array for change in ice surface density
    ice_change = np.zeros_like(ice)
    carbon_change = np.zeros_like(dust)
    has_carbon = carbon is not None

    def result():
        return (ice_change, carbon_change) if has_carbon else ice_change

    # Initialize the remaining time for the current timestep
    remaining = float(dt)

    # Convert fractional radius tolerance to mass tolerance 
    mass_tolerance = 1.0-(1.0-radius_tolerance)**3

    # Loop over substeps with a maximum of max_substeps iterations to ensure convergence
    for _ in range(max_substeps):

        # If there is no remaining time or no ice to transfer, return the changes
        if remaining <= 0.0:
            return result()

        # Calculate the total mass of dust and volatile ices in each bin
        mass = dust+ice_change.sum(axis=0)+carbon_change

        # Calculate the new grain radius 
        if variable_radius:
            a = radius*np.cbrt(mass/dust)
        else:
            a = radius

        # Calculate the collision rates and monolayer surface densities for new grain radius
        rates = collision*(a/radius)[None, :]**2
        ml = monolayer*(a[None, :]+molecular_radius[:, None])**2

        # Update ice and vapor for change in surface density
        current_ice = ice+ice_change
        current_vapor = vapor-ice_change.sum(axis=1)
        if has_carbon:
            current_carbon = np.maximum(carbon+carbon_change, 0.)
            current_vapor[methane_index] -= methane_mass_ratio*carbon_change.sum()


        h = remaining
        if has_carbon and carbon_rate > 0:
            # An exact bound avoids repeatedly halving long trials when
            # destruction is very fast. Exhaustible small reservoirs do not
            # restrict the timestep once their entire mass fits the budget.
            fraction = np.divide(mass_tolerance*mass, current_carbon,
                out=np.full_like(mass, np.inf), where=current_carbon > 0)
            limiting = fraction < 1
            if np.any(limiting):
                h = min(h, float(np.min(-np.log1p(-fraction[limiting])/carbon_rate)))
        for _ in range(100):
            trial_carbon = np.zeros_like(dust)
            if has_carbon:
                with np.errstate(over='ignore'):
                    trial_carbon = current_carbon*np.expm1(-carbon_rate*h)
                # Newly produced methane becomes available after this
                # substep, when the full carbon loss is committed.
                mid_a = radius*np.cbrt((mass+0.5*trial_carbon)/dust) if variable_radius else radius
                trial_rates = collision*(mid_a/radius)[None, :]**2
                trial_ml = monolayer*(mid_a[None, :]+molecular_radius[:, None])**2
            else:
                trial_rates, trial_ml = rates, ml
            # Calculate the trial change in ice surface density for the current substep
            trial = _constant_radius(h, current_ice, current_vapor, equilibrium, trial_rates, trial_ml)

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

            # Only use bins that are significant for determining the timestep, and set flux to zero for bins with insignificant change
            flux = np.where(significant, flux, 0.0)

            # Calculate the gross rate of significant bins
            gross_rate = np.abs(flux).sum(axis=0)
            if has_carbon:
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
        remaining = max(0.0, remaining-h)

        # If there is no remaining time, return the change in ice surface density
        if remaining == 0.0:
            return result()
    raise RuntimeError('Adaptive chemistry exceeded max_substeps.')


def sublimation_condensation_all(sim, radius_tolerance=None, max_substeps=10000):
    """Ice-only interface. Carbon-enabled callers must use solid_chemistry_all."""
    if hasattr(sim, 'refractory_carbon'):
        raise ValueError('Use solid_chemistry_all for coupled carbon/ice chemistry.')
    ice, vapor, _ = solid_chemistry_all(sim, radius_tolerance, max_substeps)
    return ice, vapor


def refractory_carbon_rate(temperature, prefactor=4.e13, activation_temperature=24500.):
    """Organic-carbon decay rate [s^-1]; Gail & Trieloff (2017), Table A.2.

    Temperature is in K. These adopted organic-pyrolysis coefficients do not
    describe pure graphite sublimation. CH4 yield is a separate assumption.
    """
    temperature = np.asarray(temperature, dtype=float)
    return prefactor*np.exp(-activation_temperature/temperature)


def solid_chemistry_all(sim, radius_tolerance=None, max_substeps=10000):
    """Return ice, vapor, carbon changes: (Ns,Nr,Nm), (Ns,Nr), (Nr,Nm).

    Call after transport of ALL species and before any chemistry/remapping.
    T, H_d, N, solid density and equilibrium vapor pressures remain fixed.
    Geometry varies unless sim.volatiles.chemistry_variable_radius is
    disabled. In variable-radius mode radius_tolerance bounds excursions within
    each substep, defaulting to chemistry_radius_tolerance (0.01).
    """

    # Get the names of all volatile species in the simulation
    names = tuple(sim.volatiles.names)
    carbon_species = getattr(sim, 'refractory_carbon', None)
    has_carbon = carbon_species is not None

    # Stack the ice and vapor surface densities for all species into arrays
    ice = np.stack([np.asarray(getattr(sim.volatiles, n).Sigmaice) for n in names])
    vapor = np.stack([np.asarray(getattr(sim.volatiles, n).Sigmavap) for n in names])

    # Get the timestep size
    dt = float(sim.t.prevstepsize)

    # Check if variable radius chemistry is enabled in the simulation, defaulting to True if not specified
    variable_radius = getattr(sim.volatiles, 'chemistry_variable_radius', True)

    if radius_tolerance is None:
        radius_tolerance = getattr(sim.volatiles, 'chemistry_radius_tolerance', 0.01)

    # Initialize bookkeeping array for change in ice and carbon surface density
    ice_change = np.zeros_like(ice)
    carbon_change = np.zeros_like(sim.dust.Sigma)
    if dt == 0:
        return ice_change, np.zeros_like(vapor), carbon_change

    # Define simulation quantities for the adaptive chemistry calculations
    dust = np.asarray(sim.dust.Sigma)
    radius = np.asarray(sim.dust.a)
    mass = np.asarray(sim.grid.m)
    n_d = dust / mass
    n_midplane = np.asarray(sim.dust.rho) / mass
    floor = np.asarray(sim.dust.SigmaFloor)

    # Check for invalid conditions in the dust and ice surface densities
    resolved = dust > floor
    if has_carbon:
        carbon = np.asarray(carbon_species.Sigma)
    else:
        carbon = np.zeros_like(dust)

    if has_carbon:
        rate = refractory_carbon_rate(sim.gas.T, float(carbon_species.prefactor), float(carbon_species.activation_temperature))
        methane_index = names.index('CH4')
        methane_ratio = float(sim.volatiles.CH4.mass/carbon_species.mass)

        carbon_change = carbon*np.expm1(-rate[:, None]*dt)
        carbon_change[~resolved] = 0.
        carbon_change[[0, -1]] = 0.
    
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

        # Boundary rows are imposed by transport; the inner dust row is
        # also preserved by the remapper. Do not generate vapor there.
        if has_carbon and r in (0, dust.shape[0]-1):
            continue

        # Only consider radial bins that have more dust than the floor value
        use = resolved[r]
        if not np.any(use):
            continue

        # Calculate change in ice surface density using adaptive chemistry or constant-radius chemistry based on the variable_radius flag
        try:
            # At machine-scale loss, retain the original ice solver. The
            # exact (possibly tiny) carbon loss still feeds methane below.
            reactive = has_carbon and np.any(
                -carbon_change[r, use] > 8*np.finfo(float).eps*dust[r, use])
            if reactive:
                ice_change[:, r, use], carbon_change[r, use] = _adaptive_radius(
                    dt, ice[:, r, use], vapor[:, r], equilibrium[:, r],
                    dust[r, use], radius[r, use], collision[:, r, use],
                    mono[:, r, use], r_mol, radius_tolerance, max_substeps,
                    carbon=carbon[r, use], carbon_rate=rate[r],
                    methane_index=methane_index, methane_mass_ratio=methane_ratio,
                    variable_radius=variable_radius)
            elif variable_radius:
                # If variable_radius is True, use the adaptive radius chemistry method
                ice_change[:, r, use] = _adaptive_radius(
                    dt, ice[:, r, use], vapor[:, r], equilibrium[:, r],
                    dust[r, use], radius[r, use], collision[:, r, use],
                    mono[:, r, use], r_mol, radius_tolerance=radius_tolerance,
                    max_substeps=max_substeps)
            else:
                # If variable_radius is False, use the constant-radius chemistry method
                monolayer = mono[:, r, use] * (radius[r, use][None, :] + r_mol[:, None])**2
                ice_change[:, r, use] = _constant_radius(
                    dt, ice[:, r, use], vapor[:, r], equilibrium[:, r],
                    collision[:, r, use], monolayer)
        except RuntimeError as error:
            raise RuntimeError(f'Chemistry at radial bin {r}: {error}') from error

        # Return the change in ice surface density and the negative sum of the change equal to the change in vapor surface density
    vapor_change = -ice_change.sum(axis=-1)
    if has_carbon:
        vapor_change[methane_index] -= methane_ratio*carbon_change.sum(axis=-1)
    return ice_change, vapor_change, carbon_change


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


def remapper(sim, include_carbon=False):
    """
    Remap dust, volatile ice, and optional refractory carbon, keeping
    particle radii and Stokes numbers consistent with solid mass gained/lost
    during the chemistry step.

    The total change in ice mass across all volatile species determines
    the movement of the dust grains in mass space. Each individual ice
    species is then remapped using the same movement.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame.
    include_carbon : bool
        Required for carbon-bearing simulations; return a third array.

    Returns
    -------
    dust_new : ndarray
        Remapped dust surface density with shape (Nr, Nm).

    ice_new : ndarray
        Remapped ice surface densities with shape
        (Nvolatiles, Nr, Nm). The first dimension follows the order in
        sim.volatiles.names.
    carbon_new : ndarray, optional
        Remapped carbon surface density (Nr, Nm), when include_carbon=True.
    """

    has_carbon = hasattr(sim, 'refractory_carbon')
    if has_carbon and not include_carbon:
        raise ValueError('Use include_carbon=True when remapping carbon-bearing grains.')

    # Only the stacked tracer array needs a copy; the kernel never mutates inputs.
    dust = np.asarray(sim.dust.Sigma)
    tracers = [np.asarray(getattr(sim.volatiles, name).Sigmaice)
               for name in sim.volatiles.names]
    if has_carbon:
        tracers.append(np.asarray(sim.refractory_carbon.Sigma))
    dust_new, ice_new = _remap_arrays(
        dust, np.stack(tracers), np.asarray(sim.dust.Sigma.chemdelta),
        float(sim.grid.B))

    if include_carbon:
        return dust_new, ice_new[:-1] if has_carbon else ice_new, (
            ice_new[-1] if has_carbon else np.zeros_like(dust))
    return dust_new, ice_new


@njit(cache=True, error_model="numpy")
def _remap_arrays(dust, ice, ice_delta_total, B):
    """Redistribute positive grain masses on a logarithmic mass grid.

    Dust and chemistry deltas have shape (Nr, Nm); tracers (Ns, Nr, Nm).
    Inputs are read-only. Out-of-grid destinations accumulate in edge bins.
    """
    dust_new = np.zeros_like(dust)
    ice_new = np.zeros_like(ice)
    dust_new[0] = dust[0]
    ice_new[:, 0, :] = ice[:, 0, :]
    log_B = np.log(B)
    denom = 1.0 / (B - 1.0)
    nr, nm = dust.shape

    for r in range(1, nr):
        for j in range(nm):
            ratio = (dust[r, j] + ice_delta_total[r, j]) / dust[r, j]
            mass = dust[r, j] * ratio
            displacement = np.log(ratio) / log_B
            offset = int(np.floor(displacement))
            fraction = displacement - offset
            power = B ** (1.0 - fraction)
            weight_left = (power - 1.0) * denom
            weight_right = (B - power) * denom
            left = min(max(j + offset, 0), nm - 1)
            right = min(max(j + offset + 1, 0), nm - 1)

            # Direct scatter replaces per-row masks, temporaries and np.add.at.
            dust_new[r, left] += mass * weight_left
            dust_new[r, right] += mass * weight_right
            for x in range(ice.shape[0]):
                ice_new[x, r, left] += ice[x, r, j] * weight_left
                ice_new[x, r, right] += ice[x, r, j] * weight_right
    return dust_new, ice_new
