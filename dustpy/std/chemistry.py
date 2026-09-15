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

    ##############  Surface density of one monolayer ################


    sigma_mol = np.pi * r_mol**2

    ML_per_grain = (
        4.0 * np.pi * (a + r_mol)**2
        / sigma_mol
        * m_mol
    )

    Sigma_ML = (sim.dust.Sigma / m) * ML_per_grain

    #################################################################
    ############## Equilibrium vapor surface density ################

    P_eq = vapor_pressure(T)

    Sigma_eq_vap = (
        np.sqrt(2.0 * np.pi)
        * H_P
        * m_mol
        * P_eq
        / (ac.k_B.cgs.value * T)
    )

    Sigma_eq_vap = np.clip(
        Sigma_eq_vap,
        1e-300,
        None,
    )

    #################################################################
    ###################### Thermal velocity #########################

    v_therm = np.sqrt(
        8.0
        * ac.k_B.cgs.value
        * T
        / (np.pi * m_mol)
    )

    return Sigma_ML, Sigma_eq_vap, v_therm

def _constant_radius_cell(dt, ice, vapor, equilibrium, collision, monolayer):
    """Analytic phase exchange at fixed geometry, with depletion events.

    Arrays have shape (species, bins), except vapor/equilibrium (species,).
    Each species sees the same geometry and its own conserved vapor pool.
    """
    change = np.zeros_like(ice)
    for x in range(len(vapor)):
        deficit = float(vapor[x]-equilibrium[x])
        if deficit == 0.0:
            continue
        condensing = deficit > 0.0
        active = collision[x] > 0.0
        if not condensing:
            active &= ice[x] > monolayer[x]
        remaining = float(dt)
        for _ in range(ice.shape[1]+1):
            rates = np.where(active, collision[x], 0.0)
            total = rates.sum()
            if total == 0.0 or remaining <= 0.0:
                break
            weights = rates/total
            available = max(0.0, np.sign(deficit)*(deficit-change[x].sum()))
            if available == 0.0:
                break
            event_time = np.inf
            if not condensing:
                capacity = np.maximum(ice[x]+change[x]-monolayer[x], 0.0)
                required = np.divide(capacity, weights, out=np.full_like(weights, np.inf), where=active)
                first = int(np.argmin(required))
                fraction = required[first]/available
                if fraction < 1.0:
                    event_time = -np.log1p(-fraction)/total
            step = min(remaining, event_time)
            with np.errstate(over='ignore'):
                transferred = available*(-np.expm1(-total*step))
            delta = weights*transferred
            if not condensing:
                delta = -np.minimum(delta, capacity)
            change[x] += delta
            finished = step == remaining
            remaining = max(0.0, remaining-step)
            if finished:
                # The full remaining interval has been integrated.
                break
            if event_time <= step:
                active[first] = False
            else:
                break
    return change


def sublimation_condensation(sim, volatile_name):
    """Original constant-radius chemistry interface (no radius adaptation)."""
    species = getattr(sim.volatiles, volatile_name)
    ml, equilibrium, speed = volatile_quantities(sim, volatile_name)
    collision = np.pi*sim.dust.a**2*(sim.dust.rho/sim.grid.m)*speed[:, None]
    change = np.zeros_like(species.Sigmaice)
    for r in range(len(sim.grid.r)):
        change[r] = _constant_radius_cell(
            float(sim.t.prevstepsize), np.asarray(species.Sigmaice[r])[None, :],
            np.array([species.Sigmavap[r]]), np.array([equilibrium[r]]),
            collision[r][None, :], ml[r][None, :])[0]
    return change, -change.sum(axis=-1)


# Retain the explicit name for comparisons with adaptive chemistry.
sublimation_condensation_constant_radius = sublimation_condensation


def _adaptive_cell(dt, ice, vapor, equilibrium, dust, radius, collision,
                   monolayer, molecular_radius, radius_tolerance=0.01,
                   max_substeps=10000):
    """Local analytic substeps; all species share each frozen geometry.

    monolayer*(a+r_mol)**2 is the monolayer inventory. Particle numbers,
    density and temperature stay fixed until the final, single remapping.
    """
    change = np.zeros_like(ice)
    remaining = float(dt)
    # Bound gross rather than net mass exchange so simultaneous growth and
    # shrinkage cannot hide a large intermediate excursion of the radius.
    mass_tolerance = 1.0-(1.0-radius_tolerance)**3
    for _ in range(max_substeps):
        if remaining <= 0.0 or ice.size == 0:
            return change
        mass = dust+change.sum(axis=0)
        if np.any(mass <= 0.0):
            raise RuntimeError('Chemistry exhausted the grain mass.')
        a = radius*np.cbrt(mass/dust)
        rates = collision*(a/radius)[None, :]**2
        ml = monolayer*(a[None, :]+molecular_radius[:, None])**2
        current_ice = ice+change
        current_vapor = vapor-change.sum(axis=1)
        h = remaining
        for _ in range(100):
            trial = _constant_radius_cell(h, current_ice, current_vapor,
                                          equilibrium, rates, ml)
            gross = np.abs(trial).sum(axis=0)
            if np.all(gross <= mass_tolerance*mass):
                break
            # Use the instantaneous net phase rates only after a full-step
            # trial fails: rapid microscopic exchange near equilibrium need
            # not force small steps when the total transfer is negligible.
            flux = rates*(current_vapor-equilibrium)[:, None]
            flux = np.where((flux < 0.0) & (current_ice <= ml), 0.0, flux)
            # A nearly bare species can have an enormous instantaneous
            # sublimation rate but almost no ice left to transfer. Its
            # depletion is handled analytically; it must not dictate the
            # timestep for a slower species with a substantial mantle.
            significant = np.abs(trial) > mass_tolerance*mass/(4*len(vapor))
            flux = np.where(significant, flux, 0.0)
            gross_rate = np.abs(flux).sum(axis=0)
            bound = np.divide(mass_tolerance*mass, gross_rate,
                              out=np.full_like(mass, np.inf), where=gross_rate > 0)
            h = min(0.5*h, 0.8*bound.min())
            if h <= 0.0:
                raise RuntimeError('Adaptive chemistry timestep underflow.')
        else:
            raise RuntimeError('Adaptive chemistry could not bound the radius change.')
        change += trial
        remaining = max(0.0, remaining-h)
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
    names = tuple(sim.volatiles.names)
    ice = np.stack([np.asarray(getattr(sim.volatiles, n).Sigmaice) for n in names])
    vapor = np.stack([np.asarray(getattr(sim.volatiles, n).Sigmavap) for n in names])
    dt = float(sim.t.prevstepsize)
    variable_radius = getattr(sim.volatiles, 'chemistry_variable_radius', True)
    if radius_tolerance is None:
        radius_tolerance = getattr(sim.volatiles, 'chemistry_radius_tolerance', 0.01)
    if not np.isfinite(dt) or dt < 0:
        raise ValueError('Chemistry timestep must be finite and nonnegative.')
    if not np.isfinite(radius_tolerance) or not 0 < radius_tolerance < 1:
        raise ValueError('radius_tolerance must lie strictly between zero and one.')
    if not isinstance(max_substeps, (int, np.integer)) or max_substeps < 1:
        raise ValueError('max_substeps must be a positive integer.')
    change = np.zeros_like(ice)
    if dt == 0:
        return change, np.zeros_like(vapor)
    if np.any(sim.dust.Sigma.chemdelta):
        raise ValueError('Chemistry requires a fresh chemistry step.')
    dust = np.asarray(sim.dust.Sigma)
    radius = np.asarray(sim.dust.a)
    mass = np.asarray(sim.grid.m)
    number = dust / mass
    n_midplane = np.asarray(sim.dust.rho) / mass
    floor = np.asarray(sim.dust.SigmaFloor)
    resolved = dust > floor
    invalid = (dust < 0) | np.any(ice < 0, axis=0) | (
        resolved & (ice.sum(axis=0) > dust*(1+1e-10)))
    if np.any(invalid):
        r, k = np.argwhere(invalid)[0]
        raise ValueError(f'Inconsistent total ice at radial bin {r}, mass bin {k}: '
                         f'ice={ice[:,r,k].sum():.17g}, dust={dust[r,k]:.17g}.')
    quantities = [volatile_quantities(sim, n) for n in names]
    equilibrium = np.stack([q[1] for q in quantities])
    velocity = np.stack([q[2] for q in quantities])
    rm = np.array([float(getattr(sim.volatiles, n).radius) for n in names])
    mm = np.array([float(getattr(sim.volatiles, n).mass) for n in names])
    mono = (4*mm/rm**2)[:, None, None] * number[None, :, :]
    collision = np.pi*radius[None, :, :]**2*n_midplane[None, :, :]*velocity[:, :, None]
    for r in range(dust.shape[0]):
        use = resolved[r] & (dust[r] > 0)
        if not np.any(use):
            continue
        try:
            if variable_radius:
                change[:, r, use] = _adaptive_cell(
                    dt, ice[:, r, use], vapor[:, r], equilibrium[:, r],
                    dust[r, use], radius[r, use], collision[:, r, use],
                    mono[:, r, use], rm, radius_tolerance=radius_tolerance,
                    max_substeps=max_substeps)
            else:
                monolayer = mono[:, r, use] * (radius[r, use][None, :] + rm[:, None])**2
                change[:, r, use] = _constant_radius_cell(
                    dt, ice[:, r, use], vapor[:, r], equilibrium[:, r],
                    collision[:, r, use], monolayer)
        except RuntimeError as error:
            raise RuntimeError(f'Chemistry at radial bin {r}: {error}') from error
    return change, -change.sum(axis=-1)


def monolayer_shrinkage(sim):
    """Return ice to release once, before remapping (Nspecies, Nr, Nm).

    Ordinary chemistry has already set dust.Sigma.chemdelta. Use that mass
    change to estimate the new radius at fixed grain number and material
    density. Only bins at/below the old monolayer are corrected; thick ice
    mantles remain under ordinary chemistry. No underfilled layer is filled.

    This assumes fast sublimation: the proposed release must take <= 1% of
    the previous timestep, using the vapor deficit AFTER the whole proposed
    transfer. Saturated and slow bins are left unchanged. The extra radius
    reduction from this release is deliberately not iterated, but its mass
    loss must still be included in the subsequent remapping.
    """
    dust = np.asarray(sim.dust.Sigma)
    ratio = 1.0 + np.divide(sim.dust.Sigma.chemdelta, dust, out=np.zeros_like(dust), where=dust > 0)
    ratio[0] = 1.0  # The original remapper preserves the inner boundary.
    if np.any(ratio <= 0):
        raise ValueError("Chemistry must leave positive grain mass.")

    radius = sim.dust.a * np.cbrt(ratio)
    number_density = dust / (np.sqrt(2*np.pi) * sim.dust.H * sim.grid.m)
    budget = 0.01 * max(float(sim.t.prevstepsize), 0.0)

    released = []
    for name in sim.volatiles.names:
        volatile = getattr(sim.volatiles, name)
        ml, equilibrium, thermal_speed = volatile_quantities(sim, name)

        # Grain number is unchanged; only the surface per grain changes.
        ml_new = ml * ((radius + volatile.radius) / (sim.dust.a + volatile.radius))**2
        candidate = np.where(volatile.Sigmaice <= ml*(1.0 + 1e-10), np.maximum(volatile.Sigmaice - ml_new, 0.0), 0.0)
        candidate[0] = 0.0

        deficit = np.maximum(equilibrium - volatile.Sigmavap - candidate.sum(axis=-1), 0.0)
        rate = (np.pi * radius**2 * thermal_speed[:, None] * number_density * deficit[:, None])
        released.append(np.where((budget > 0) & (rate > 0) & (candidate <= budget*rate), candidate, 0.0))
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
