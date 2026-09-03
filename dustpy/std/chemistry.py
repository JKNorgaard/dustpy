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

def delta_ice_analytic(dt, Ci, C_sum, Ei, vapor):
    """
    Analytic change in ice surface density per bin over dt.

    Shapes:
      Ci, Ei, ice: (Nr, Nm)
      C_sum:       (Nr, 1)
      vapor:       (Nr,)
    """
    vapor_2d = vapor[:, None]  # (Nr,1)

    x = C_sum * dt
    one_minus_exp = 1.0 - np.exp(-x)

    num = (Ci * vapor_2d - Ei)  # (Nr,Nm)

    C_safe = np.where(C_sum > 0.0, C_sum, 1.0) 
    return np.where(C_sum > 0.0, (num / C_safe) * one_minus_exp, num * dt)

def sublimation_substepping(dt, Ci, Ei, ice, vapor, sublimation_mask, ML):

    d_ice_subl = np.zeros_like(ice)

    idx = np.flatnonzero(sublimation_mask)
    if idx.size == 0:
        return d_ice_subl

    # Work only on sublimating radii
    ice_s = ice[idx].copy()
    ML_s  = ML[idx]
    Ci_s  = Ci[idx]
    Ei_s  = Ei[idx]
    vap_s = vapor[idx].copy()

    dt_used = 0.0
    it = 0
    max_iter = ice_s.shape[0] * ice_s.shape[1]

    tmp = np.empty_like(ice_s, dtype=float)

    while dt_used < dt:
        it += 1
        if it > max_iter:
            break

        active = (ice_s > ML_s)
        if not np.any(active):
            break

        Ci_use = Ci_s * active
        Ei_use = Ei_s * active
        C_rest = np.sum(Ci_use, axis=-1, keepdims=True)  # (Ns,1)

        # if nothing actually sublimates, stop
        if not np.any(C_rest > 0.0):
            break

        denom = Ci_use * vap_s[:, None] - Ei_use

        # ratio = ((ice-ML)*C_rest + denom) / denom
        numer = (ice_s - ML_s) * C_rest + denom

        ratio = np.full_like(denom, np.inf, dtype=float)
        valid = (C_rest > 0.0) & active & (denom < 0.0)
        np.divide(numer, denom, out=ratio, where=valid)

        valid2 = valid & np.isfinite(ratio) & (ratio > 0.0) & (ratio < 1.0)
        if np.any(valid2):
            # dt_candidate = -log(ratio)/C_rest
            np.copyto(tmp, np.inf)
            tmp.fill(np.inf)

            if np.any(valid2):
                r_idx, c_idx = np.where(valid2)
                tmp[r_idx, c_idx] = -np.log(ratio[r_idx, c_idx]) / C_rest[r_idx, 0]
                dt_sub = tmp.min()
            else:
                dt_sub = np.inf
        else:
            dt_sub = np.inf

        dt_rem = dt - dt_used
        dt_step = dt_rem if (not np.isfinite(dt_sub) or dt_sub >= dt_rem) else dt_sub

        # Analytic update for this substep (only Ns radii)
        dice_full = delta_ice_analytic(dt_step, Ci_use, C_rest, Ei_use, vap_s)

        # Enforce monolayer floor only where sublimating
        ice_prev = ice_s
        ice_trial = ice_prev + dice_full
        ice_new = np.maximum(ice_trial, ML_s)
        dice_eff = ice_new - ice_prev

        # Accumulate + update state
        d_ice_subl[idx, :] += dice_eff
        ice_s = ice_new
        vap_s += -np.sum(dice_eff, axis=-1)

        dt_used += dt_step
        if dt_step >= dt_rem:
            break

    return d_ice_subl


def sublimation_condensation(sim, volatile_name):
    """
    Returns:
      d_ice: (Nr, Nm)
      d_vap: (Nr,)
    """
    # Save ice and vapor before chemistry
    ice0 = np.array(getattr(sim.volatiles, volatile_name).Sigmaice, copy=True)
    vap0 = np.array(getattr(sim.volatiles, volatile_name).Sigmavap, copy=True)

    # Initialize bookkeeping arrays for chemistry contributions
    d_ice = np.zeros_like(ice0)
    d_vap = np.zeros_like(vap0)

    # Stepsize of transport step
    dt = sim.t.prevstepsize

    # Calculate species-specific quantities for sublimation and condensation
    Sigma_ML, Sigma_eq_vap, v_therm = volatile_quantities(sim, volatile_name)

    # Simulation parameters for chemistry rates
    a = sim.dust.a
    n_dust = sim.dust.rho / sim.grid.m 

    # Chemistry rates
    Ci = np.pi * a**2 * v_therm[:, None] * n_dust
    Ei = Ci * Sigma_eq_vap[:, None]

    # Clip numerically problematic quantities
    Ci = np.clip(Ci, 0, None) 
    Ei = np.clip(Ei, 0, None) 

    # Create regime masks
    condensation = vap0 > Sigma_eq_vap
    sublimation = ~condensation

    # Condensation (analytic, all bins participate)
    if np.any(condensation):
        C_all = np.sum(Ci, axis=-1, keepdims=True) # Sum over mass bins
        dice_all = delta_ice_analytic(dt, Ci, C_all, Ei, vap0) # Analytic change in ice

        # Only save radii in condensation regime
        d_ice[condensation, :] = dice_all[condensation, :] 

    # Sublimation (substepping, restricted set)
    if np.any(sublimation):
        # Substep to only consider non-bare dust grains
        d_ice_subl = sublimation_substepping(dt, Ci, Ei, ice0, vap0, sublimation, Sigma_ML)

        # Only save radii in sublimation regime
        d_ice[sublimation, :] = d_ice_subl[sublimation, :]

    d_vap = -np.sum(d_ice, axis=-1)
    
    return d_ice, d_vap

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