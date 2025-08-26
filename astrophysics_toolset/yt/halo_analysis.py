from bisect import bisect
from typing import Union, Literal
from more_itertools import always_iterable

import numpy as np
import yt
from unyt.array import unyt_array, unyt_quantity

from .utils import conform_to_arr, conform_to_qty


def shrinking_sphere(
    ds,
    center: Union[tuple, unyt_array],
    radius: Union[tuple, unyt_quantity],
    *,
    center_on: str = "star",
    shrink_factor: float = 0.95,
) -> tuple[unyt_array, unyt_array]:
    """
    Compute the center of mass and velocity of a sphere by iteratively shrinking it.

    Parameters
    ----------
    ds : yt.data_objects.static_output.Dataset
        The dataset containing the sphere.
    center : tuple or unyt_array
        The initial guess for the center of the sphere.
    radius : tuple or unyt_quantity
        The search radius for the center.
    center_on : str, optional
        The particle type to center on, by default "star".
    shrink_factor : float, optional
        The fraction of particles to retain in the sphere, by default 95%.

    Returns
    -------
    tuple[unyt_array, unyt_array]
        The center and bulk velocity.
    """

    radius = conform_to_qty(ds, radius).to("code_length")
    center = conform_to_arr(ds, center).to("code_length")

    sp0 = ds.sphere(center, radius)
    pos = []
    vel = []
    m = []

    for pt in always_iterable(center_on):
        if pt == "gas":
            pos.append(
                np.stack(
                    [sp0[pt, k].to("code_length") for k in ("x", "y", "z")], axis=-1
                )
            )
            vel.append(
                np.stack(
                    [
                        sp0[pt, f"velocity_{k}"].to("code_velocity")
                        for k in ("x", "y", "z")
                    ],
                    axis=-1,
                )
            )
            m.append(sp0[pt, "cell_mass"][:, None].value)
        else:
            pos.append(sp0[pt, "particle_position"].to("code_length"))
            vel.append(sp0[pt, "particle_velocity"].to("code_velocity"))
            m.append(sp0[pt, "particle_mass"][:, None].value)

    pos = ds.arr(np.concatenate(pos), pos[0].units)
    vel = ds.arr(np.concatenate(vel), vel[0].units)
    m = np.concatenate(m)

    center_0 = center.copy()

    while len(pos) > 1000:
        yt.mylog.info(
            "Shrinking sphere radius: dx: %s kpc/h\tnpart: %8d",
            list((center - center_0).to("kpc/h").value),
            len(pos),
        )

        r = np.linalg.norm(pos - center, axis=1)
        order = np.argsort(r)

        # Retain the x% closest to the center
        new_len = min(int(len(pos) * shrink_factor), len(pos) - 1)
        pos = pos[order][:new_len]
        m = m[order][:new_len]
        vel = vel[order][:new_len]

        # Compute new center of mass
        center = np.sum((pos * m), axis=0) / m.sum()

    # Compute center velocity
    center_velocity = np.sum((vel * m), axis=0) / m.sum()

    return center, center_velocity


def virial_quantities(
    ds,
    center,
    radius=None,
    rmax=None,
    rho_def: Literal["critical", "matter"] = "critical",
    overdensity: float = 200.0,
    use_gas: bool = True,
    use_particles: bool = True,
    particle_types: tuple[str] = ("star", "DM"),
) -> unyt_quantity:
    """
    Compute the virial radius of a halo.

    Parameters
    ----------
    ds : yt.data_objects.static_output.Dataset
        The dataset containing the halo.
    center : tuple or unyt_array
        The center of the halo.
    radius : unyt_quantity, optional
        The initial guess for the virial radius, by default None.
    rmax : unyt_quantity, optional
        The maximum radius to search for the virial radius, by default None.
    rho_def : Literal["critical", "matter"], optional
        The definition of the density, by default "critical".
    overdensity : float, optional
        The overdensity threshold for the virial radius, by default 200.0.
    use_gas : bool, optional
        Whether to include gas, by default True.
    use_particles : bool, optional
        Whether to include particles, by default True.
    particle_types : tuple[str], optional
        The particle types to include, by default ("star", "DM").

    Returns
    -------
    unyt_quantity, unyt_quantity
        The virial radius and virial mass

    """
    particle_types = list(always_iterable(particle_types))

    if rmax is None:
        rmax = ds.domain_width.max()
    else:
        rmax = conform_to_qty(ds, rmax).to("code_length")

    if not use_gas and not use_particles:
        raise ValueError(
            "You need to specify gas or particles to compute virial quantities."
        )
    if use_particles and not particle_types:
        raise ValueError(
            "You need to specify particle types to compute virial quantities."
        )
    for pt in particle_types if use_particles else []:
        if pt not in ds.particle_types:
            raise ValueError(f"Particle type {pt} not found in dataset.")
    if rho_def not in ("critical", "matter"):
        raise ValueError("Invalid density definition. Use 'critical' or 'matter'.")

    new_center, new_velocity = shrinking_sphere(
        ds, center, radius, center_on=particle_types
    )

    sp = ds.sphere(new_center, rmax)
    sp.set_field_parameter("bulk_velocity", new_velocity)
    rr = []
    mm = []

    if use_gas:
        rr.append(sp["index", "radius"].to("code_length"))
        mm.append(sp["gas", "cell_mass"].to("Msun"))

    if use_particles:
        for pt in particle_types:
            rr.append(sp[pt, "particle_radius"].to("code_length"))
            mm.append(sp[pt, "particle_mass"].to("Msun"))

    rr = ds.arr(np.concatenate(rr), rr[0].units)
    mm = ds.arr(np.concatenate(mm), mm[0].units)

    order = np.argsort(rr)
    rr = rr[order]
    mm = mm[order].cumsum()

    rho = mm / (4 / 3 * np.pi * rr**3)

    # Compute
    rhoc = ds.cosmology.critical_density(ds.current_redshift)
    rhom = rhoc * ds.cosmology.omega_matter * (1 + ds.current_redshift) ** 3

    if rho_def == "critical":
        rho_tgt = rhoc * overdensity
    elif rho_def == "matter":
        rho_tgt = rhom * overdensity
    else:
        raise ValueError("Invalid density definition. Use 'critical' or 'matter'.")

    ind = bisect(-rho, -rho_tgt.to(rho.units))
    Rvir = rr[ind]
    Mvir = mm[ind]

    return new_center, Rvir.to("kpc"), Mvir.to("Msun")
