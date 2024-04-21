from typing import Union

import numpy as np
import yt
from unyt.array import unyt_array, unyt_quantity


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

    if isinstance(radius, tuple):
        radius = ds.quan(*radius).to("code_length")
    if isinstance(center, tuple):
        center = ds.arr(*center)

    sp0 = ds.sphere(center, radius)
    pos = sp0[center_on, "particle_position"].to("code_length")
    vel = sp0[center_on, "particle_velocity"].to("code_velocity")
    m = sp0[center_on, "particle_mass"][:, None].value

    center_0 = center.copy()

    while len(pos) > 1000:
        yt.mylog.info(
            "Shrinking sphere radius: dx: %s kpc/h\tnpart: %8d"
            % (
                (center - center_0).to("kpc/h").value,
                len(pos),
            )
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
