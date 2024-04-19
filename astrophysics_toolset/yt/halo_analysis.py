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
):
    if isinstance(radius, tuple):
        radius = ds.quan(*radius).to("code_length")
    if isinstance(center, tuple):
        center = ds.arr(*center)

    sp0 = ds.sphere(center, radius)
    pos = sp0[center_on, "particle_position"].to("code_length")
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

        # Compute new center of mass
        center = np.sum((pos * m), axis=0) / m.sum()

    return center
