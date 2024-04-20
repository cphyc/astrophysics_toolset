from functools import partial

import unyt as u

from astrophysics_toolset.yt.emission_lines import nuclide_data

elements = (
    "H",
    "He",
    "CO",
    "calcium",
    "carbon",
    "magnesium",
    "neon",
    "nitrogen",
    "oxygen",
    "silicon",
    "suflur",
)


def add_element_densities(ds):
    def element_density(field, data, *, elem_fraction):
        return data["gas", "density"] * data[elem_fraction]

    def element_number_density(field, data, *, elem_fraction, atomic_mass):
        return data["gas", "density"] * data[elem_fraction] / atomic_mass

    def ion_density(field, data, *, elem_fraction, ion_fraction):
        return data["gas", "density"] * data[elem_fraction] * data[ion_fraction]

    def ion_number_density(field, data, *, elem_fraction, ion_fraction, atomic_mass):
        return (
            data["gas", "density"]
            * data[elem_fraction]
            * data[ion_fraction]
            / atomic_mass
        )

    for elem in elements:
        elem_fraction = ("ramses", f"hydro_{elem}_fraction")
        if elem_fraction not in ds.field_list:
            continue

        # Element-by-element densities
        ds.add_field(
            ("gas", f"{elem}_density"),
            function=partial(element_density, elem_fraction=elem_fraction),
            units="g/cm**3",
            sampling_type="cell",
        )
        if elem == "CO":
            atomic_weight = (
                nuclide_data.getStandardAtomicWeight("C") * u.mp
                + nuclide_data.getStandardAtomicWeight("O") * u.mp
            )
        else:
            atomic_weight = nuclide_data.getStandardAtomicWeight(elem) * u.mp
        # Element-by-element number densities
        ds.add_field(
            ("gas", f"{elem}_number_density"),
            function=partial(
                element_number_density,
                elem_fraction=elem_fraction,
                atomic_mass=atomic_weight,
            ),
            units="cm**-3",
            sampling_type="cell",
        )

        for iion in range(20):
            ion_fraction = ("ramses", f"hydro_{elem}_{iion:02}")
            if ion_fraction not in ds.field_list:
                continue

            # Ion fractions
            ds.add_field(
                ("gas", f"{elem}_{iion:02}_density"),
                function=partial(
                    ion_density, elem_fraction=elem_fraction, ion_fraction=ion_fraction
                ),
                units="dimensionless",
                sampling_type="cell",
            )

            # Ion number densities
            ds.add_field(
                ("gas", f"{elem}_{iion:02}_number_density"),
                function=partial(
                    ion_number_density,
                    elem_fraction=elem_fraction,
                    ion_fraction=ion_fraction,
                    atomic_mass=nuclide_data.getStandardAtomicWeight(elem) * u.mp,
                ),
                units="cm**-3",
                sampling_type="cell",
            )
