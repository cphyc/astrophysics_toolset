from functools import partial

import unyt as u

from astrophysics_toolset.yt.emission_lines import name2symbol, nuclide_data

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
    "sulfur",
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
        elem_symbol = name2symbol.get(elem.capitalize(), elem.capitalize())
        if elem_fraction not in ds.field_list:
            continue

        # Element-by-element densities
        ds.add_field(
            ("gas", f"{elem}_density"),
            function=partial(element_density, elem_fraction=elem_fraction),
            units="g/cm**3",
            sampling_type="cell",
            display_name=f"{elem_symbol} Density",
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
            display_name=f"{elem_symbol} Number Density",
        )

        for iion in range(20):
            ion_fraction = ("ramses", f"hydro_{elem}_{iion:02}")
            if ion_fraction not in ds.field_list:
                continue

            # Convert integer to roman numeral
            roman = {
                1: "I",
                2: "II",
                3: "III",
                4: "IV",
                5: "V",
                6: "VI",
                7: "VII",
                8: "VIII",
                9: "IX",
                10: "X",
                11: "XI",
                12: "XII",
                13: "XIII",
                14: "XIV",
                15: "XV",
                16: "XVI",
                17: "XVII",
                18: "XVIII",
                19: "XIX",
            }[iion]

            # Ion fractions
            ds.add_field(
                ("gas", f"{elem}_{iion:02}_density"),
                function=partial(
                    ion_density, elem_fraction=elem_fraction, ion_fraction=ion_fraction
                ),
                units="g/cm**3",
                sampling_type="cell",
                display_name=f"{elem_symbol}{roman} Number Density",
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
                display_name=f"{elem_symbol}{roman} Number Density",
            )
