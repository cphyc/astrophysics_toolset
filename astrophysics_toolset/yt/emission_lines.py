from dataclasses import dataclass
from functools import partial
from itertools import product
from math import floor
from pathlib import Path
from typing import Optional, Union

import numpy as np
import unyt as u
from scipy.interpolate import RegularGridInterpolator
from yt.fields.field_detector import FieldDetector
from yt.utilities.on_demand_imports import NotAModule

symbol2name = {
    "H": "Hydrogen",
    "D": "Deuterium",
    "T": "Tritium",
    "He": "Helium",
    "Li": "Lithium",
    "Be": "Beryllium",
    "B": "Boron",
    "C": "Carbon",
    "N": "Nitrogen",
    "O": "Oxygen",
    "F": "Fluorine",
    "Ne": "Neon",
    "Na": "Sodium",
    "Mg": "Magnesium",
    "Al": "Aluminium",
    "Si": "Silicon",
    "P": "Phosphorus",
    "S": "Sulfur",
    "Cl": "Chlorine",
    "Ar": "Argon",
    "K": "Potassium",
    "Ca": "Calcium",
    "Sc": "Scandium",
    "Ti": "Titanium",
    "V": "Vanadium",
    "Cr": "Chromium",
    "Mn": "Manganese",
    "Fe": "Iron",
    "Co": "Cobalt",
    "Ni": "Nickel",
    "Cu": "Copper",
    "Zn": "Zinc",
    "Ga": "Gallium",
    "Ge": "Germanium",
    "As": "Arsenic",
    "Se": "Selenium",
    "Br": "Bromine",
    "Kr": "Krypton",
    "Rb": "Rubidium",
    "Sr": "Strontium",
    "Y": "Yttrium",
    "Zr": "Zirconium",
    "Nb": "Niobium",
    "Mo": "Molybdenum",
    "Tc": "Technetium",
    "Ru": "Ruthenium",
    "Rh": "Rhodium",
    "Pd": "Palladium",
    "Ag": "Silver",
    "Cd": "Cadmium",
    "In": "Indium",
    "Sn": "Tin",
    "Sb": "Antimony",
    "Te": "Tellurium",
    "I": "Iodine",
    "Xe": "Xenon",
    "Cs": "Cesium",
    "Ba": "Barium",
    "La": "Lanthanum",
    "Ce": "Cerium",
    "Pr": "Praseodymium",
    "Nd": "Neodymium",
    "Pm": "Promethium",
    "Sm": "Samarium",
    "Eu": "Europium",
    "Gd": "Gadolinium",
    "Tb": "Terbium",
    "Dy": "Dysprosium",
    "Ho": "Holmium",
    "Er": "Erbium",
    "Tm": "Thulium",
    "Yb": "Ytterbium",
    "Lu": "Lutetium",
    "Hf": "Hafnium",
    "Ta": "Tantalum",
    "W": "Tungsten",
    "Re": "Rhenium",
    "Os": "Osmium",
    "Ir": "Iridium",
    "Pt": "Platinum",
    "Au": "Gold",
    "Hg": "Mercury",
    "Tl": "Thallium",
    "Pb": "Lead",
    "Bi": "Bismuth",
    "Po": "Polonium",
    "At": "Astatine",
    "Rn": "Radon",
    "Fr": "Francium",
    "Ra": "Radium",
    "Ac": "Actinium",
    "Th": "Thorium",
    "Pa": "Protactinium",
    "U": "Uranium",
    "Np": "Neptunium",
    "Pu": "Plutonium",
    "Am": "Americium",
    "Cm": "Curium",
    "Bk": "Berkelium",
    "Cf": "Californium",
    "Es": "Einsteinium",
    "Fm": "Fermium",
    "Md": "Mendelevium",
    "No": "Nobelium",
    "Lr": "Lawrencium",
    "Rf": "Rutherfordium",
    "Db": "Dubnium",
    "Sg": "Seaborgium",
    "Bh": "Bohrium",
    "Hs": "Hassium",
    "Mt": "Meitnerium",
    "Ds": "Darmstadtium",
    "Rg": "Roentgenium",
    "Cn": "Copernicium",
    "Nh": "Nihonium",
    "Fl": "Flerovium",
    "Mc": "Moscovium",
    "Lv": "Livermorium",
    "Ts": "Tennessine",
    "Og": "Oganesson",
    "Uut": "Nihonium",
    "Uuq": "Flerovium",
    "Uup": "Moscovium",
    "Uuh": "Livermorium",
    "Uus": "Tennessine",
    "Uuo": "Oganesson",
}

name2symbol = {v: k for k, v in symbol2name.items()}


class PynebImports:
    _name = "pyneb"
    _module = None

    def __init__(self):
        try:
            import pyneb

            self._module = pyneb
        except ImportError:
            self._module = NotAModule(self._name)

    def __getattr__(self, attr):
        return getattr(self._module, attr)


pyneb = PynebImports()


@dataclass
class Isotope:
    atomic_number: int
    symbol: str
    name: str
    mass_number: int
    relative_atomic_mass: Union[float, None]
    isotopic_composition: Union[float, None]
    standard_atomic_weight: float
    notes: str


class NISTNuclideData(dict):
    def __init__(self):
        with (Path(__file__).parent / "nist-nuclide-data.txt").open() as f:
            while True:
                _an = f.readline()
                if not _an:
                    break
                atomic_number = int(_an.split("=")[1])
                symbol = f.readline().split("=")[1].strip()
                mass_number = int(f.readline().split("=")[1])
                _raw = f.readline().split("=")[1].strip()
                if _raw:
                    relative_atomic_mass = float(_raw.split("(")[0])
                else:
                    relative_atomic_mass = None
                _composition = f.readline().split("=")[1].strip()
                if _composition:
                    isotopic_composition = float(_composition.split("(")[0])
                else:
                    isotopic_composition = None
                _saw = f.readline().split("=")[1].strip()
                if "[" in _saw:
                    standard_atomic_weight = float(
                        _saw.replace("[", "").replace("]", "")
                    )
                else:
                    standard_atomic_weight = float(_saw.split("(")[0])
                notes = f.readline().split("=")[1].strip()
                f.readline()

                self[f"{symbol}{mass_number}"] = Isotope(
                    atomic_number,
                    symbol,
                    symbol2name[symbol],
                    mass_number,
                    relative_atomic_mass,
                    isotopic_composition,
                    standard_atomic_weight,
                    notes,
                )

    def getStandardAtomicWeight(self, symbolOrName: str) -> float:
        for isotope in self.values():
            if (
                isotope.symbol == symbolOrName
                or isotope.name.lower() == symbolOrName.lower()
            ):
                return isotope.standard_atomic_weight
        raise ValueError(f"Symbol or name {symbolOrName} not found in NIST data")


nuclide_data = NISTNuclideData()


n_vals = 400
temperatures = np.logspace(1, 9, n_vals)
e_densities = np.logspace(-7, 6, n_vals)


def format_wavelength(wavelength: float) -> tuple[str, str]:
    "Formats a wavelength in Å into mm/µm/Å"
    if wavelength > 1e8:
        wl = f"{wavelength/1e7:.0f}"
        wlu = "mm"
    elif wavelength > 1e5:
        wl = f"{wavelength/1e4:.0f}"
        wlu = "µm"
    else:
        wl = f"{wavelength:.0f}"
        wlu = "Å"

    return wl, wlu



def _create_transition_from_wavelength(
    ds, atom: "pyneb.Atom", wavelength: float
) -> list[tuple[str, str]]:
    if int(wavelength) == wavelength:
        wavelength = int(wavelength)
    element = atom.elem
    ionization_level = atom.spec
    name_mapping = {
        "Ca": "calcium",
        "C": "carbon",
        "Mg": "magnesium",
        "Ne": "neon",
        "N": "nitrogen",
        "O": "oxygen",
        "Si": "silicon",
        "S": "sulfur",
        "Fe": "iron",
    }
    element_full_name = name_mapping[element]

    xMass = nuclide_data.getStandardAtomicWeight(element) * u.mp

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
    }[ionization_level]

    solutions = np.argwhere(np.abs(wavelength - atom.lineList) < 2).flatten()

    if len(solutions) == 0 or len(solutions) > 2:
        raise ValueError(f"No line transition found for wavelength {wavelength} Å.")
    elif len(solutions) == 2:
        decimals = -floor(np.log10(np.min(np.abs(np.diff(atom.lineList[solutions])))))
        lines = [
            _create_transition_from_wavelength(
                ds, atom, np.round(atom.lineList[ind], decimals=decimals)
            )[0]
            for ind in solutions
        ]

        def line_ratio(field, data, l1, l2):
            return data[l1] / data[l2]

        def line_sum(field, data, l1, l2):
            return data[l1] + data[l2]

        all_lines = lines.copy()

        for l1, l2 in product(lines, repeat=2):
            if l1 == l2:
                continue

            wl, wu = format_wavelength(wavelength)

            # Add doublet emissivity
            field_name: tuple[str, str] = (
                "gas",
                f"{element}{ionization_level}_{wl}{wu}_doublet_emissivity",
            )

            ds.add_field(
                field_name,
                function=partial(line_sum, l1=l1, l2=l2),
                units="erg/s/cm**3",
                sampling_type="cell",
                display_name=(
                    rf"[{element}{roman}]$\lambda\lambda{wl}${wu} Emissivity"
                ),
            )
            all_lines.append(field_name)

            # Add line ratio
            field_name: tuple[str, str] = (
                "gas",
                f"{element}{ionization_level}_{wl}{wu}_doublet_ratio",
            )
            ds.add_field(
                field_name,
                function=partial(line_ratio, l1=l1, l2=l2),
                units="",
                sampling_type="cell",
                display_name=(
                    rf"[{element}{roman}]$\lambda\lambda{wl}${wu} Line Ratio"
                ),
            )
            all_lines.append(field_name)

            return all_lines

    em_grid = atom.getEmissivity(
        tem=temperatures, den=e_densities, wave=wavelength, product=True
    )
    emissivity = RegularGridInterpolator(
        (np.log10(temperatures), np.log10(e_densities)),
        em_grid,
        bounds_error=False,
        fill_value=0,
    )

    def ion_emissivity(field, data):
        T = data["gas", "temperature"].to("K")
        ne = data["gas", "electron_number_density"].to("cm**-3")
        nX = (
            data["gas", "density"]
            * data["ramses", f"hydro_{element_full_name}_fraction"]
            * data["ramses", f"hydro_{element_full_name}_{ionization_level:02}"]
            / xMass
        )

        if isinstance(data, FieldDetector):
            eps = data.apply_units(1, "erg/s*cm**3")
        else:
            eps = data.apply_units(
                emissivity(
                    (T.flatten(), ne.flatten()),
                ).reshape(T.shape),
                "erg/s*cm**3",
            )
        return ne * nX * eps

    wl, wu = format_wavelength(wavelength)
    field_name = (
        "gas",
        f"{element}{ionization_level}_{wl}{wu}_emissivity",
    )

    ds.add_field(
        field_name,
        function=ion_emissivity,
        units="erg/s/cm**3",
        sampling_type="cell",
        display_name=rf"[{element}{roman}]$\lambda{wl}${wu} Emissivity",
    )

    return [field_name]


def _create_hydrogen_emission(
    ds, atom: "pyneb.Atom", *, lev_i: int, lev_j: int, name: str, name_latex: str
) -> tuple[str, str]:
    HMass = nuclide_data.getStandardAtomicWeight("H") * u.mp

    em_grid = atom.getEmissivity(
        tem=temperatures, den=e_densities, lev_i=lev_i, lev_j=lev_j, product=True
    )
    emissivity = RegularGridInterpolator(
        (np.log10(temperatures), np.log10(e_densities)),
        em_grid,
        bounds_error=False,
        fill_value=0,
    )

    def H_emissivity(field, data):
        T = data["gas", "temperature"].to("K")
        ne = data["gas", "electron_number_density"].to("cm**-3")
        Z = data["gas", "metallicity"]
        nH = (
            data["gas", "density"]
            * data["ramses", "hydro_H_01"]
            * (1 - Z)
            * 0.76
            / HMass
        )

        if isinstance(data, FieldDetector):
            eps = data.apply_units(1, "erg/s*cm**3")
        else:
            eps = data.apply_units(
                emissivity(
                    (T.flatten(), ne.flatten()),
                ).reshape(T.shape),
                "erg/s*cm**3",
            )
        return ne * nH * eps

    field_name = ("gas", f"{name}_emissivity")
    ds.add_field(
        field_name,
        function=H_emissivity,
        units="erg/s/cm**3",
        sampling_type="cell",
        display_name=f"{name_latex} Emissivity",
    )
    return field_name


def create_emission_line(
    ds, element: str, ionization_level: int, wavelength: Optional[float] = None
) -> list[tuple[str, str]]:
    # If electron number density is not found, create it from
    if ("gas", "electron_number_density") not in ds.derived_field_list:
        H_mass = nuclide_data.getStandardAtomicWeight("H") * u.mp
        He_mass = nuclide_data.getStandardAtomicWeight("He") * u.mp

        def electron_number_density(field, data):
            # Compute hydrogen and helium number density
            Z = data["gas", "metallicity"]
            nH = data["gas", "density"] * (1 - Z) * 0.76 / H_mass
            nHe = data["gas", "density"] * (1 - Z) * 0.24 / He_mass

            # Ionized fractions
            xHII = data["ramses", "hydro_H_02"]  # H+
            xHeII = data["ramses", "hydro_He_02"]  # He+
            xHeIII = data["ramses", "hydro_He_03"]  # He2+

            # This should really take into account metals, but let's ignore that for now
            return nH * xHII + nHe * (xHeII + 2 * xHeIII)

        ds.add_field(
            ("gas", "electron_number_density"),
            function=electron_number_density,
            units="cm**-3",
            sampling_type="cell",
        )

    if element not in ("Halpha", "Hbeta", "Hgamma"):
        atom = pyneb.Atom(element, ionization_level)
        return _create_transition_from_wavelength(ds, atom, wavelength)
    elif element == "Halpha":
        atom = pyneb.RecAtom("H", 1)
        lev_i = 3
        lev_j = 2
        name = "Halpha"
        name_latex = r"${\rm H}\alpha$"
    elif element == "Hbeta":
        atom = pyneb.RecAtom("H", 1)
        lev_i = 4
        lev_j = 2
        name = "Hbeta"
        name_latex = r"${\rm H}\beta$"
    elif element == "Hgamma":
        atom = pyneb.RecAtom("H", 1)
        lev_i = 5
        lev_j = 2
        name = "Hgamma"
        name_latex = r"${\rm H}\gamma$"

    return [
        _create_hydrogen_emission(
            ds, atom, lev_i=lev_i, lev_j=lev_j, name=name, name_latex=name_latex
        )
    ]
