from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import unyt as u
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
}


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
            if isotope.symbol == symbolOrName or isotope.name == symbolOrName:
                return isotope.standard_atomic_weight
        raise ValueError(f"Symbol or name {symbolOrName} not found in NIST data")


nuclide_data = NISTNuclideData()


def create_emission_line(
    ds, element: str, ionization_level: int, wavelength: Optional[float] = None
) -> tuple[str, str]:
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

    atom = pyneb.Atom(element, ionization_level)

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

    def emissivity(field, data):
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
                atom.getEmissivity(
                    T.flatten(),
                    ne.flatten(),
                    wave=wavelength,
                    product=False,
                ).reshape(T.shape),
                "erg/s*cm**3",
            )
        return ne * nX * eps

    field_name = (
        "gas",
        f"{element}{ionization_level}_{wavelength}A_emissivity",
    )

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
    ds.add_field(
        field_name,
        function=emissivity,
        units="erg/s/cm**3",
        sampling_type="cell",
        display_name=rf"[{element}{roman}]$\lambda{wavelength:.0f}Å$ emissivity",
    )
    return field_name
