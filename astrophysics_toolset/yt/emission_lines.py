from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import unyt as u
from yt.fields.field_detector import FieldDetector
from yt.utilities.on_demand_imports import NotAModule


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
                    mass_number,
                    relative_atomic_mass,
                    isotopic_composition,
                    standard_atomic_weight,
                    notes,
                )

    def getStandardAtomicWeight(self, symbol: str) -> float:
        for isotope in self.values():
            if isotope.symbol == symbol:
                return isotope.standard_atomic_weight
        raise ValueError(f"Symbol {symbol} not found in NIST data")


nuclide_data = NISTNuclideData()


def create_emission_line(
    ds, element: str, ionization_level: int, wavelength: Optional[float] = None
):
    # If electron number density is not found, create it from
    if ("gas", "electron_number_density") not in ds.derived_field_list:
        H_mass = nuclide_data.getStandardAtomicWeight("H") * u.mp
        He_mass = nuclide_data.getStandardAtomicWeight("He") * u.mp

        def electron_number_density(field, data):
            # Compute hydrogen and helium number density
            Z = data["gas", "metallicity"]
            nH = data["gas", "density"] * (1 - Z) * 0.74 / H_mass
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
        V = data["gas", "cell_volume"].in_units("cm**3")
        if isinstance(data, FieldDetector):
            eps = data.apply_units(1, "erg/s*cm**3")
        else:
            eps = data.apply_units(
                atom.getEmissivity(T.flatten(), ne.flatten(), wave=wavelength).reshape(
                    T.shape
                ),
                "erg/s*cm**3",
            )
        return ne * nX * eps * V

    ds.add_field(
        ("gas", f"{element}_{wavelength}A_emissivity"),
        function=emissivity,
        units="erg/s",
        sampling_type="cell",
    )