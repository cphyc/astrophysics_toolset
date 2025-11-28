import numpy as np
from . import atomic_data as ad
from .atomic_data import c, h, k
import yt
from .emission_lines import nuclide_data
import roman
from collections import namedtuple


def three_level(
    g,
    lam,
    A,
    z,
    l0_interp,
    l1_interp,
    l2_interp,
    T,
    n_ion,
    ne,
    nH,
    nHp,
    nHe,
    nHep,
    nHepp,
    nH2,
):
    g_0, g_1, g_2 = g  # degeneracy

    lam_10, lam_20, lam_21 = lam  # microns

    A_10, A_20, A_21 = A  #!s^-1

    T_cmb = 2.725 * (1.0 + z)  #! CMB temperature

    # Get log T and enforce bounds
    logT = np.log10(T)
    logT[logT < np.log10(T_cmb)] = np.log10(T_cmb)
    logT[logT > 5.0] = 5.0

    nu_10 = c / (lam_10 * 1.0e-4)  #! Hz
    nu_20 = c / (lam_20 * 1.0e-4)  #! Hz
    nu_21 = c / (lam_21 * 1.0e-4)  #! Hz

    E_10 = (h * c / (lam_10 * 1e-4)) / k  #! E/K (K)
    E_20 = (h * c / (lam_20 * 1e-4)) / k  #! E/K (K)
    E_21 = (h * c / (lam_21 * 1e-4)) / k  #! E/K (K)

    B_01 = A_10 * ((lam_10 * 1.0e-4) ** 3.0) * (g_1 / g_0) / (2.0 * h * c)
    B_02 = A_20 * ((lam_20 * 1.0e-4) ** 3.0) * (g_2 / g_0) / (2.0 * h * c)
    B_12 = A_21 * ((lam_21 * 1.0e-4) ** 3.0) * (g_2 / g_1) / (2.0 * h * c)

    B_10 = (g_0 / g_1) * B_01
    B_20 = (g_0 / g_2) * B_02
    B_21 = (g_1 / g_2) * B_12

    #! CMB black body spectrum
    B_nu_10 = (2.0 * h * (nu_10**3.0) / (c**2.0)) / (
        np.exp(h * nu_10 / (k * T_cmb)) - 1.0
    )
    B_nu_20 = (2.0 * h * (nu_20**3.0) / (c**2.0)) / (
        np.exp(h * nu_20 / (k * T_cmb)) - 1.0
    )
    B_nu_21 = (2.0 * h * (nu_21**3.0) / (c**2.0)) / (
        np.exp(h * nu_21 / (k * T_cmb)) - 1.0
    )

    # ! All collison strengths are in cm^3 s^-1
    # ! For collisions with o-H2
    q10_oH2 = l0_interp["rate_oH2"](logT)
    q20_oH2 = l1_interp["rate_oH2"](logT)
    q21_oH2 = l2_interp["rate_oH2"](logT)

    #! For collisions with p-H2
    q10_pH2 = l0_interp["rate_pH2"](logT)
    q20_pH2 = l1_interp["rate_pH2"](logT)
    q21_pH2 = l2_interp["rate_pH2"](logT)

    # ! For collisions with H
    q10_H = l0_interp["rate_HI"](logT)
    q20_H = l1_interp["rate_HI"](logT)
    q21_H = l2_interp["rate_HI"](logT)

    # ! For collisions with H+
    q10_Hp = l0_interp["rate_HII"](logT)
    q20_Hp = l1_interp["rate_HII"](logT)
    q21_Hp = l2_interp["rate_HII"](logT)

    # ! For collisions with e
    q10_e = l0_interp["rate_electron"](logT)
    q20_e = l1_interp["rate_electron"](logT)
    q21_e = l2_interp["rate_electron"](logT)

    # ! For collisions with He
    q10_He = l0_interp["rate_HeI"](logT)
    q20_He = l1_interp["rate_HeI"](logT)
    q21_He = l2_interp["rate_HeI"](logT)

    # ! For collisions with He++
    q10_Hep = l0_interp["rate_HeII"](logT)
    q20_Hep = l1_interp["rate_HeII"](logT)
    q21_Hep = l2_interp["rate_HeII"](logT)

    # ! For collisions with He++
    q10_Hepp = l0_interp["rate_HeIII"](logT)
    q20_Hepp = l1_interp["rate_HeIII"](logT)
    q21_Hepp = l2_interp["rate_HeIII"](logT)

    # ! Net collision strengths
    C_10 = (
        (q10_e * ne)
        + (q10_H * nH)
        + (q10_Hp * nHp)
        + (q10_oH2 * 0.75 * nH2)
        + (q10_pH2 * 0.25 * nH2)
        + (q10_He * nHe)
        + (q10_Hep * nHep)
        + (q10_Hepp * nHepp)
    )
    C_20 = (
        (q20_e * ne)
        + (q20_H * nH)
        + (q20_Hp * nHp)
        + (q20_oH2 * 0.75 * nH2)
        + (q20_pH2 * 0.25 * nH2)
        + (q20_He * nHe)
        + (q20_Hep * nHep)
        + (q20_Hepp * nHepp)
    )
    C_21 = (
        (q21_e * ne)
        + (q21_H * nH)
        + (q21_Hp * nHp)
        + (q21_oH2 * 0.75 * nH2)
        + (q21_pH2 * 0.25 * nH2)
        + (q21_He * nHe)
        + (q21_Hep * nHep)
        + (q21_Hepp * nHepp)
    )

    C_01 = C_10 * (g_1 / g_0) * np.exp(-1.0 * E_10 / T)
    C_02 = C_20 * (g_2 / g_0) * np.exp(-1.0 * E_20 / T)
    C_12 = C_21 * (g_2 / g_1) * np.exp(-1.0 * E_21 / T)

    C_01 += B_01 * B_nu_10
    C_02 += B_02 * B_nu_10
    C_12 += B_12 * B_nu_10

    C_10 += B_10 * B_nu_10
    C_20 += B_20 * B_nu_10
    C_21 += B_21 * B_nu_10

    #! Analytic solution to the 2 level system (see paul goldsmith papers)
    #! Done this way to avoid numerical errors
    n2_n1_top = (C_12 * (C_01 + C_02)) + (C_02 * (A_10 + C_10))
    n2_n1_bot = ((A_21 + C_21 + C_20) * (C_01 + C_02)) - (C_20 * C_02)
    n2_n1 = n2_n1_top / n2_n1_bot

    n1_n0_top = ((A_21 + C_21 + C_20) * (C_01 + C_02)) - (C_20 * C_02)
    n1_n0_bot = ((A_21 + C_21 + C_20) * (A_10 + C_10)) + (C_20 * C_12)
    n1_n0 = n1_n0_top / n1_n0_bot

    n_0 = 1.0 / (1.0 + n1_n0 + (n2_n1 * n1_n0))
    n_1 = 1.0 / (1.0 + n2_n1 + (1.0 / n1_n0))
    n_2 = 1.0 / (1.0 + (1.0 / n2_n1) + ((1.0 / n2_n1) * (1.0 / n1_n0)))

    # ! Cooling and heating rates
    cool_0 = (A_10 + (B_10 * B_nu_10)) * E_10 * k * n_1 * n_ion
    cool_1 = (A_20 + (B_20 * B_nu_20)) * E_20 * k * n_2 * n_ion
    cool_2 = (A_21 + (B_21 * B_nu_21)) * E_21 * k * n_2 * n_ion

    heat_0 = B_01 * B_nu_10 * E_10 * k * n_0 * n_ion
    heat_1 = B_02 * B_nu_20 * E_20 * k * n_0 * n_ion
    heat_2 = B_12 * B_nu_21 * E_21 * k * n_1 * n_ion

    # ! Total cooling rate
    return [cool_0 - heat_0, cool_1 - heat_1, cool_2 - heat_2]


def two_level(g, lam, A, z, l0_interp, T, n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2):
    g_0, g_1 = g

    lam_10 = lam  # microns

    A_10 = A  # s^-1

    T_cmb = 2.725 * (1.0 + z)  # CMB temperature

    # Get log T and enforce bounds
    logT = np.log10(T)
    logT[logT < np.log10(T_cmb)] = np.log10(T_cmb)
    logT[logT > 5.0] = 5.0

    nu_10 = c / (lam_10 * 1.0e-4)  # Hz

    E_10 = (h * c / (lam_10 * 1e-4)) / k  # E/K (K)

    B_01 = A_10 * ((lam_10 * 1.0e-4) ** 3.0) * (g_1 / g_0) / (2.0 * h * c)

    B_10 = (g_0 / g_1) * B_01

    # CMB black body spectrum
    B_nu_10 = (2.0 * h * (nu_10**3.0) / (c**2.0)) / (
        np.exp(h * nu_10 / (k * T_cmb)) - 1.0
    )

    # All collison strengths are in cm^3 s^-1
    q10_oH2 = l0_interp["rate_oH2"](logT)

    #! For collisions with p-H2
    q10_pH2 = l0_interp["rate_pH2"](logT)

    # ! For collisions with H
    q10_H = l0_interp["rate_HI"](logT)

    # ! For collisions with H+
    q10_Hp = l0_interp["rate_HII"](logT)

    # ! For collisions with e
    q10_e = l0_interp["rate_electron"](logT)

    # ! For collisions with He
    q10_He = l0_interp["rate_HeI"](logT)

    # ! For collisions with He++
    q10_Hep = l0_interp["rate_HeII"](logT)

    # ! For collisions with He++
    q10_Hepp = l0_interp["rate_HeIII"](logT)

    #! Net collision strengths
    C_10 = C_10 = (
        (q10_e * ne)
        + (q10_H * nH)
        + (q10_Hp * nHp)
        + (q10_oH2 * 0.75 * nH2)
        + (q10_pH2 * 0.25 * nH2)
        + (q10_He * nHe)
        + (q10_Hep * nHep)
        + (q10_Hepp * nHepp)
    )

    C_01 = C_10 * (g_1 / g_0) * np.exp(-1.0 * E_10 / T)

    #! Analytic solution to the 2 level system (modified version of paul
    # goldsmith papers). Done this way to avoid numerical errors
    t1 = (B_01 * B_nu_10) + C_01
    t2 = A_10 + (B_10 * B_nu_10) + C_10

    nu_over_nl = t1 / t2
    n_0 = (nu_over_nl + 1.0) ** -1.0
    n_1 = (nu_over_nl**-1.0 + 1.0) ** -1.0

    #! Cooling and heating rates
    cool_0 = (A_10 + (B_10 * B_nu_10)) * E_10 * k * n_1 * n_ion

    heat_0 = B_01 * B_nu_10 * E_10 * k * n_0 * n_ion

    #! Total cooling rate
    return cool_0 - heat_0


"""
Lines from lise's paper
[O i]λλ63,145μm ==> done
[Si ii]λ34μm (8.2eV) ==> done
[C ii]λ158μm (11.3eV) ==> done
Huαλ12μm (13.6eV),
[N ii]λλ122,205μm (14.5eV), ==> done
[Ne ii]λ12μm (21.6eV), ==> done
[S iii]λλ18,33μm (23.3eV), ==> done
[N iii]λ57μm (29.6eV), ==> done
[S iv]λ10μm (34.7eV), ==> done
[O iii]λ88μm (35.1eV), ==> done
[Ne iii]λ15μm (40.9eV) ==> done
[O iv]λ26μm (54.9eV), ==> done
[Ne v]λλ14,24μm (97.1eV), ==> done
"""


def O_I_fine_structure(T, n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2, z):
    # ! Three level ion
    g = [5.0, 3.0, 1.0]

    lam = [63.1679, 44.0453, 145.495]  #!microns

    A = [8.910e-05, 1.340e-10, 1.750e-05]  #!s^-1

    lum_63, lum_44, lum_145 = three_level(
        g,
        lam,
        A,
        z,
        ad.OI_63_interp_dict,
        ad.OI_44_interp_dict,
        ad.OI_145_interp_dict,
        T,
        n_ion,
        ne,
        nH,
        nHp,
        nHe,
        nHep,
        nHepp,
        nH2,
    )

    return lum_63, lum_44, lum_145


def C_I_fine_structure(T, n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2, z):
    # ! Three level ion
    g = [1.0, 3.0, 5.0]

    lam = [609.590, 230.352, 370.269]  #!microns

    A = [7.930e-08, 1.000e-30, 2.650e-07]  #!s^-1

    lum_609, lum_230, lum_370 = three_level(
        g,
        lam,
        A,
        z,
        ad.CI_609_interp_dict,
        ad.CI_230_interp_dict,
        ad.CI_370_interp_dict,
        T,
        n_ion,
        ne,
        nH,
        nHp,
        nHe,
        nHep,
        nHepp,
        nH2,
    )

    return lum_609, lum_230, lum_370


def C_II_fine_structure(T, n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2, z):
    g = [2.0, 4.0]

    lam = 157.636  # microns

    A = 2.290e-06  # s^-1

    lum_157 = two_level(
        g,
        lam,
        A,
        z,
        ad.CII_158_interp_dict,
        T,
        n_ion,
        ne,
        nH,
        nHp,
        nHe,
        nHep,
        nHepp,
        nH2,
    )

    return lum_157


def N_II_fine_structure(T, n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2, z):
    # ! Three level ion
    g = [1.0, 3.0, 5.0]

    lam = [205.244, 76.4318, 121.767]  #!microns

    A = [2.080e-06, 1.000e-30, 7.460e-06]  #!s^-1

    lum_205, lum_76, lum_122 = three_level(
        g,
        lam,
        A,
        z,
        ad.NII_205_interp_dict,
        ad.NII_76_interp_dict,
        ad.NII_122_interp_dict,
        T,
        n_ion,
        ne,
        nH,
        nHp,
        nHe,
        nHep,
        nHepp,
        nH2,
    )

    return lum_205, lum_76, lum_122


def Si_II_fine_structure(T, n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2, z):
    g = [2.0, 4.0]

    lam = 34.8046  # microns

    A = 2.131e-04  # s^-1

    lum_35 = two_level(
        g,
        lam,
        A,
        z,
        ad.SiII_35_interp_dict,
        T,
        n_ion,
        ne,
        nH,
        nHp,
        nHe,
        nHep,
        nHepp,
        nH2,
    )

    return lum_35


def Ne_II_fine_structure(T, n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2, z):
    g = [4.0, 2.0]

    lam = 12.8101  # microns

    A = 8.590e-03  # s^-1

    lum_13 = two_level(
        g,
        lam,
        A,
        z,
        ad.NeII_13_interp_dict,
        T,
        n_ion,
        ne,
        nH,
        nHp,
        nHe,
        nHep,
        nHepp,
        nH2,
    )

    return lum_13


def S_III_fine_structure(T, n_ion, ne):
    t_ne_grid = np.zeros((len(T), 2))
    t_ne_grid[:, 0] = np.log10(T)
    t_ne_grid[:, 1] = np.log10(ne)
    emis_18 = ad.line_dict["S3-18"]["emis_grid"](t_ne_grid)
    emis_33 = ad.line_dict["S3-33"]["emis_grid"](t_ne_grid)

    lum_18 = emis_18 * ne * n_ion
    lum_33 = emis_33 * ne * n_ion

    return lum_18, lum_33


def S_IV_fine_structure(T, n_ion, ne):
    t_ne_grid = np.zeros((len(T), 2))
    t_ne_grid[:, 0] = np.log10(T)
    t_ne_grid[:, 1] = np.log10(ne)
    emis_10 = ad.line_dict["S4-10"]["emis_grid"](t_ne_grid)

    lum_10 = emis_10 * ne * n_ion

    return lum_10


def N_III_fine_structure(T, n_ion, ne):
    t_ne_grid = np.zeros((len(T), 2))
    t_ne_grid[:, 0] = np.log10(T)
    t_ne_grid[:, 1] = np.log10(ne)
    emis_57 = ad.line_dict["N3-57"]["emis_grid"](t_ne_grid)

    lum_57 = emis_57 * ne * n_ion

    return lum_57


def O_III_fine_structure(T, n_ion, ne):
    t_ne_grid = np.zeros((len(T), 2))
    t_ne_grid[:, 0] = np.log10(T)
    t_ne_grid[:, 1] = np.log10(ne)
    emis_88 = ad.line_dict["O3-88"]["emis_grid"](t_ne_grid)
    emis_52 = ad.line_dict["O3-52"]["emis_grid"](t_ne_grid)

    lum_88 = emis_88 * ne * n_ion
    lum_52 = emis_52 * ne * n_ion

    return lum_88, lum_52


def Ne_III_fine_structure(T, n_ion, ne):
    t_ne_grid = np.zeros((len(T), 2))
    t_ne_grid[:, 0] = np.log10(T)
    t_ne_grid[:, 1] = np.log10(ne)
    emis_36 = ad.line_dict["Ne3-36"]["emis_grid"](t_ne_grid)
    emis_15 = ad.line_dict["Ne3-15"]["emis_grid"](t_ne_grid)

    lum_36 = emis_36 * ne * n_ion
    lum_15 = emis_15 * ne * n_ion

    return lum_36, lum_15


def O_IV_fine_structure(T, n_ion, ne):
    t_ne_grid = np.zeros((len(T), 2))
    t_ne_grid[:, 0] = np.log10(T)
    t_ne_grid[:, 1] = np.log10(ne)
    emis_26 = ad.line_dict["O4-26"]["emis_grid"](t_ne_grid)

    lum_26 = emis_26 * ne * n_ion

    return lum_26


def Ne_V_fine_structure(T, n_ion, ne):
    t_ne_grid = np.zeros((len(T), 2))
    t_ne_grid[:, 0] = np.log10(T)
    t_ne_grid[:, 1] = np.log10(ne)
    emis_14 = ad.line_dict["Ne5-14"]["emis_grid"](t_ne_grid)
    emis_24 = ad.line_dict["Ne5-24"]["emis_grid"](t_ne_grid)

    lum_14 = emis_14 * ne * n_ion
    lum_24 = emis_24 * ne * n_ion

    return lum_14, lum_24


def H_I_IR_red(T, n_ion, ne):
    t_ne_grid = np.zeros((len(T), 2))
    t_ne_grid[:, 0] = np.log10(T)
    t_ne_grid[:, 1] = np.log10(ne)
    emis_12 = ad.line_dict["Hua"]["emis_grid"](t_ne_grid)

    lum_12 = emis_12 * ne * n_ion

    return lum_12


def _create_element_number_density(ds, element_name, element_short):
    # print(f"Creating {element_name=} ({element_short=}) number density field")
    atomic_weight = nuclide_data.getStandardAtomicWeight(element_name)

    def number_density(field, data):
        rho = data["ramses", f"hydro_{element_name}_fraction"] * data["gas", "density"]
        w = data.apply_units(atomic_weight, "amu")
        return rho / w

    nelem_field_name = ("gas", f"{element_short}_number_density")
    ds.add_field(
        nelem_field_name,
        function=number_density,
        units="1/cm**3",
        sampling_type="cell",
    )
    return nelem_field_name


def _create_ion_number_density(ds, element_name, element_short, ion_level):
    # print(f"Creating {element_name=} {ion_level=} ({element_short=}) ion number density field")
    def number_density(field, data):
        return (
            data["gas", f"{element_short}_number_density"]
            * data["ramses", f"hydro_{element_name}_{ion_level:02d}"]
        )

    iX = roman.toRoman(ion_level)
    nion_field_name = ("gas", f"{element_short}_{iX}_number_density")
    ds.add_field(
        nion_field_name,
        function=number_density,
        units="1/cm**3",
        sampling_type="cell",
    )
    return nion_field_name


IonDesc = namedtuple("IonDesc", ["nion", "ion_level"])


def create_emission_lines(ds):
    # Create hydrogen number density field
    mH = ds.quan(nuclide_data.getStandardAtomicWeight("H"), "amu")
    mHe = ds.quan(nuclide_data.getStandardAtomicWeight("He"), "amu")
    mFe = ds.quan(nuclide_data.getStandardAtomicWeight("Fe"), "amu")

    # Create H, He, Fe number density fields
    ds.add_field(
        ("gas", "H_number_density"),
        function=lambda field, data: data["gas", "density"] * 0.76 / mH,
        units="1/cm**3",
        sampling_type="cell",
    )
    ds.add_field(
        ("gas", "He_number_density"),
        function=lambda field, data: data["gas", "density"] * 0.24 / mHe,
        units="1/cm**3",
        sampling_type="cell",
    )
    ds.add_field(
        ("gas", "Fe_number_density"),
        function=lambda field, data: data["gas", "density"]
        * data["ramses", "Metallicity"]
        / mFe,
        units="1/cm**3",
        sampling_type="cell",
    )

    # Create missing H2 and He_01
    ds.add_field(
        ("ramses", "hydro_H2"),
        function=lambda field, data: 1
        - data["ramses", "hydro_H_01"]
        - data["ramses", "hydro_H_02"],
        units="1",
        sampling_type="cell",
    )
    ds.add_field(
        ("ramses", "hydro_He_01"),
        function=lambda field, data: np.clip(
            1 - data["ramses", "hydro_He_02"] - data["ramses", "hydro_He_03"],
            0,
            1,
        ),
        units="1",
        sampling_type="cell",
    )

    # Create nH2 field
    ds.add_field(
        ("gas", "H2_number_density"),
        function=lambda field, data: data["gas", "H_number_density"]
        * np.clip(
            1 - data["ramses", "hydro_H_01"] - data["ramses", "hydro_H_02"],
            0,
            1,
        ),
        units="1/cm**3",
        sampling_type="cell",
    )

    # Alias H_XX and He_XX fields
    ds.field_info.alias(
        ("ramses", "hydro_hydrogen_01"),
        ("ramses", "hydro_H_01"),
        units="1",
    )
    ds.field_info.alias(
        ("ramses", "hydro_hydrogen_02"),
        ("ramses", "hydro_H_02"),
        units="1",
    )
    ds.field_info.alias(
        ("ramses", "hydro_helium_01"),
        ("ramses", "hydro_He_01"),
        units="1",
    )
    ds.field_info.alias(
        ("ramses", "hydro_helium_02"),
        ("ramses", "hydro_He_02"),
        units="1",
    )
    ds.field_info.alias(
        ("ramses", "hydro_helium_03"),
        ("ramses", "hydro_He_03"),
        units="1",
    )

    element_list = (
        ("hydrogen", "H"),
        ("helium", "He"),
        ("oxygen", "O"),
        ("carbon", "C"),
        ("nitrogen", "N"),
        ("silicon", "Si"),
        ("sulfur", "S"),
        ("neon", "Ne"),
        ("iron", "Fe"),
    )

    # Create other element number density fields + ion number density fields
    ion_fields: list[IonDesc] = []
    for element, element_short in element_list:
        # Skip for H, He and Fe (already done)
        if element not in ("hydrogen", "helium", "iron"):
            _create_element_number_density(ds, element, element_short)

        # Create ion number density fields
        for i in range(1, 20):
            ion_field = ("ramses", f"hydro_{element}_{i:02d}")
            # print(f"\t\t{ion_field}")

            if ion_field not in ds.field_info:
                continue

            nion_field_name = _create_ion_number_density(ds, element, element_short, i)

            ion_fields.append(
                IonDesc(
                    nion=nion_field_name,
                    ion_level=i,
                )
            )

    # Create electron number density field and mean molecular weight field
    def electron_number_density(field, data):
        ret = None
        for tmp in ion_fields:
            i = tmp.ion_level
            if i > 1:
                if ret is None:
                    ret = data[tmp.nion] * (i - 1)
                else:
                    ret += data[tmp.nion] * (i - 1)

        return ret

    def mean_molecular_weight(field, data):
        one_amu = data.apply_units(1, "amu")
        ntot = data["gas", "electron_number_density"]
        for _element, element_short in element_list:
            ntot += data["gas", f"{element_short}_number_density"]

        return data["gas", "density"] / (ntot * one_amu)

    ds.add_field(
        ("gas", "electron_number_density"),
        function=electron_number_density,
        units="1/cm**3",
        sampling_type="cell",
    )
    ds.add_field(
        ("gas", "mean_molecular_weight"),
        function=mean_molecular_weight,
        units="1",
        sampling_type="cell",
        force_override=True,
    )

    def wrap_long(func, nion_field, axis=None):
        def wrapped(field, data):
            input_shape = data["gas", "temperature"].shape

            T = data["gas", "temperature"].to("K").value
            nion = data[nion_field].to("1/cm**3").value
            ne = data["gas", "electron_number_density"].to("1/cm**3").value
            nH = data["gas", "H_number_density"].to("1/cm**3").value
            nHII = data["gas", "H_II_number_density"].to("1/cm**3").value
            nHe = data["gas", "He_number_density"].to("1/cm**3").value
            nHeII = data["gas", "He_II_number_density"].to("1/cm**3").value
            nHeIII = data["gas", "He_III_number_density"].to("1/cm**3").value
            nH2 = data["gas", "H2_number_density"].to("1/cm**3").value
            z = data.ds.current_redshift
            if isinstance(data, yt.fields.field_detector.FieldDetector):
                return np.random.rand(*input_shape)

            ret = func(T, nion, ne, nH, nHII, nHe, nHeII, nHeIII, nH2, z)
            if isinstance(ret, tuple) and axis is None:
                if len(ret[0]) == 0:
                    ret = np.array([])
                else:
                    ret = np.sum(ret, axis=0)
            elif isinstance(ret, tuple) and axis is not None:
                ret = ret[axis]
            elif axis is not None:
                raise ValueError("axis is not None but ret is not a tuple")

            return data.apply_units(ret, "erg/s/cm**3")

        return wrapped

    def wrap_short(func, nion_field, axis=None):
        def wrapped(field, data):
            input_shape = data["gas", "temperature"].shape

            T = data["gas", "temperature"].to("K").value
            nion = data["gas", nion_field].to("1/cm**3").value
            ne = data["gas", "electron_number_density"].to("1/cm**3").value
            if isinstance(data, yt.fields.field_detector.FieldDetector):
                return np.random.rand(*input_shape)

            ret = func(T, nion, ne)
            if isinstance(ret, tuple) and axis is None:
                if len(ret[0]) == 0:
                    ret = np.array([])
                else:
                    ret = np.sum(ret, axis=0)
            elif isinstance(ret, tuple) and axis is not None:
                ret = ret[axis]
            elif axis is not None:
                raise ValueError("axis is not None but ret is not a tuple")
            return data.apply_units(ret, "erg/s/cm**3")

        return wrapped

    all_fields = []

    for field_name, func in (
        # OI fine structure lines
        ("O_I_63µm", wrap_long(O_I_fine_structure, "O_I_number_density", 0)),
        ("O_I_44µm", wrap_long(O_I_fine_structure, "O_I_number_density", 1)),
        ("O_I_145µm", wrap_long(O_I_fine_structure, "O_I_number_density", 2)),
        ("O_I_63_44_145µm", wrap_long(O_I_fine_structure, "O_I_number_density")),
        # CI fine structure lines
        ("C_I_609µm", wrap_long(C_I_fine_structure, "C_I_number_density", 0)),
        ("C_I_230µm", wrap_long(C_I_fine_structure, "C_I_number_density", 1)),
        ("C_I_370µm", wrap_long(C_I_fine_structure, "C_I_number_density", 2)),
        ("C_I_609_230_370µm", wrap_long(C_I_fine_structure, "C_I_number_density")),
        # CII fine structure line
        ("C_II_158µm", wrap_long(C_II_fine_structure, "C_II_number_density")),
        # NII fine structure lines
        ("N_II_205µm", wrap_long(N_II_fine_structure, "N_II_number_density", 0)),
        ("N_II_76µm", wrap_long(N_II_fine_structure, "N_II_number_density", 1)),
        ("N_II_122µm", wrap_long(N_II_fine_structure, "N_II_number_density", 2)),
        ("N_II_205_76_122µm", wrap_long(N_II_fine_structure, "N_II_number_density")),
        # SiII fine structure line
        ("Si_II_35µm", wrap_long(Si_II_fine_structure, "Si_II_number_density")),
        # NeII fine structure line
        ("Ne_II_13µm", wrap_long(Ne_II_fine_structure, "Ne_II_number_density")),
        # SIII fine structure lines
        ("S_III_18µm", wrap_short(S_III_fine_structure, "S_III_number_density", 0)),
        ("S_III_33µm", wrap_short(S_III_fine_structure, "S_III_number_density", 1)),
        ("S_III_18_33µm", wrap_short(S_III_fine_structure, "S_III_number_density")),
        # SIV fine structure line
        ("S_IV_10µm", wrap_short(S_IV_fine_structure, "S_IV_number_density")),
        # NIII fine structure line
        ("N_III_57µm", wrap_short(N_III_fine_structure, "N_III_number_density")),
        # OIII fine structure line
        ("O_III_88µm", wrap_short(O_III_fine_structure, "O_III_number_density", 0)),
        ("O_III_52µm", wrap_short(O_III_fine_structure, "O_III_number_density", 1)),
        ("O_III_88_52µm", wrap_short(O_III_fine_structure, "O_III_number_density")),
        # NeIII fine structure lines
        ("Ne_III_36µm", wrap_short(Ne_III_fine_structure, "Ne_III_number_density", 0)),
        ("Ne_III_15µm", wrap_short(Ne_III_fine_structure, "Ne_III_number_density", 1)),
        ("Ne_III_36_15µm", wrap_short(Ne_III_fine_structure, "Ne_III_number_density")),
        # OIV fine structure line
        ("O_IV_26µm", wrap_short(O_IV_fine_structure, "O_IV_number_density")),
        # NeV fine structure lines
        ("Ne_V_14µm", wrap_short(Ne_V_fine_structure, "Ne_V_number_density", 0)),
        ("Ne_V_24µm", wrap_short(Ne_V_fine_structure, "Ne_V_number_density", 1)),
        ("Ne_V_14_24µm", wrap_short(Ne_V_fine_structure, "Ne_V_number_density")),
        # Halpha line
        ("Hua_12µm", wrap_short(H_I_IR_red, "H_I_number_density")),
    ):
        ds.add_field(
            ("gas", field_name),
            function=func,
            units="erg/s/cm**3",
            sampling_type="cell",
        )
        all_fields.append(("gas", field_name))

    return all_fields
