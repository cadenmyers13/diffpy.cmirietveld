import re
from pathlib import Path

import numpy as np
import pytest
from pyobjcryst.crystal import create_crystal_from_cif
from pyobjcryst.powderpattern import PowderPattern, ReflectionProfileType

from diffpy.cmirietveld.rietveldgenerator import RietveldGenerator
from diffpy.srfit.exceptions import SrFitError
from diffpy.srfit.fitbase import Profile
from diffpy.srfit.structure import struToParameterSet
from diffpy.srfit.structure.objcrystparset import ObjCrystCrystalParSet

PBSO4_CIF = str(Path(__file__).parent / "testdata" / "PbSO4-Wyckoff.cif")


def compute_pyobjcryst_pattern(crystal, tth, wavelength):
    """Compute a PbSO4 pattern directly with pyobjcryst, as a
    reference."""
    pattern = PowderPattern()
    pattern.SetWavelength(wavelength)
    pattern.SetPowderPatternX(np.radians(tth))
    pattern.SetPowderPatternObs(np.zeros_like(tth))
    diffraction = pattern.AddPowderPatternDiffraction(crystal)
    diffraction.SetReflectionProfilePar(
        ReflectionProfileType.PROFILE_PSEUDO_VOIGT
    )
    diffraction.GetProfile().GetPar("W").SetHumanValue(0.01)
    reference_pattern = np.array(pattern.GetPowderPatternCalc())
    return reference_pattern


def test_parameters_before_structure():
    # Test which managed Parameters exist before a structure is set
    # C1: A generator is created and no structure is set.
    # Expected: There are no managed Parameters yet, as in srfit's
    # PDFGenerator.
    generator = RietveldGenerator("bragg")
    expected_names = []
    actual_names = generator.get_names()
    assert actual_names == expected_names


def test_parameter_names(pbso4_crystal):
    # Test which managed Parameters the generator creates
    # C1: A PbSO4 structure is set on the generator.
    # Expected: Creates the scale, the instrument Parameters, and the
    # pseudo-Voigt profile Parameters including the asymmetry terms.
    generator = RietveldGenerator("bragg")
    generator.set_structure(pbso4_crystal)
    expected_names = [
        "scale",
        "zero",
        "displacement",
        "transparency",
        "U",
        "V",
        "W",
        "eta0",
        "eta1",
        "asym0",
        "asym1",
        "asym2",
    ]
    actual_names = generator.get_names()
    assert actual_names == expected_names


def test_set_structure(pbso4_crystal):
    # Test setting the structure from a pyobjcryst Crystal
    # C1: A PbSO4 Crystal is set as the structure.
    # Expected: The generator manages an ObjCrystCrystalParSet named phase
    # that adapts that Crystal.
    generator = RietveldGenerator("bragg")
    generator.set_structure(pbso4_crystal)
    actual_phase_type = type(generator.phase)
    expected_phase_type = ObjCrystCrystalParSet
    assert actual_phase_type == expected_phase_type
    actual_phase_name = generator.phase.name
    expected_phase_name = "phase"
    assert actual_phase_name == expected_phase_name
    actual_structure = generator.phase.stru
    expected_structure = pbso4_crystal
    assert actual_structure == expected_structure


def test_set_structure_from_parset(pbso4_crystal):
    # Test setting the structure from an existing ParameterSet, so it can be
    # shared with other generators
    # C1: A ParameterSet made with struToParameterSet is set as the
    # structure.
    # Expected: The generator manages that same ParameterSet.
    phase = struToParameterSet("PbSO4", pbso4_crystal)
    generator = RietveldGenerator("bragg")
    generator.set_structure_from_parset(phase)
    expected_phase = phase
    actual_phase = generator.phase
    actual_managed_phase = generator.get("PbSO4")
    assert actual_phase == expected_phase
    assert actual_managed_phase == expected_phase


def test_call(pbso4_generator, pbso4_tth):
    # Test calculating the pattern by calling the generator
    # C1: The generator is called on a 2theta grid with Cu Kalpha1 x-rays.
    # Expected: Returns the pattern pyobjcryst calculates directly for the
    # same structure, wavelength and peak width.
    expected_pattern = compute_pyobjcryst_pattern(
        create_crystal_from_cif(PBSO4_CIF), pbso4_tth, 1.5406
    )
    actual_pattern = pbso4_generator(pbso4_tth)
    assert np.allclose(actual_pattern, expected_pattern, rtol=1e-6)


def test_call_with_xray_tube(pbso4_generator, pbso4_tth):
    # Test calculating the pattern with an x-ray tube's Kalpha doublet
    # C1: The wavelength is set to a Cu tube with a Kalpha2/Kalpha1 ratio of
    # 0.5.
    # Expected: Returns the pattern pyobjcryst calculates directly for a Cu
    # tube with the same ratio.
    pbso4_generator.set_xray_tube("Cu", kalpha2_ratio=0.5)
    pattern = PowderPattern()
    pattern.SetWavelength("Cu", 0.5)
    pattern.SetPowderPatternX(np.radians(pbso4_tth))
    pattern.SetPowderPatternObs(np.zeros_like(pbso4_tth))
    diffraction = pattern.AddPowderPatternDiffraction(
        create_crystal_from_cif(PBSO4_CIF)
    )
    diffraction.SetReflectionProfilePar(
        ReflectionProfileType.PROFILE_PSEUDO_VOIGT
    )
    diffraction.GetProfile().GetPar("W").SetHumanValue(0.01)
    expected_pattern = np.array(pattern.GetPowderPatternCalc())
    actual_pattern = pbso4_generator(pbso4_tth)
    assert np.allclose(actual_pattern, expected_pattern, rtol=1e-6)


def test_value_after_scale_change(pbso4_generator, pbso4_tth):
    # Test that the cached pattern is recomputed after the scale changes
    # C1: The pattern is computed, then the scale is doubled.
    # Expected: The new pattern is twice the first one.
    profile = Profile()
    profile.set_calculation_points(pbso4_tth)
    pbso4_generator.set_profile(profile)
    initial_pattern = pbso4_generator.value
    pbso4_generator.scale.set_value(2.0)
    expected_pattern = 2 * initial_pattern
    actual_pattern = pbso4_generator.value
    assert np.allclose(actual_pattern, expected_pattern, rtol=1e-6)


def test_value_after_lattice_change(pbso4_generator, pbso4_tth):
    # Test that the cached pattern is recomputed after a structure
    # Parameter changes
    # C1: The pattern is computed, then the lattice parameter a is set to
    # 8.50 angstroms.
    # Expected: The first pattern is the one pyobjcryst calculates directly
    # with the CIF lattice, and the new pattern is the one it calculates
    # with a = 8.50 angstroms.
    profile = Profile()
    profile.set_calculation_points(pbso4_tth)
    pbso4_generator.set_profile(profile)
    expected_initial_pattern = compute_pyobjcryst_pattern(
        create_crystal_from_cif(PBSO4_CIF), pbso4_tth, 1.5406
    )
    actual_initial_pattern = pbso4_generator.value
    assert np.allclose(
        actual_initial_pattern, expected_initial_pattern, rtol=1e-6
    )
    pbso4_generator.phase.a.set_value(8.50)
    reference_crystal = create_crystal_from_cif(PBSO4_CIF)
    reference_crystal.GetPar("a").SetValue(8.50)
    expected_pattern = compute_pyobjcryst_pattern(
        reference_crystal, pbso4_tth, 1.5406
    )
    actual_pattern = pbso4_generator.value
    assert np.allclose(actual_pattern, expected_pattern, rtol=1e-6)


@pytest.mark.parametrize(
    "input_parameter_name, input_value, expected_internal_value",
    [
        # Test that Parameters are in degrees while pyobjcryst stores
        # radians
        # C1: The zero shift is set to 0.1 degrees.
        # Expected: pyobjcryst stores 0.1 degrees in radians.
        ("zero", 0.1, np.radians(0.1)),
        # C2: The Caglioti W is set to 0.01 degrees squared.
        # Expected: pyobjcryst stores 0.01 degrees squared in radians
        # squared.
        ("W", 0.01, 0.01 * np.radians(1.0) ** 2),
    ],
)
def test_parameter_units(
    pbso4_generator, input_parameter_name, input_value, expected_internal_value
):
    parameter = pbso4_generator.get(input_parameter_name)
    parameter.set_value(input_value)
    actual_internal_value = parameter.obj.GetValue()
    assert actual_internal_value == pytest.approx(expected_internal_value)


@pytest.mark.parametrize(
    "input_radiation_type",
    [
        # Test setting the radiation type
        # C1: The radiation type is set to x-rays.
        # Expected: get_radiation_type returns "X".
        "X",
        # C2: The radiation type is set to neutrons.
        # Expected: get_radiation_type returns "N".
        "N",
        # C3: The radiation type is set to electrons.
        # Expected: get_radiation_type returns "E".
        "E",
    ],
)
def test_set_radiation_type(pbso4_generator, input_radiation_type):
    pbso4_generator.set_radiation_type(input_radiation_type)
    expected_radiation_type = input_radiation_type
    actual_radiation_type = pbso4_generator.get_radiation_type()
    assert actual_radiation_type == expected_radiation_type


def test_set_wavelength(pbso4_generator):
    # Test setting a single wavelength
    # C1: The wavelength is set to 1.909 angstroms.
    # Expected: get_wavelength returns 1.909.
    pbso4_generator.set_wavelength(1.909)
    expected_wavelength = 1.909
    actual_wavelength = pbso4_generator.get_wavelength()
    assert actual_wavelength == pytest.approx(expected_wavelength)


def test_set_radiation_type_unknown(pbso4_generator):
    # Test setting a radiation type that is not supported
    # C1: The radiation type is set to "Q".
    # Expected: Raises ValueError naming the value and listing the valid
    # ones.
    expected_msg = (
        "'Q' is not a supported radiation type. Please use 'X' for "
        "x-rays, 'N' for neutrons or 'E' for electrons."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        pbso4_generator.set_radiation_type("Q")


def test_validate_without_structure():
    # Test validating a generator that has no structure
    # C1: No structure has been set.
    # Expected: Raises SrFitError saying the structure is missing and how
    # to set it.
    generator = RietveldGenerator("bragg")
    expected_msg = (
        "The generator has no structure. Please set one with "
        "set_structure or set_structure_from_parset."
    )
    with pytest.raises(SrFitError, match=re.escape(expected_msg)):
        generator._validate()


def test_validate_without_profile(pbso4_generator):
    # Test validating a generator that has a structure but no Profile
    # C1: The structure is set but no Profile has been given.
    # Expected: Raises SrFitError saying the profile is missing.
    expected_msg = "profile is None"
    with pytest.raises(SrFitError, match=re.escape(expected_msg)):
        pbso4_generator._validate()
