import re
import warnings

import numpy as np
import pytest
from conftest import PBSO4_CIF, TESTDATA_DIR, ExternalPeakProfile
from pyobjcryst.crystal import Crystal, create_crystal_from_cif
from pyobjcryst.reflectionprofile import (
    ReflectionProfilePseudoVoigtAnisotropic,
    ReflectionProfilePseudoVoigtTCH,
)

from diffpy.cmirietveld.instruments import ConstantWavelengthInstrument
from diffpy.cmirietveld.peak_profiles import PseudoVoigtProfile, TCHProfile
from diffpy.cmirietveld.powderdiffractiongenerator import (
    PowderDiffractionGenerator,
)
from diffpy.cmistructure import structure_to_parameter_set
from diffpy.cmistructure.objcrystparset import ObjCrystCrystalParSet
from diffpy.srfit.exceptions import SrFitError
from diffpy.srfit.fitbase import FitContribution, FitRecipe, Profile
from diffpy.structure import load_structure


def test_init():
    # Test a new generator before anything is set
    # Expected: It is named bragg, has no structure, uses a pseudo-Voigt
    # peak profile and a constant-wavelength instrument, and its scale is 1.
    generator = PowderDiffractionGenerator()
    actual_name = generator.name
    actual_phase = generator.phase
    actual_peak_profile_type = type(generator.peak_profile)
    actual_instrument_type = type(generator.instrument)
    actual_scale = generator.scale.value
    expected_name = "bragg"
    expected_phase = None
    expected_peak_profile_type = PseudoVoigtProfile
    expected_instrument_type = ConstantWavelengthInstrument
    expected_scale = 1.0
    assert actual_name == expected_name
    assert actual_phase == expected_phase
    assert actual_peak_profile_type == expected_peak_profile_type
    assert actual_instrument_type == expected_instrument_type
    assert actual_scale == expected_scale


@pytest.mark.parametrize(
    "set_structure_kwargs, expected_phase_name",
    [
        # Test setting the structure from a pyobjcryst Crystal
        # C1: A user sets the PbSO4 crystal without naming it.
        # Expected: The structure's Parameters are under generator.phase.
        ({}, "phase"),
        # C2: A user sets the PbSO4 crystal and names it pbso4.
        # Expected: The structure's Parameters are under generator.pbso4.
        ({"name": "pbso4"}, "pbso4"),
    ],
)
def test_set_structure_with_crystal(set_structure_kwargs, expected_phase_name):
    crystal = create_crystal_from_cif(str(PBSO4_CIF))
    generator = PowderDiffractionGenerator()
    generator.set_structure(crystal, **set_structure_kwargs)
    actual_phase_type = type(generator.phase)
    actual_phase_name = generator.phase.name
    actual_managed_phase = generator.get(expected_phase_name)
    actual_structure = generator.phase.structure
    expected_phase_type = ObjCrystCrystalParSet
    expected_managed_phase = generator.phase
    expected_structure = crystal
    assert actual_phase_type == expected_phase_type
    assert actual_phase_name == expected_phase_name
    assert actual_managed_phase == expected_managed_phase
    assert actual_structure == expected_structure


@pytest.mark.parametrize(
    "cif_path",
    [
        # Test setting the structure from the path to a CIF file
        # C1: A user gives the path to the PbSO4 CIF as a pathlib.Path.
        # Expected: The generator reads the file into a pyobjcryst Crystal,
        # and the pattern equals plain pyobjcryst's for that CIF.
        PBSO4_CIF,
        # C2: A user gives the same path as a string.
        # Expected: The same Crystal type and pattern as for the Path.
        str(PBSO4_CIF),
    ],
)
def test_set_structure_with_cif_path(pbso4_pattern, cif_path):
    generator = PowderDiffractionGenerator()
    generator.set_structure(cif_path)
    generator.instrument.set_wavelength(1.5406)
    actual_structure_type = type(generator.phase.structure)
    actual_pattern = generator(pbso4_pattern.tth)
    expected_structure_type = Crystal
    expected_pattern = pbso4_pattern.calculate()
    assert actual_structure_type == expected_structure_type
    assert np.allclose(actual_pattern, expected_pattern)


def test_call_with_cif_path_given_to_two_generators(pbso4_pattern):
    # Test giving the same CIF path to two generators
    # Expected: Each generator reads its own structure, so after the lattice
    # parameter a of one is set to 8.50 angstroms, the other's pattern is
    # still plain pyobjcryst's for the CIF. Sharing one structure needs one
    # ParameterSet given to both generators.
    # Give two generators the same path
    generator = PowderDiffractionGenerator()
    other_generator = PowderDiffractionGenerator()
    generator.set_structure(PBSO4_CIF)
    other_generator.set_structure(PBSO4_CIF)
    generator.instrument.set_wavelength(1.5406)
    other_generator.instrument.set_wavelength(1.5406)
    # Change a on the first generator only
    generator.phase.a.set_value(8.50)
    actual_other_pattern = other_generator(pbso4_pattern.tth)
    expected_other_pattern = pbso4_pattern.calculate()
    assert np.allclose(actual_other_pattern, expected_other_pattern)


@pytest.mark.parametrize(
    "file_path",
    [
        # Test setting the structure from a file that isn't a CIF file
        # holding a crystal structure
        # C1: A user gives the path to their data file by mistake.
        # Expected: A ValueError names the file and says to give a CIF file
        # with a crystal structure.
        TESTDATA_DIR / "PBSO4.XRA",
        # C2: A user gives the .stru file from their PDF fits, which
        # diffpy.cmipdf's PDF generators read.
        # Expected: The same ValueError, since the generator reads CIF
        # files only.
        TESTDATA_DIR / "pbso4.stru",
    ],
)
def test_set_structure_with_non_cif_file(file_path):
    generator = PowderDiffractionGenerator()
    expected_msg = (
        f"No crystal structure could be read from {file_path}. "
        "Please give a CIF file that holds a crystal structure."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        generator.set_structure(file_path)


@pytest.mark.parametrize(
    "phase_name, set_structure_kwargs",
    [
        # Test setting the structure from an existing ParameterSet, as a
        # user does to share one structure between datasets
        # C1: A user makes a ParameterSet named phase from the PbSO4 crystal
        # and gives it to set_structure.
        # Expected: The generator manages that same ParameterSet, under
        # generator.phase.
        ("phase", {}),
        # C2: A user makes the ParameterSet with the name pbso4.
        # Expected: The ParameterSet keeps its name, so it is under
        # generator.pbso4, and generator.phase is it too.
        ("pbso4", {}),
        # C3: A user makes the ParameterSet with the name pbso4 and gives
        # set_structure the same name.
        # Expected: The same as without the name, since the names match.
        ("pbso4", {"name": "pbso4"}),
    ],
)
def test_set_structure_with_parameter_set(phase_name, set_structure_kwargs):
    phase = structure_to_parameter_set(
        phase_name, create_crystal_from_cif(str(PBSO4_CIF))
    )
    generator = PowderDiffractionGenerator()
    generator.set_structure(phase, **set_structure_kwargs)
    actual_phase = generator.phase
    actual_managed_phase = generator.get(phase_name)
    expected_phase = phase
    assert actual_phase == expected_phase
    assert actual_managed_phase == expected_phase


def test_set_structure_with_parameter_set_and_other_name():
    # Test giving set_structure a name with a ParameterSet that has another
    # name of its own
    # Expected: A ValueError says the name can't be used and how to name
    # the ParameterSet instead.
    phase = structure_to_parameter_set(
        "phase", create_crystal_from_cif(str(PBSO4_CIF))
    )
    generator = PowderDiffractionGenerator()
    expected_msg = (
        "set_structure was given the name 'pbso4' with a ParameterSet, "
        "which keeps its own name, 'phase'. Please leave out the name, or "
        "give it when you create the ParameterSet."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        generator.set_structure(phase, name="pbso4")


def test_set_structure_with_diffpy_structure():
    # Test giving set_structure the diffpy.structure Structure a user has
    # from their PDF fits, which ObjCryst can't calculate with
    # Expected: A ValueError names the type and lists what to give instead.
    diffpy_structure = load_structure(str(PBSO4_CIF))
    generator = PowderDiffractionGenerator()
    expected_msg = (
        "set_structure was given a Structure, which the powder diffraction "
        "generator can't calculate with. Please give a pyobjcryst Crystal, "
        "the path to a CIF file, or a ParameterSet made from a Crystal."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        generator.set_structure(diffpy_structure)


def test_set_structure_from_parset():
    # Test setting the structure with set_structure_from_parset, which
    # srfit's and diffpy.cmipdf's PDF generators also have
    # Expected: The generator manages that same ParameterSet.
    phase = structure_to_parameter_set(
        "phase", create_crystal_from_cif(str(PBSO4_CIF))
    )
    generator = PowderDiffractionGenerator()
    generator.set_structure_from_parset(phase)
    actual_phase = generator.phase
    expected_phase = phase
    assert actual_phase == expected_phase


def test_call_after_set_structure_twice(pbso4_pattern):
    # Test setting a second structure, as when a user reruns the setup cell
    # of their script with another CIF
    # Expected: The generator holds only the second structure, and the
    # pattern equals plain pyobjcryst's for it alone, not the sum of both.
    # Set a first structure, with a changed lattice, and calculate once
    generator = PowderDiffractionGenerator()
    first_crystal = create_crystal_from_cif(str(PBSO4_CIF))
    first_crystal.GetPar("a").SetHumanValue(8.50)
    generator.set_structure(first_crystal)
    generator.instrument.set_wavelength(1.5406)
    generator(pbso4_pattern.tth)
    # Replace it with the CIF's structure and calculate again
    second_crystal = create_crystal_from_cif(str(PBSO4_CIF))
    generator.set_structure(second_crystal)
    actual_structure = generator.phase.structure
    actual_pattern = generator(pbso4_pattern.tth)
    expected_structure = second_crystal
    expected_pattern = pbso4_pattern.calculate()
    assert actual_structure == expected_structure
    assert np.allclose(actual_pattern, expected_pattern)


def test_get_after_set_structure_with_new_name(pbso4_generator):
    # Test setting a second structure under a new name, as when a user
    # renames their phase and reruns the setup cell of their script
    # Expected: The first structure, named phase, is no longer under the
    # generator, and the second one is under generator.pbso4.
    pbso4_generator.set_structure(
        create_crystal_from_cif(str(PBSO4_CIF)), name="pbso4"
    )
    actual_old_phase = pbso4_generator.get("phase")
    actual_new_phase = pbso4_generator.get("pbso4")
    expected_old_phase = None
    expected_new_phase = pbso4_generator.phase
    assert actual_old_phase == expected_old_phase
    assert actual_new_phase == expected_new_phase


def test_call_with_default_components(pbso4_generator, pbso4_pattern):
    # Test calculating a pattern with the default peak profile and
    # instrument
    # Expected: The pattern equals the one plain pyobjcryst calculates for
    # the same structure, grid and wavelength.
    actual_pattern = pbso4_generator(pbso4_pattern.tth)
    expected_pattern = pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


def test_call_after_set_wavelength(pbso4_generator, pbso4_pattern):
    # Test that a wavelength set after a calculation is used in the next one
    # Expected: After a user sets a single wavelength of 1.909 angstroms,
    # the pattern equals plain pyobjcryst's at 1.909 angstroms.
    # Calculate once, so the new wavelength has to be installed again
    pbso4_generator(pbso4_pattern.tth)
    # Set the new wavelength and calculate again
    pbso4_generator.instrument.set_wavelength(1.909)
    actual_pattern = pbso4_generator(pbso4_pattern.tth)
    # Set the same wavelength directly in pyobjcryst
    pbso4_pattern.pattern.SetWavelength(1.909)
    expected_pattern = pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


def test_call_after_set_xray_source(pbso4_generator, pbso4_pattern):
    # Test that an x-ray tube set after a calculation is used in the next one
    # Expected: After a user sets a Cu tube, the pattern equals plain
    # pyobjcryst's for a Cu tube with a Kalpha2/Kalpha1 ratio of 0.5.
    # Calculate once, so the tube has to be installed again
    pbso4_generator(pbso4_pattern.tth)
    # Set the Cu tube and calculate again
    pbso4_generator.instrument.set_xray_source("Cu")
    actual_pattern = pbso4_generator(pbso4_pattern.tth)
    # Set the same tube directly in pyobjcryst
    pbso4_pattern.pattern.SetWavelength("Cu", 0.5)
    expected_pattern = pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


@pytest.mark.parametrize(
    "peak_profile_class, pyobjcryst_class, parameter_name, new_value",
    [
        # Test that the generator calculates with the peak profile it is
        # given
        # C1: A user switches to a TCH profile and sets its Lorentzian
        # width X.
        # Expected: The pattern equals plain pyobjcryst's with a TCH profile
        # and the same X.
        (TCHProfile, ReflectionProfilePseudoVoigtTCH, "X", 0.02),
        # C2: A user writes their own peak profile in another package, here
        # one for pyobjcryst's anisotropic pseudo-Voigt, and sets its G_HH.
        # Expected: The pattern equals plain pyobjcryst's with that profile
        # and the same G_HH, so a new profile needs no change to the
        # generator.
        (
            ExternalPeakProfile,
            ReflectionProfilePseudoVoigtAnisotropic,
            "G_HH",
            0.0005,
        ),
    ],
)
def test_call_after_set_peak_profile(
    pbso4_generator,
    pbso4_pattern,
    peak_profile_class,
    pyobjcryst_class,
    parameter_name,
    new_value,
):
    # Give the generator the new profile and change one of its Parameters
    pbso4_generator.set_peak_profile(peak_profile_class())
    pbso4_generator.peak_profile.get(parameter_name).set_value(new_value)
    actual_pattern = pbso4_generator(pbso4_pattern.tth)
    # Install the same profile and value directly in pyobjcryst
    pbso4_pattern.diffraction.SetProfile(pyobjcryst_class())
    installed_profile = pbso4_pattern.diffraction.GetProfile()
    installed_profile.GetPar(parameter_name).SetHumanValue(new_value)
    expected_pattern = pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


def test_call_after_set_peak_profile_before_set_structure(pbso4_pattern):
    # Test setting up the peak profile before the structure, as a user may
    # do at the top of a script
    # Expected: The pattern equals plain pyobjcryst's with a TCH profile,
    # because the profile is installed once the structure is set.
    # Set the profile first, then the structure and the wavelength
    generator = PowderDiffractionGenerator()
    tch_profile = TCHProfile()
    tch_profile.X.set_value(0.02)
    generator.set_peak_profile(tch_profile)
    generator.set_structure(create_crystal_from_cif(str(PBSO4_CIF)))
    generator.instrument.set_wavelength(1.5406)
    actual_pattern = generator(pbso4_pattern.tth)
    # Install the same profile directly in pyobjcryst
    pbso4_pattern.diffraction.SetProfile(ReflectionProfilePseudoVoigtTCH())
    installed_profile = pbso4_pattern.diffraction.GetProfile()
    installed_profile.GetPar("X").SetHumanValue(0.02)
    expected_pattern = pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


@pytest.mark.parametrize(
    "setter_name, component_name, component_class",
    [
        # Test that a setter replaces the component and its Parameters
        # C1: A user switches to a TCH profile.
        # Expected: generator.peak_profile is the TCH profile, and the
        # generator manages it, so a change to its Parameters reaches a
        # FitContribution.
        ("set_peak_profile", "peak_profile", TCHProfile),
        # C2: A user gives the generator an instrument they set up.
        # Expected: generator.instrument is that instrument, and the
        # generator manages it.
        ("set_instrument", "instrument", ConstantWavelengthInstrument),
    ],
)
def test_component_after_setter(
    pbso4_generator, setter_name, component_name, component_class
):
    new_component = component_class()
    getattr(pbso4_generator, setter_name)(new_component)
    actual_component = getattr(pbso4_generator, component_name)
    actual_managed_component = pbso4_generator.get(component_name)
    expected_component = new_component
    assert actual_component == expected_component
    assert actual_managed_component == expected_component


def test_call_after_peak_profile_parameter_change(
    pbso4_generator, pbso4_pattern
):
    # Test that a changed peak profile Parameter is used in the next
    # calculation
    # Expected: After W is set to 0.05, the pattern equals plain
    # pyobjcryst's with W = 0.05.
    # Calculate once, then change W and calculate again
    pbso4_generator(pbso4_pattern.tth)
    pbso4_generator.peak_profile.W.set_value(0.05)
    actual_pattern = pbso4_generator(pbso4_pattern.tth)
    # Set the same W directly in pyobjcryst
    installed_profile = pbso4_pattern.diffraction.GetProfile()
    installed_profile.GetPar("W").SetHumanValue(0.05)
    expected_pattern = pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


def test_call_after_instrument_parameter_change(
    pbso4_generator, pbso4_pattern
):
    # Test that a changed instrument Parameter is used in the next
    # calculation
    # Expected: After zero is set to 0.1 degrees, the pattern equals plain
    # pyobjcryst's with a zero shift of 0.1 degrees.
    # Calculate once, then change zero and calculate again
    pbso4_generator(pbso4_pattern.tth)
    pbso4_generator.instrument.zero.set_value(0.1)
    actual_pattern = pbso4_generator(pbso4_pattern.tth)
    # Set the same zero shift directly in pyobjcryst
    pbso4_pattern.pattern.GetPar("Zero").SetHumanValue(0.1)
    expected_pattern = pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


def test_call_after_wavelength_change(pbso4_generator, pbso4_pattern):
    # Test that a refined wavelength is used in the next calculation
    # Expected: After the wavelength Parameter is set to 1.55 angstroms, as
    # a refinement would, the pattern equals plain pyobjcryst's at 1.55
    # angstroms.
    # Calculate once, then change the wavelength and calculate again
    pbso4_generator(pbso4_pattern.tth)
    pbso4_generator.instrument.wavelength.set_value(1.55)
    actual_pattern = pbso4_generator(pbso4_pattern.tth)
    # Set the same wavelength directly in pyobjcryst
    pbso4_pattern.pattern.SetWavelength(1.55)
    expected_pattern = pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


def test_call_after_structure_parameter_change(pbso4_generator, pbso4_pattern):
    # Test that a changed structure Parameter is used in the next
    # calculation
    # Expected: After the lattice parameter a is set to 8.50 angstroms, the
    # pattern equals plain pyobjcryst's with a = 8.50 angstroms.
    # Calculate once, then change a and calculate again
    pbso4_generator(pbso4_pattern.tth)
    pbso4_generator.phase.a.set_value(8.50)
    actual_pattern = pbso4_generator(pbso4_pattern.tth)
    # Set the same lattice parameter directly in pyobjcryst
    pbso4_pattern.crystal.GetPar("a").SetHumanValue(8.50)
    expected_pattern = pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


def test_call_after_scale_change(pbso4_generator, pbso4_pattern):
    # Test that the scale is applied to the calculated pattern
    # Expected: Doubling the scale doubles the pattern.
    initial_pattern = pbso4_generator(pbso4_pattern.tth)
    pbso4_generator.scale.set_value(2.0)
    actual_pattern = pbso4_generator(pbso4_pattern.tth)
    expected_pattern = 2 * initial_pattern
    assert np.allclose(actual_pattern, expected_pattern)


def test_call_after_grid_change(pbso4_generator, pbso4_pattern):
    # Test calculating on a new 2theta grid, as when a user changes the
    # calculation range
    # Expected: The pattern equals plain pyobjcryst's on the new 20-60
    # degree grid, with one value per new point.
    # Calculate once on the full grid, then on the new one
    pbso4_generator(pbso4_pattern.tth)
    new_tth = np.arange(20.0, 60.0, 0.02)
    actual_pattern = pbso4_generator(new_tth)
    # Calculate on the same grid directly in pyobjcryst
    pbso4_pattern.pattern.SetPowderPatternX(np.radians(new_tth))
    pbso4_pattern.pattern.SetPowderPatternObs(np.zeros_like(new_tth))
    expected_pattern = pbso4_pattern.calculate()
    actual_number_of_points = len(actual_pattern)
    expected_number_of_points = len(new_tth)
    assert actual_number_of_points == expected_number_of_points
    assert np.allclose(actual_pattern, expected_pattern)


def test_call_with_shared_peak_profile(
    pbso4_generator, other_pbso4_generator, pbso4_pattern
):
    # Test one peak profile given to two generators, as a user does for one
    # phase measured in two datasets
    # Expected: After W is set to 0.05, both patterns equal plain
    # pyobjcryst's with W = 0.05.
    # Give both generators the same profile and calculate once
    shared_profile = PseudoVoigtProfile()
    pbso4_generator.set_peak_profile(shared_profile)
    other_pbso4_generator.set_peak_profile(shared_profile)
    pbso4_generator(pbso4_pattern.tth)
    other_pbso4_generator(pbso4_pattern.tth)
    # Change W once and calculate both again
    shared_profile.W.set_value(0.05)
    actual_pattern = pbso4_generator(pbso4_pattern.tth)
    actual_other_pattern = other_pbso4_generator(pbso4_pattern.tth)
    # Set the same W directly in pyobjcryst
    installed_profile = pbso4_pattern.diffraction.GetProfile()
    installed_profile.GetPar("W").SetHumanValue(0.05)
    expected_pattern = pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)
    assert np.allclose(actual_other_pattern, expected_pattern)


def test_call_with_shared_instrument_after_parameter_change(
    pbso4_generator, other_pbso4_generator, pbso4_pattern
):
    # Test one instrument given to two generators, as a user does for two
    # phases in one dataset
    # Expected: After zero is set to 0.1 degrees, both patterns equal plain
    # pyobjcryst's with a zero shift of 0.1 degrees.
    # Give both generators the same instrument and calculate once
    shared_instrument = ConstantWavelengthInstrument()
    shared_instrument.set_wavelength(1.5406)
    pbso4_generator.set_instrument(shared_instrument)
    other_pbso4_generator.set_instrument(shared_instrument)
    pbso4_generator(pbso4_pattern.tth)
    other_pbso4_generator(pbso4_pattern.tth)
    # Change zero once and calculate both again
    shared_instrument.zero.set_value(0.1)
    actual_pattern = pbso4_generator(pbso4_pattern.tth)
    actual_other_pattern = other_pbso4_generator(pbso4_pattern.tth)
    # Set the same zero shift directly in pyobjcryst
    pbso4_pattern.pattern.GetPar("Zero").SetHumanValue(0.1)
    expected_pattern = pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)
    assert np.allclose(actual_other_pattern, expected_pattern)


def test_call_with_shared_instrument_after_setting_change(
    pbso4_generator, other_pbso4_generator, pbso4_pattern
):
    # Test one instrument given to two generators when a setting changes
    # Expected: After a user sets a Cu tube, both patterns equal plain
    # pyobjcryst's for a Cu tube, because both generators install the
    # instrument again.
    # Give both generators the same instrument and calculate once
    shared_instrument = ConstantWavelengthInstrument()
    shared_instrument.set_wavelength(1.5406)
    pbso4_generator.set_instrument(shared_instrument)
    other_pbso4_generator.set_instrument(shared_instrument)
    pbso4_generator(pbso4_pattern.tth)
    other_pbso4_generator(pbso4_pattern.tth)
    # Set a Cu tube once and calculate both again
    shared_instrument.set_xray_source("Cu")
    actual_pattern = pbso4_generator(pbso4_pattern.tth)
    actual_other_pattern = other_pbso4_generator(pbso4_pattern.tth)
    # Set the same tube directly in pyobjcryst
    pbso4_pattern.pattern.SetWavelength("Cu", 0.5)
    expected_pattern = pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)
    assert np.allclose(actual_other_pattern, expected_pattern)


def test_call_with_shared_structure(pbso4_pattern):
    # Test one structure ParameterSet given to two generators, as a user
    # does to refine one structure against two datasets
    # Expected: After the lattice parameter a is set to 8.50 angstroms,
    # both patterns equal plain pyobjcryst's with a = 8.50 angstroms.
    # Give two generators the same structure and calculate once
    shared_phase = structure_to_parameter_set(
        "phase", create_crystal_from_cif(str(PBSO4_CIF))
    )
    generator = PowderDiffractionGenerator()
    other_generator = PowderDiffractionGenerator()
    generator.set_structure(shared_phase)
    other_generator.set_structure(shared_phase)
    generator.instrument.set_wavelength(1.5406)
    other_generator.instrument.set_wavelength(1.5406)
    generator(pbso4_pattern.tth)
    other_generator(pbso4_pattern.tth)
    # Change a once and calculate both again
    shared_phase.a.set_value(8.50)
    actual_pattern = generator(pbso4_pattern.tth)
    actual_other_pattern = other_generator(pbso4_pattern.tth)
    # Set the same lattice parameter directly in pyobjcryst
    pbso4_pattern.crystal.GetPar("a").SetHumanValue(8.50)
    expected_pattern = pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)
    assert np.allclose(actual_other_pattern, expected_pattern)


@pytest.mark.parametrize(
    "component_name, parameter_name, new_value",
    [
        # Test that a FitContribution recalculates after a Parameter
        # changes, as it does at every step of a fit
        # C1: A fit steps the peak profile's W to 0.05.
        # Expected: The contribution's value equals the generator's pattern
        # with W = 0.05, not the cached pattern.
        ("peak_profile", "W", 0.05),
        # C2: A fit steps the instrument's zero to 0.1 degrees.
        # Expected: The contribution's value equals the generator's pattern
        # with a zero shift of 0.1 degrees.
        ("instrument", "zero", 0.1),
        # C3: A fit steps the wavelength to 1.55 angstroms.
        # Expected: The contribution's value equals the generator's pattern
        # at 1.55 angstroms.
        ("instrument", "wavelength", 1.55),
        # C4: A fit steps the lattice parameter a to 8.50 angstroms.
        # Expected: The contribution's value equals the generator's pattern
        # with a = 8.50 angstroms.
        ("phase", "a", 8.50),
    ],
)
def test_evaluate_after_parameter_change(
    pbso4_generator,
    pbso4_contribution,
    pbso4_pattern,
    component_name,
    parameter_name,
    new_value,
):
    # Evaluate the contribution once, then change the Parameter and
    # evaluate again
    pbso4_contribution.evaluate()
    component = getattr(pbso4_generator, component_name)
    component.get(parameter_name).set_value(new_value)
    actual_pattern = pbso4_contribution.evaluate()
    # Calculate with the generator directly, which has no cache
    expected_pattern = pbso4_generator(pbso4_pattern.tth)
    assert np.allclose(actual_pattern, expected_pattern)


def test_evaluate_after_instrument_setting_change(
    pbso4_generator, pbso4_contribution, pbso4_pattern
):
    # Test that a FitContribution recalculates after an instrument setting
    # changes, as it does after a Parameter changes
    # Expected: After a user sets a Cu tube, the contribution's value equals
    # plain pyobjcryst's pattern for a Cu tube, not the cached pattern.
    # Evaluate the contribution once
    pbso4_contribution.evaluate()
    # Set a Cu tube and evaluate again
    pbso4_generator.instrument.set_xray_source("Cu")
    actual_pattern = pbso4_contribution.evaluate()
    # Set the same tube directly in pyobjcryst
    pbso4_pattern.pattern.SetWavelength("Cu", 0.5)
    expected_pattern = pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


def test_evaluate_after_set_peak_profile(
    pbso4_generator, pbso4_contribution, pbso4_pattern
):
    # Test that a FitContribution recalculates after the peak profile is
    # replaced, and that the replaced profile no longer counts
    # Expected: After a user switches to a TCH profile, the contribution's
    # value equals plain pyobjcryst's with a TCH profile, and it stays so
    # after the old profile's W is changed.
    # Evaluate the contribution once
    pbso4_contribution.evaluate()
    # Switch to a TCH profile and evaluate again
    old_profile = pbso4_generator.peak_profile
    pbso4_generator.set_peak_profile(TCHProfile())
    actual_pattern = pbso4_contribution.evaluate()
    # Change the old profile's W and evaluate once more
    old_profile.W.set_value(0.05)
    actual_pattern_after_old_change = pbso4_contribution.evaluate()
    # Install a TCH profile directly in pyobjcryst
    pbso4_pattern.diffraction.SetProfile(ReflectionProfilePseudoVoigtTCH())
    expected_pattern = pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)
    assert np.allclose(actual_pattern_after_old_change, expected_pattern)


def test_peak_profile_after_add_profile_generator(pbso4_generator):
    # Test that a FitContribution doesn't replace the peak profile, since
    # srfit gives each generator a data Profile named profile
    # Expected: generator.peak_profile is still the peak profile, and
    # generator.profile is the contribution's data Profile.
    expected_peak_profile = pbso4_generator.peak_profile
    expected_profile = Profile()
    contribution = FitContribution("xray")
    contribution.set_profile(expected_profile)
    contribution.add_profile_generator(pbso4_generator)
    actual_peak_profile = pbso4_generator.peak_profile
    actual_profile = pbso4_generator.profile
    assert actual_peak_profile == expected_peak_profile
    assert actual_profile == expected_profile


@pytest.mark.parametrize(
    "component_name, parameter_name, new_value, limits, used_value",
    [
        # Test that a value outside ObjCryst's limits stops the calculation,
        # since ObjCryst would silently use another value
        # C1: A fit steps the peak profile's eta0 to 1.5, past its limits of
        # 0 to 1.
        # Expected: A ValueError says ObjCryst would use 1.0, and how to
        # keep the value within the limits.
        ("peak_profile", "eta0", 1.5, "[0.0, 1.0]", "1.0"),
        # C2: A fit steps the instrument's zero to 3 degrees, past its
        # limits of +-0.05 rad.
        # Expected: A ValueError says ObjCryst would use 0.05 rad in
        # degrees, and how to keep the value within the limits.
        (
            "instrument",
            "zero",
            3.0,
            "[-2.8647889756541165, 2.8647889756541165]",
            "2.8647889756541165",
        ),
    ],
)
def test_call_with_value_outside_limits(
    pbso4_generator,
    pbso4_pattern,
    component_name,
    parameter_name,
    new_value,
    limits,
    used_value,
):
    component = getattr(pbso4_generator, component_name)
    component.get(parameter_name).set_value(new_value)
    expected_msg = (
        f"The value of '{parameter_name}', {new_value}, is outside its "
        f"limits {limits}. ObjCryst would silently use {used_value} "
        "instead, so the refined value would not be the one used to "
        "calculate the pattern. Please keep the value within the limits, "
        "for example by passing the recipe's bounds to the optimizer: "
        "least_squares(..., bounds=recipe.get_bounds_array())."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        pbso4_generator(pbso4_pattern.tth)


@pytest.mark.parametrize(
    "setter_name, wrong_component_class, expected_msg",
    [
        # Test that a setter rejects the wrong kind of component
        # C1: A user passes an instrument to set_peak_profile.
        # Expected: A ValueError says it is not a peak profile and names the
        # peak profiles to use.
        (
            "set_peak_profile",
            ConstantWavelengthInstrument,
            "set_peak_profile was given a ConstantWavelengthInstrument, "
            "which is not a peak profile. Please give a PeakProfile, such as "
            "PseudoVoigtProfile or TCHProfile.",
        ),
        # C2: A user passes a peak profile to set_instrument.
        # Expected: A ValueError says it is not an instrument and names the
        # instrument to use.
        (
            "set_instrument",
            PseudoVoigtProfile,
            "set_instrument was given a PseudoVoigtProfile, which is not an "
            "instrument. Please give an Instrument, such as "
            "ConstantWavelengthInstrument.",
        ),
    ],
)
def test_setter_with_wrong_component(
    pbso4_generator, setter_name, wrong_component_class, expected_msg
):
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        getattr(pbso4_generator, setter_name)(wrong_component_class())


def test_call_before_set_structure(pbso4_pattern):
    # Test calculating a pattern before a structure is set
    # Expected: An SrFitError says that the structure is missing and how to
    # set it.
    generator = PowderDiffractionGenerator()
    generator.instrument.set_wavelength(1.5406)
    expected_msg = (
        "The generator has no structure. Please set one with "
        "set_structure or set_structure_from_parset."
    )
    with pytest.raises(SrFitError, match=re.escape(expected_msg)):
        generator(pbso4_pattern.tth)


def test_call_before_set_wavelength(pbso4_pattern):
    # Test calculating a pattern before the instrument has a wavelength
    # Expected: An SrFitError says that the wavelength is missing and how to
    # set it.
    generator = PowderDiffractionGenerator()
    generator.set_structure(create_crystal_from_cif(str(PBSO4_CIF)))
    expected_msg = (
        "The instrument has no wavelength. Please set one with "
        "set_wavelength, set_xray_source or set_xray_wavelengths."
    )
    with pytest.raises(SrFitError, match=re.escape(expected_msg)):
        generator(pbso4_pattern.tth)


@pytest.mark.parametrize(
    "setter_name, component_name, parameter_name, new_component_class, "
    "expected_msg",
    [
        # Test that a setter warns when the component it replaces has
        # Parameters in a fit recipe, since the recipe would refine
        # Parameters that no longer change the pattern
        # C1: A user adds the peak profile's W to their recipe, then
        # switches to a TCH profile.
        # Expected: A UserWarning names W and says to set the profile before
        # adding its Parameters to the recipe.
        (
            "set_peak_profile",
            "peak_profile",
            "W",
            TCHProfile,
            "set_peak_profile replaced a peak profile whose Parameters are "
            "used in a fit recipe: W. The recipe would still refine them, "
            "but they no longer change the pattern. Please call "
            "set_peak_profile before adding the peak profile's Parameters "
            "to the recipe.",
        ),
        # C2: A user adds the instrument's zero to their recipe, then gives
        # the generator an instrument they set up.
        # Expected: A UserWarning names zero and says to set the instrument
        # before adding its Parameters to the recipe.
        (
            "set_instrument",
            "instrument",
            "zero",
            ConstantWavelengthInstrument,
            "set_instrument replaced an instrument whose Parameters are used "
            "in a fit recipe: zero. The recipe would still refine them, but "
            "they no longer change the pattern. Please call set_instrument "
            "before adding the instrument's Parameters to the recipe.",
        ),
    ],
)
def test_setter_warns_recipe_variable_replaced(
    pbso4_generator,
    setter_name,
    component_name,
    parameter_name,
    new_component_class,
    expected_msg,
):
    # Add one of the current component's Parameters to a recipe
    recipe = FitRecipe()
    component = getattr(pbso4_generator, component_name)
    recipe.add_variable(component.get(parameter_name))
    # Replace the component
    with pytest.warns(UserWarning, match=re.escape(expected_msg)):
        getattr(pbso4_generator, setter_name)(new_component_class())


def test_set_peak_profile_warns_constrained_parameter_replaced(
    pbso4_generator,
):
    # Test replacing a peak profile whose Parameter is constrained in a fit
    # recipe, as when a user ties the peak width of two datasets to one
    # recipe variable
    # Expected: The same UserWarning as for a recipe variable, naming W,
    # since the recipe would still set W, but W no longer changes the
    # pattern.
    # Constrain the peak profile's W to a recipe variable
    recipe = FitRecipe()
    recipe.create_new_variable("width", 0.01)
    recipe.constrain(pbso4_generator.peak_profile.W, "width")
    # Switch to a TCH profile
    expected_msg = (
        "set_peak_profile replaced a peak profile whose Parameters are "
        "used in a fit recipe: W. The recipe would still refine them, "
        "but they no longer change the pattern. Please call "
        "set_peak_profile before adding the peak profile's Parameters "
        "to the recipe."
    )
    with pytest.warns(UserWarning, match=re.escape(expected_msg)):
        pbso4_generator.set_peak_profile(TCHProfile())


def test_set_structure_warns_recipe_variable_replaced(pbso4_generator):
    # Test setting a second structure after the first one's Parameters are
    # in a fit recipe, as when a user reruns the setup cell of their script
    # after building their recipe
    # Expected: A UserWarning names a and says to set the structure before
    # adding its Parameters to the recipe.
    # Add the first structure's lattice parameter a to a recipe
    recipe = FitRecipe()
    recipe.add_variable(pbso4_generator.phase.a)
    # Set a second structure
    expected_msg = (
        "set_structure replaced a structure whose Parameters are used in a "
        "fit recipe: a. The recipe would still refine them, but they no "
        "longer change the pattern. Please call set_structure before "
        "adding the structure's Parameters to the recipe."
    )
    with pytest.warns(UserWarning, match=re.escape(expected_msg)):
        pbso4_generator.set_structure(create_crystal_from_cif(str(PBSO4_CIF)))


@pytest.mark.parametrize(
    "setter_name, new_component_class",
    [
        # Test that the setters stay silent when none of the replaced
        # component's Parameters is in a recipe, even with the generator in
        # a FitContribution of a recipe that refines other Parameters
        # C1: A user builds a recipe that refines the scale, then switches
        # to a TCH profile.
        # Expected: No warning.
        ("set_peak_profile", TCHProfile),
        # C2: A user builds a recipe that refines the scale, then gives the
        # generator an instrument they set up.
        # Expected: No warning.
        ("set_instrument", ConstantWavelengthInstrument),
    ],
)
def test_setter_without_recipe_variables(
    pbso4_generator, pbso4_contribution, setter_name, new_component_class
):
    # Put the contribution in a recipe that refines only the scale
    recipe = FitRecipe()
    recipe.add_contribution(pbso4_contribution)
    recipe.add_variable(pbso4_generator.scale)
    # Replace the component, with warnings turned into errors
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        getattr(pbso4_generator, setter_name)(new_component_class())


def test_set_structure_without_recipe_variables(
    pbso4_generator, pbso4_contribution
):
    # Test that set_structure stays silent when none of the replaced
    # structure's Parameters is in a recipe
    # Expected: After a user builds a recipe that refines the scale and
    # then sets a second structure, no warning.
    # Put the contribution in a recipe that refines only the scale
    recipe = FitRecipe()
    recipe.add_contribution(pbso4_contribution)
    recipe.add_variable(pbso4_generator.scale)
    # Set a second structure, with warnings turned into errors
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pbso4_generator.set_structure(create_crystal_from_cif(str(PBSO4_CIF)))
