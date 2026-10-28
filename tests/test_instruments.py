import re

import numpy as np
import pytest
from pyobjcryst.radiation import RadiationType

from diffpy.cmirietveld.instruments import ConstantWavelengthInstrument
from diffpy.srfit.exceptions import SrFitError

# ObjCryst limits the 2theta corrections to +-0.05 rad, in degrees
TWO_THETA_CORRECTION_LIMIT = np.degrees(0.05)
# ObjCryst's tabulated Cu tube wavelengths, in angstroms. ObjCryst
# stores them in single precision, so they aren't 1.540560 and
# 1.544390 exactly
CU_KALPHA1 = 1.540560007095337
CU_KALPHA2 = 1.5443899631500244
CU_MEAN_WAVELENGTH = 1.5418366591135662
# ObjCryst's Cu tube mean wavelength for a Kalpha2/Kalpha1 ratio of 0.4
CU_MEAN_WAVELENGTH_RATIO_0_4 = 1.5416542802538191


def test_parameter_names():
    # Test the name and Parameters of the instrument
    # Expected: It is named instrument and has the 2theta corrections, the
    # absorption and the wavelength.
    instrument = ConstantWavelengthInstrument()
    actual_name = instrument.name
    actual_parameter_names = sorted(instrument.get_names())
    expected_name = "instrument"
    expected_parameter_names = [
        "displacement",
        "mu_r",
        "transparency",
        "wavelength",
        "zero",
    ]
    assert actual_name == expected_name
    assert actual_parameter_names == expected_parameter_names


@pytest.mark.parametrize(
    "parameter_name, expected_value, expected_bounds",
    [
        # Test that the pattern Parameters take ObjCryst's defaults and
        # limits
        # C1: The zero shift is created from the pattern's Zero.
        # Expected: Its value is 0 and its bounds are +-0.05 rad, in degrees.
        (
            "zero",
            0.0,
            [-TWO_THETA_CORRECTION_LIMIT, TWO_THETA_CORRECTION_LIMIT],
        ),
        # C2: The displacement is created from the pattern's 2ThetaDispl.
        # Expected: Its value is 0 and its bounds are +-0.05 rad, in degrees.
        (
            "displacement",
            0.0,
            [-TWO_THETA_CORRECTION_LIMIT, TWO_THETA_CORRECTION_LIMIT],
        ),
        # C3: The transparency is created from the pattern's 2ThetaTransp.
        # Expected: Its value is 0 and its bounds are +-0.05 rad, in degrees.
        (
            "transparency",
            0.0,
            [-TWO_THETA_CORRECTION_LIMIT, TWO_THETA_CORRECTION_LIMIT],
        ),
        # C4: The absorption is created from the pattern's MuR.
        # Expected: Its value is 0 and its bounds are [0, 1000].
        ("mu_r", 0.0, [0.0, 1000.0]),
    ],
)
def test_pattern_parameter_values_and_bounds(
    parameter_name, expected_value, expected_bounds
):
    instrument = ConstantWavelengthInstrument()
    parameter = instrument.get(parameter_name)
    actual_value = parameter.value
    actual_bounds = list(parameter.bounds)
    assert actual_value == pytest.approx(expected_value)
    assert actual_bounds == pytest.approx(expected_bounds)


def test_wavelength_before_set():
    # Test the wavelength of a new instrument
    # Expected: It has no value, because no dataset uses ObjCryst's default
    # of 1 angstrom, and its bounds are the Wavelength limits [0.05, 20].
    instrument = ConstantWavelengthInstrument()
    actual_value = instrument.wavelength.value
    actual_bounds = list(instrument.wavelength.bounds)
    expected_value = None
    expected_bounds = [0.05, 20.0]
    assert actual_value == expected_value
    assert actual_bounds == pytest.approx(expected_bounds)


def test_validate_before_wavelength_set():
    # Test validating an instrument before a wavelength is set, as srfit
    # does before a fit
    # Expected: An SrFitError says that the wavelength is missing and how
    # to set it.
    instrument = ConstantWavelengthInstrument()
    expected_msg = (
        "The instrument has no wavelength. Please set one with "
        "set_wavelength, set_xray_source or set_xray_wavelengths."
    )
    with pytest.raises(SrFitError, match=re.escape(expected_msg)):
        instrument._validate()


@pytest.mark.parametrize(
    "setter_name, setter_args, expected_wavelength",
    [
        # Test that each way of setting the wavelength sets the wavelength
        # Parameter, so the instrument validates
        # C1: A single wavelength is set.
        # Expected: The wavelength is that wavelength.
        ("set_wavelength", [1.5406], 1.5406),
        # C2: A Cu tube is set.
        # Expected: The wavelength is the intensity-weighted mean of
        # ObjCryst's Cu Kalpha1 and Kalpha2, as ObjCryst stores it.
        ("set_xray_source", ["Cu"], CU_MEAN_WAVELENGTH),
        # C3: The Cu Kalpha1 and Kalpha2 wavelengths are set explicitly.
        # Expected: The wavelength is their intensity-weighted mean,
        # Kalpha1 + (Kalpha2 - Kalpha1) * 0.5 / 1.5.
        (
            "set_xray_wavelengths",
            [CU_KALPHA1, CU_KALPHA2],
            CU_MEAN_WAVELENGTH,
        ),
        # C4: A Cu tube is set with a Kalpha2/Kalpha1 ratio of 0.4.
        # Expected: The wavelength is the mean weighted by that ratio,
        # Kalpha1 + (Kalpha2 - Kalpha1) * 0.4 / 1.4, as ObjCryst stores it.
        ("set_xray_source", ["Cu", 0.4], CU_MEAN_WAVELENGTH_RATIO_0_4),
        # C5: The Cu Kalpha1 and Kalpha2 wavelengths are set explicitly
        # with a Kalpha2/Kalpha1 ratio of 0.4.
        # Expected: The wavelength is the mean weighted by that ratio.
        (
            "set_xray_wavelengths",
            [CU_KALPHA1, CU_KALPHA2, 0.4],
            CU_MEAN_WAVELENGTH_RATIO_0_4,
        ),
    ],
)
def test_wavelength_after_setter(
    setter_name, setter_args, expected_wavelength
):
    instrument = ConstantWavelengthInstrument()
    getattr(instrument, setter_name)(*setter_args)
    actual_wavelength = instrument.wavelength.value
    assert actual_wavelength == pytest.approx(expected_wavelength)
    instrument._validate()


@pytest.mark.parametrize(
    "setter_name, setter_args",
    [
        # Test that each setting method is a settings change, so every
        # generator holding the instrument runs install again
        # C1: The radiation type is set.
        # Expected: settings_version goes from 0 to 1.
        ("set_radiation_type", ["N"]),
        # C2: A single wavelength is set, which also sets the wavelength
        # type.
        # Expected: settings_version goes from 0 to 1.
        ("set_wavelength", [1.5406]),
        # C3: A Cu tube is set.
        # Expected: settings_version goes from 0 to 1.
        ("set_xray_source", ["Cu"]),
        # C4: The Kalpha1 and Kalpha2 wavelengths are set explicitly.
        # Expected: settings_version goes from 0 to 1.
        ("set_xray_wavelengths", [CU_KALPHA1, CU_KALPHA2]),
        # C5: The polarization rate is set.
        # Expected: settings_version goes from 0 to 1.
        ("set_polarization", [0.95]),
    ],
)
def test_settings_version_after_setter(setter_name, setter_args):
    instrument = ConstantWavelengthInstrument()
    getattr(instrument, setter_name)(*setter_args)
    actual_settings_version = instrument.settings_version
    expected_settings_version = 1
    assert actual_settings_version == expected_settings_version


@pytest.mark.parametrize(
    "parameter_name, pyobjcryst_par_name, new_value",
    [
        # Test that update pushes the pattern Parameters into the pattern
        # C1: The zero shift is changed.
        # Expected: The pattern equals one with Zero set directly.
        ("zero", "Zero", 0.05),
        # C2: The displacement is changed.
        # Expected: The pattern equals one with 2ThetaDispl set directly.
        ("displacement", "2ThetaDispl", 0.05),
        # C3: The transparency is changed.
        # Expected: The pattern equals one with 2ThetaTransp set directly.
        ("transparency", "2ThetaTransp", 0.05),
        # C4: The cylinder absorption is set to 1.
        # Expected: The pattern equals one with MuR set directly.
        ("mu_r", "MuR", 1.0),
    ],
)
def test_update_pattern_parameters(
    pbso4_pattern,
    other_pbso4_pattern,
    parameter_name,
    pyobjcryst_par_name,
    new_value,
):
    # Install the instrument at 1.5406 angstroms, change a Parameter and
    # push it with update
    instrument = ConstantWavelengthInstrument()
    instrument.set_wavelength(1.5406)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.get(parameter_name).set_value(new_value)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # On a second pattern, already at 1.5406 angstroms, set the same value
    # directly in pyobjcryst
    other_pbso4_pattern.pattern.GetPar(pyobjcryst_par_name).SetHumanValue(
        new_value
    )
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


def test_update_wavelength(pbso4_pattern, other_pbso4_pattern):
    # Test that update pushes the wavelength into the pattern's radiation
    # Expected: After the wavelength is changed from 1.5406 to 1.6
    # angstroms, the pattern equals one with the Radiation's Wavelength set
    # directly.
    # Install the instrument at 1.5406 angstroms, then refine the
    # wavelength to 1.6 angstroms
    instrument = ConstantWavelengthInstrument()
    instrument.set_wavelength(1.5406)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.wavelength.set_value(1.6)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # On a second pattern, set the Wavelength directly in pyobjcryst
    radiation = other_pbso4_pattern.pattern.GetRadiation()
    radiation.GetPar("Wavelength").SetHumanValue(1.6)
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


def test_install_shared_instrument(pbso4_pattern, other_pbso4_pattern):
    # Test one instrument shared by two patterns, as the generators of the
    # phases of one dataset share it
    # Expected: Both patterns get the instrument's radiation type,
    # wavelength and zero shift.
    # Set up one neutron instrument with a zero shift
    instrument = ConstantWavelengthInstrument()
    instrument.set_radiation_type("N")
    instrument.set_wavelength(1.909)
    instrument.zero.set_value(0.05)
    shared_patterns = [pbso4_pattern, other_pbso4_pattern]
    # Install and update it in both patterns before reading either, so a
    # change to the second pattern that reaches the first is caught
    for shared_pattern in shared_patterns:
        instrument.install(shared_pattern.pattern, shared_pattern.diffraction)
        instrument.update(shared_pattern.pattern, shared_pattern.diffraction)
    actual_radiation_types = []
    actual_wavelengths = []
    actual_zeros = []
    for shared_pattern in shared_patterns:
        radiation = shared_pattern.pattern.GetRadiation()
        actual_radiation_types.append(radiation.GetRadiationType())
        actual_wavelengths.append(radiation.GetWavelength())
        zero = shared_pattern.pattern.GetPar("Zero").GetHumanValue()
        actual_zeros.append(zero)
    expected_radiation_types = [
        RadiationType.RAD_NEUTRON,
        RadiationType.RAD_NEUTRON,
    ]
    expected_wavelengths = [1.909, 1.909]
    expected_zeros = [0.05, 0.05]
    assert actual_radiation_types == expected_radiation_types
    assert actual_wavelengths == pytest.approx(expected_wavelengths)
    assert actual_zeros == pytest.approx(expected_zeros)


def test_install_neutron_radiation(pbso4_pattern, other_pbso4_pattern):
    # Test installing neutron radiation
    # Expected: The pattern equals a pyobjcryst pattern set directly to
    # neutrons at 1.909 angstroms.
    # Install a neutron instrument
    instrument = ConstantWavelengthInstrument()
    instrument.set_radiation_type("N")
    instrument.set_wavelength(1.909)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # On a second pattern, set neutrons directly in pyobjcryst
    other_pbso4_pattern.pattern.SetRadiationType(RadiationType.RAD_NEUTRON)
    other_pbso4_pattern.pattern.SetWavelength(1.909)
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


@pytest.mark.parametrize(
    "radiation_type, wavelength, expected_radiation_type",
    [
        # Test that install sets the radiation type of the pattern
        # C1: Neutrons at 1.909 angstroms are set.
        # Expected: The pattern's radiation is neutrons.
        ("N", 1.909, RadiationType.RAD_NEUTRON),
        # C2: Electrons at 0.0536 angstroms, those of a 50 kV beam, are
        # set. Faster electrons are below ObjCryst's wavelength limit of
        # 0.05 angstroms.
        # Expected: The pattern's radiation is electrons.
        ("E", 0.0536, RadiationType.RAD_ELECTRON),
    ],
)
def test_install_radiation_type(
    pbso4_pattern, radiation_type, wavelength, expected_radiation_type
):
    instrument = ConstantWavelengthInstrument()
    instrument.set_radiation_type(radiation_type)
    instrument.set_wavelength(wavelength)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    radiation = pbso4_pattern.pattern.GetRadiation()
    actual_radiation_type = radiation.GetRadiationType()
    assert actual_radiation_type == expected_radiation_type


def test_install_neutrons_after_tube(pbso4_pattern, other_pbso4_pattern):
    # Test switching an installed x-ray tube instrument to neutrons
    # Expected: The pattern equals a pyobjcryst pattern set directly to
    # neutrons at 1.909 angstroms, with no doublet left from the tube.
    # Install and calculate with a Cu tube
    instrument = ConstantWavelengthInstrument()
    instrument.set_xray_source("Cu")
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    pbso4_pattern.calculate()
    # Switch to a single neutron wavelength, then install and calculate
    # again
    instrument.set_wavelength(1.909)
    instrument.set_radiation_type("N")
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # On a second pattern, set neutrons directly in pyobjcryst
    other_pbso4_pattern.pattern.SetRadiationType(RadiationType.RAD_NEUTRON)
    other_pbso4_pattern.pattern.SetWavelength(1.909)
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


def test_install_polarization_ratio(pbso4_pattern, other_pbso4_pattern):
    # Test the polarization factor that a polarization rate gives
    # Expected: With a Cu tube, a rate f = 0.95 changes the strongest
    # PbSO4 peak, relative to an unpolarized beam (f = 0), by
    # (1 + A cos^2 2theta) / (1 + A) / ((1 + cos^2 2theta) / 2), with
    # A = (1 - f) / (1 + f) and 2theta that peak's position. (The peak is
    # at 29.69 degrees, and the ratio is 1.133.)
    # Calculate the pattern of an unpolarized beam
    unpolarized_instrument = ConstantWavelengthInstrument()
    unpolarized_instrument.set_xray_source("Cu")
    unpolarized_instrument.install(
        pbso4_pattern.pattern, pbso4_pattern.diffraction
    )
    unpolarized_instrument.update(
        pbso4_pattern.pattern, pbso4_pattern.diffraction
    )
    unpolarized_pattern = pbso4_pattern.calculate()
    # Calculate the pattern of a beam with f = 0.95 on a second pattern
    polarized_instrument = ConstantWavelengthInstrument()
    polarized_instrument.set_xray_source("Cu")
    polarized_instrument.set_polarization(0.95)
    polarized_instrument.install(
        other_pbso4_pattern.pattern, other_pbso4_pattern.diffraction
    )
    polarized_instrument.update(
        other_pbso4_pattern.pattern, other_pbso4_pattern.diffraction
    )
    polarized_pattern = other_pbso4_pattern.calculate()
    # Compare the strongest peak with the polarization formula
    peak_index = np.argmax(unpolarized_pattern)
    peak_two_theta = np.degrees(
        pbso4_pattern.pattern.GetPowderPatternX()[peak_index]
    )
    cos_squared = np.cos(np.radians(peak_two_theta)) ** 2
    polarization_a = (1 - 0.95) / (1 + 0.95)
    actual_ratio = polarized_pattern[peak_index] / (
        unpolarized_pattern[peak_index]
    )
    expected_ratio = (
        (1 + polarization_a * cos_squared)
        / (1 + polarization_a)
        / ((1 + cos_squared) / 2)
    )
    assert actual_ratio == pytest.approx(expected_ratio, rel=1e-3)


def test_install_after_polarization_change(pbso4_pattern, other_pbso4_pattern):
    # Test installing again after only the polarization rate changed,
    # which ObjCryst doesn't recalculate on its own
    # Expected: The pattern equals a pyobjcryst pattern built with a Cu
    # tube and f = 0.95 from the start.
    # Install and calculate with an unpolarized Cu tube
    instrument = ConstantWavelengthInstrument()
    instrument.set_xray_source("Cu")
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    pbso4_pattern.calculate()
    # Change only the polarization rate, then install and calculate again
    instrument.set_polarization(0.95)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # On a second pattern, build the Cu tube with f = 0.95 directly
    other_pbso4_pattern.pattern.SetWavelength("Cu", 0.5)
    other_pbso4_pattern.pattern.GetRadiation().SetLinearPolarRate(0.95)
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


def test_install_tube_set_after_polarization(
    pbso4_pattern, other_pbso4_pattern
):
    # Test setting the tube after the polarization rate, since ObjCryst
    # resets the rate when a tube is set
    # Expected: The pattern still has the rate, f = 0.95: it equals a
    # pyobjcryst pattern with the Cu tube set first and the rate second.
    # Set the polarization first and the tube second, then install
    instrument = ConstantWavelengthInstrument()
    instrument.set_polarization(0.95)
    instrument.set_xray_source("Cu")
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # On a second pattern, set the Cu tube and then the rate directly
    other_pbso4_pattern.pattern.SetWavelength("Cu", 0.5)
    other_pbso4_pattern.pattern.GetRadiation().SetLinearPolarRate(0.95)
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


@pytest.mark.parametrize(
    "polarization_rate",
    [
        # Test that the polarization rates at the edges of [-1, 1] are
        # accepted and installed
        # C1: The rate is -1.
        # Expected: The pattern's radiation has a rate of -1.
        -1.0,
        # C2: The rate is 1.
        # Expected: The pattern's radiation has a rate of 1.
        1.0,
    ],
)
def test_install_polarization_rate_at_range_edges(
    pbso4_pattern, polarization_rate
):
    instrument = ConstantWavelengthInstrument()
    instrument.set_xray_source("Cu")
    instrument.set_polarization(polarization_rate)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    radiation = pbso4_pattern.pattern.GetRadiation()
    actual_polarization_rate = radiation.GetLinearPolarRate()
    expected_polarization_rate = polarization_rate
    assert actual_polarization_rate == pytest.approx(
        expected_polarization_rate
    )


def test_install_single_wavelength_after_tube(
    pbso4_pattern, other_pbso4_pattern
):
    # Test switching from a tube to a single wavelength, since ObjCryst
    # keeps the tube's Kalpha1/Kalpha2 doublet after SetWavelength(float)
    # Expected: The pattern equals a single-wavelength pyobjcryst pattern
    # at 1.5406 angstroms, not a doublet.
    # Install and calculate with a Cu tube
    instrument = ConstantWavelengthInstrument()
    instrument.set_xray_source("Cu")
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    pbso4_pattern.calculate()
    # Switch to a single wavelength, then install and calculate again
    instrument.set_wavelength(1.5406)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # The second pattern was built at a single wavelength of 1.5406
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


@pytest.mark.parametrize(
    "kalpha2_ratio",
    [
        # Test installing a Cu tube
        # C1: The Kalpha2/Kalpha1 ratio is the default, 0.5.
        # Expected: The pattern equals a pyobjcryst pattern set directly to
        # the Cu tube with that ratio.
        0.5,
        # C2: The ratio is 0.4, which ObjCryst would clamp to 0.5 if it
        # were set through its RefinablePar.
        # Expected: The pattern equals a pyobjcryst pattern set directly to
        # the Cu tube with a ratio of 0.4.
        0.4,
    ],
)
def test_install_xray_source(
    pbso4_pattern, other_pbso4_pattern, kalpha2_ratio
):
    # Install the Cu tube
    instrument = ConstantWavelengthInstrument()
    instrument.set_xray_source("Cu", kalpha2_ratio)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # On a second pattern, set the Cu tube directly in pyobjcryst
    other_pbso4_pattern.pattern.SetWavelength("Cu", kalpha2_ratio)
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


@pytest.mark.parametrize(
    "kalpha2_ratio",
    [
        # Test setting the Kalpha1 and Kalpha2 wavelengths explicitly
        # C1: ObjCryst's Cu wavelengths are set with the default
        # Kalpha2/Kalpha1 ratio, 0.5.
        # Expected: The pattern equals a pyobjcryst pattern set directly to
        # the Cu tube with that ratio.
        0.5,
        # C2: ObjCryst's Cu wavelengths are set with a ratio of 0.4.
        # Expected: The pattern equals a pyobjcryst pattern set directly to
        # the Cu tube with a ratio of 0.4.
        0.4,
    ],
)
def test_install_xray_wavelengths(
    pbso4_pattern, other_pbso4_pattern, kalpha2_ratio
):
    # Install the explicit wavelengths
    instrument = ConstantWavelengthInstrument()
    instrument.set_xray_wavelengths(CU_KALPHA1, CU_KALPHA2, kalpha2_ratio)
    instrument.install(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    instrument.update(pbso4_pattern.pattern, pbso4_pattern.diffraction)
    actual_pattern = pbso4_pattern.calculate()
    # On a second pattern, set the Cu tube directly in pyobjcryst
    other_pbso4_pattern.pattern.SetWavelength("Cu", kalpha2_ratio)
    expected_pattern = other_pbso4_pattern.calculate()
    assert np.allclose(actual_pattern, expected_pattern)


def test_set_xray_source_unknown_element():
    # Test setting a tube whose element ObjCryst has no wavelengths for
    # Expected: A ValueError names the element and how to set the
    # wavelengths instead, rather than ObjCryst's printed warning.
    instrument = ConstantWavelengthInstrument()
    expected_msg = (
        "ObjCryst has no x-ray tube wavelengths for 'Xx'. Please use one of "
        "the anode elements it has, 'Ag', 'Co', 'Cr', 'Cu', 'Fe' or 'Mo', or "
        "set the wavelengths with set_xray_wavelengths."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        instrument.set_xray_source("Xx")


@pytest.mark.parametrize(
    "setter_name, setter_args, expected_msg",
    [
        # Test setting a wavelength outside ObjCryst's Wavelength limits,
        # [0.05, 20] angstroms, which ObjCryst would clamp
        # C1: A single wavelength above 20 angstroms is set.
        # Expected: A ValueError names the wavelength and the limits.
        (
            "set_wavelength",
            [25.0],
            "The wavelength 25.0 angstroms is outside ObjCryst's limits of "
            "0.05 to 20 angstroms. Please give a wavelength within these "
            "limits.",
        ),
        # C2: The 0.0251 angstroms of a 200 kV electron beam, below 0.05
        # angstroms, is set.
        # Expected: A ValueError names the wavelength and the limits.
        (
            "set_wavelength",
            [0.0251],
            "The wavelength 0.0251 angstroms is outside ObjCryst's limits of "
            "0.05 to 20 angstroms. Please give a wavelength within these "
            "limits.",
        ),
        # C3: Kalpha1 and Kalpha2 wavelengths above 20 angstroms are set.
        # Expected: A ValueError names Kalpha1, the first wavelength
        # outside the limits.
        (
            "set_xray_wavelengths",
            [25.0, 25.004],
            "The wavelength 25.0 angstroms is outside ObjCryst's limits of "
            "0.05 to 20 angstroms. Please give a wavelength within these "
            "limits.",
        ),
    ],
)
def test_set_wavelength_outside_limits(setter_name, setter_args, expected_msg):
    instrument = ConstantWavelengthInstrument()
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        getattr(instrument, setter_name)(*setter_args)


def test_set_xray_wavelengths_kalpha2_not_longer():
    # Test setting a Kalpha2 that isn't longer than Kalpha1, which would
    # make ObjCryst's Kalpha2 - Kalpha1 negative
    # Expected: A ValueError says that the wavelengths are swapped.
    instrument = ConstantWavelengthInstrument()
    expected_msg = (
        "Kalpha2, 1.54056 angstroms, is not longer than Kalpha1, 1.54439 "
        "angstroms. Please give the wavelengths in the order kalpha1, "
        "kalpha2."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        instrument.set_xray_wavelengths(1.544390, 1.540560)


@pytest.mark.parametrize(
    "setter_name, setter_args",
    [
        # Test setting a Kalpha1/Kalpha2 doublet while the radiation type
        # is neutrons, since a doublet is x-ray tube radiation
        # C1: A Cu tube is set.
        # Expected: A ValueError says that a doublet needs x-rays.
        ("set_xray_source", ["Cu"]),
        # C2: The Cu Kalpha1 and Kalpha2 wavelengths are set explicitly.
        # Expected: A ValueError says that a doublet needs x-rays.
        ("set_xray_wavelengths", [CU_KALPHA1, CU_KALPHA2]),
    ],
)
def test_set_doublet_with_neutron_radiation(setter_name, setter_args):
    instrument = ConstantWavelengthInstrument()
    instrument.set_radiation_type("N")
    expected_msg = (
        "A Kalpha1/Kalpha2 doublet needs x-rays, but the radiation type is "
        "'N'. Please set the radiation type to 'X' first, or set a single "
        "wavelength with set_wavelength."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        getattr(instrument, setter_name)(*setter_args)


@pytest.mark.parametrize(
    "radiation_type, expected_msg",
    [
        # Test setting a radiation type other than x-rays while a
        # Kalpha1/Kalpha2 doublet is set
        # C1: Neutrons are set after a Cu tube.
        # Expected: A ValueError says to set a single wavelength first.
        (
            "N",
            "The radiation type 'N' can't be used with the Kalpha1/Kalpha2 "
            "doublet that is set, which needs x-rays. Please set a single "
            "wavelength with set_wavelength first.",
        ),
        # C2: Electrons are set after a Cu tube.
        # Expected: A ValueError says to set a single wavelength first.
        (
            "E",
            "The radiation type 'E' can't be used with the Kalpha1/Kalpha2 "
            "doublet that is set, which needs x-rays. Please set a single "
            "wavelength with set_wavelength first.",
        ),
    ],
)
def test_set_radiation_type_with_doublet(radiation_type, expected_msg):
    instrument = ConstantWavelengthInstrument()
    instrument.set_xray_source("Cu")
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        instrument.set_radiation_type(radiation_type)


def test_set_radiation_type_unknown_type():
    # Test setting a radiation type that isn't x-rays, neutrons or
    # electrons
    # Expected: A ValueError names the type and lists the supported ones.
    instrument = ConstantWavelengthInstrument()
    expected_msg = (
        "'Q' is not a supported radiation type. Please use 'X' for x-rays, "
        "'N' for neutrons or 'E' for electrons."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        instrument.set_radiation_type("Q")


@pytest.mark.parametrize(
    "polarization_rate, expected_msg",
    [
        # Test setting a polarization rate outside [-1, 1]
        # C1: The rate is above 1.
        # Expected: A ValueError names the rate and how to convert a GSAS
        # polarization.
        (
            2.0,
            "The polarization rate 2.0 is outside [-1, 1]. Please give "
            "ObjCryst's rate f; for a GSAS polarization P, f = 2P - 1.",
        ),
        # C2: The rate is below -1.
        # Expected: A ValueError names the rate and how to convert a GSAS
        # polarization.
        (
            -1.5,
            "The polarization rate -1.5 is outside [-1, 1]. Please give "
            "ObjCryst's rate f; for a GSAS polarization P, f = 2P - 1.",
        ),
    ],
)
def test_set_polarization_outside_range(polarization_rate, expected_msg):
    instrument = ConstantWavelengthInstrument()
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        instrument.set_polarization(polarization_rate)
