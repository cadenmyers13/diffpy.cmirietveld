import re

import pytest
from pyobjcryst.reflectionprofile import ReflectionProfilePseudoVoigt

from diffpy.cmirietveld.component import Component


def test_create_parameters_from_names(pseudo_voigt_component):
    # Test that each name in the mapping becomes a Parameter
    # Expected: The component holds one Parameter per name, and no others.
    actual_names = sorted(pseudo_voigt_component.get_names())
    expected_names = [
        "U",
        "V",
        "W",
        "asym0",
        "asym1",
        "asym2",
        "eta0",
        "eta1",
    ]
    assert actual_names == expected_names


@pytest.mark.parametrize(
    "parameter_name, expected_value, expected_bounds",
    [
        # Test that each Parameter takes the RefinablePar's human value
        # and limits
        # C1: The Caglioti W term is created from the pseudo-Voigt profile.
        # Expected: Its value is ObjCryst's default, 0.01 deg^2, and its
        # bounds are ObjCryst's limits, [0, 1].
        ("W", 0.01, [0.0, 1.0]),
        # C2: The Caglioti U term, which can be negative, is created.
        # Expected: Its value is 0 and its bounds are [-1, 1].
        ("U", 0.0, [-1.0, 1.0]),
        # C3: The mixing term eta0 is created under a name that differs
        # from its RefinablePar name, Eta0.
        # Expected: Its value is 0.5 and its bounds are [0, 1].
        ("eta0", 0.5, [0.0, 1.0]),
        # C4: The asymmetry term asym0, whose lower limit is not zero, is
        # created.
        # Expected: Its value is 1 and its bounds are [0.01, 10].
        ("asym0", 1.0, [0.01, 10.0]),
    ],
)
def test_create_parameters_from_values_and_bounds(
    pseudo_voigt_component, parameter_name, expected_value, expected_bounds
):
    parameter = pseudo_voigt_component.get(parameter_name)
    actual_value = parameter.value
    actual_bounds = list(parameter.bounds)
    assert actual_value == pytest.approx(expected_value)
    assert actual_bounds == expected_bounds


def test_create_parameters_from_unknown_name():
    # Test creating a Parameter from a RefinablePar the model doesn't have
    # Expected: A ValueError names the missing RefinablePar and lists the
    # names the model has.
    component = Component("peak_profile")
    expected_msg = (
        "ReflectionProfilePseudoVoigt has no refinable parameter named "
        "'nonexistent_name'. Please use one of its parameter names: U, V, "
        "W, Eta0, Eta1, Asym0, Asym1, Asym2."
    )
    bad_input = {"any_name": "nonexistent_name"}
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        component.create_parameters_from(
            ReflectionProfilePseudoVoigt(), bad_input
        )


@pytest.mark.parametrize(
    "parameter_name, pyobjcryst_par_name, new_value",
    [
        # Test that Parameter values are pushed into a pyobjcryst model
        # other than the one the Parameters were created from
        # C1: The Caglioti W term is set inside its limits.
        # Expected: The W RefinablePar has the new value.
        ("W", "W", 0.05),
        # C2: The mixing term eta0 is set under a name that differs from
        # its RefinablePar name.
        # Expected: The Eta0 RefinablePar has the new value.
        ("eta0", "Eta0", 0.3),
        # C3: The asymmetry term asym0 is set exactly at its upper limit.
        # Expected: The Asym0 RefinablePar has the new value, because a
        # value at a limit is inside the limits.
        ("asym0", "Asym0", 10.0),
    ],
)
def test_push_parameters(
    pseudo_voigt_component, parameter_name, pyobjcryst_par_name, new_value
):
    installed_profile = ReflectionProfilePseudoVoigt()
    pseudo_voigt_component.get(parameter_name).set_value(new_value)
    pseudo_voigt_component.push_parameters(installed_profile)
    actual_value = installed_profile.GetPar(
        pyobjcryst_par_name
    ).GetHumanValue()
    expected_value = new_value
    assert actual_value == pytest.approx(expected_value)


@pytest.mark.parametrize(
    "parameter_bounds, new_values, invalid_name, invalid_value, "
    "expected_limits, expected_used_value",
    [
        # Test pushing Parameter values when one is outside ObjCryst's
        # limits
        # C1: The Caglioti W term is set above its upper limit, with its
        # bounds left at ObjCryst's limits.
        # Expected: A ValueError names W, its limits and the value ObjCryst
        # would use, and the model is unchanged.
        ({}, {"W": 2.0}, "W", "2.0", "[0.0, 1.0]", "1.0"),
        # C2: The asymmetry term asym0 is set below its lower limit, with its
        # bounds left at ObjCryst's limits.
        # Expected: A ValueError names asym0, its limits and the value
        # ObjCryst would use, and the model is unchanged.
        ({}, {"asym0": 0.001}, "asym0", "0.001", "[0.01, 10.0]", "0.01"),
        # C3: The W term's bounds are widened past ObjCryst's limits to
        # [0, 5], and W is set to 2, inside the bounds.
        # Expected: The check uses ObjCryst's limits, not the Parameter's
        # bounds, so a ValueError names the limits [0, 1], and the model is
        # unchanged.
        ({"W": [0.0, 5.0]}, {"W": 2.0}, "W", "2.0", "[0.0, 1.0]", "1.0"),
        # C4: W, which comes first in the mapping, is set inside its limits,
        # and asym0 is set above its upper limit, with both bounds left at
        # ObjCryst's limits.
        # Expected: A ValueError names asym0, and no value is pushed, not
        # even W's, so the model is unchanged rather than half updated.
        (
            {},
            {"W": 0.05, "asym0": 20.0},
            "asym0",
            "20.0",
            "[0.01, 10.0]",
            "10.0",
        ),
    ],
)
def test_push_parameters_outside_limits(
    pseudo_voigt_component,
    parameter_bounds,
    new_values,
    invalid_name,
    invalid_value,
    expected_limits,
    expected_used_value,
):
    expected_msg = (
        f"The value of '{invalid_name}', {invalid_value}, is outside its "
        f"limits {expected_limits}. ObjCryst would silently use "
        f"{expected_used_value} instead, so the refined value would not be "
        "the one used to calculate the pattern. Please keep the value "
        "within the limits, for example by passing the recipe's bounds to "
        "the optimizer: least_squares(..., bounds=recipe.get_bounds_array())."
    )
    installed_profile = ReflectionProfilePseudoVoigt()
    refinable_par_count = installed_profile.GetNbPar()
    expected_values = [
        installed_profile.GetPar(index).GetHumanValue()
        for index in range(refinable_par_count)
    ]
    for parameter_name, bounds in parameter_bounds.items():
        pseudo_voigt_component.get(parameter_name).bound_range(*bounds)
    for parameter_name, new_value in new_values.items():
        pseudo_voigt_component.get(parameter_name).set_value(new_value)
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        pseudo_voigt_component.push_parameters(installed_profile)
    actual_values = [
        installed_profile.GetPar(index).GetHumanValue()
        for index in range(refinable_par_count)
    ]
    assert actual_values == expected_values


def test_push_parameters_outside_narrowed_bounds(pseudo_voigt_component):
    # Test that a value outside narrowed bounds but inside ObjCryst's
    # limits is pushed
    # Expected: W = 0.5 is outside the narrowed bounds [0, 0.1] but inside
    # ObjCryst's limits [0, 1], so the W RefinablePar has the new value.
    # Keeping a fit inside the bounds is the optimizer's job.
    installed_profile = ReflectionProfilePseudoVoigt()
    parameter = pseudo_voigt_component.get("W")
    parameter.bound_range(0.0, 0.1)
    parameter.set_value(0.5)
    pseudo_voigt_component.push_parameters(installed_profile)
    actual_value = installed_profile.GetPar("W").GetHumanValue()
    expected_value = 0.5
    assert actual_value == pytest.approx(expected_value)
