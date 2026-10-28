import json
from pathlib import Path

import pytest
from pyobjcryst.reflectionprofile import ReflectionProfilePseudoVoigt

from diffpy.cmirietveld.component import Component

PSEUDO_VOIGT_PARAMETER_NAMES = {
    "U": "U",
    "V": "V",
    "W": "W",
    "eta0": "Eta0",
    "eta1": "Eta1",
    "asym0": "Asym0",
    "asym1": "Asym1",
    "asym2": "Asym2",
}


@pytest.fixture
def pseudo_voigt_profile():
    profile = ReflectionProfilePseudoVoigt()
    return profile


@pytest.fixture
def installed_pseudo_voigt_profile():
    """Return a profile separate from the one the component's Parameters
    are created from, like the copy a generator installs."""
    profile = ReflectionProfilePseudoVoigt()
    return profile


@pytest.fixture
def pseudo_voigt_component(pseudo_voigt_profile):
    component = Component("peak_profile")
    component.create_parameters_from(
        pseudo_voigt_profile, PSEUDO_VOIGT_PARAMETER_NAMES
    )
    return component


@pytest.fixture
def user_filesystem(tmp_path):
    base_dir = Path(tmp_path)
    home_dir = base_dir / "home_dir"
    home_dir.mkdir(parents=True, exist_ok=True)
    cwd_dir = base_dir / "cwd_dir"
    cwd_dir.mkdir(parents=True, exist_ok=True)

    home_config_data = {"username": "home_username", "email": "home@email.com"}
    with open(home_dir / "diffpyconfig.json", "w") as f:
        json.dump(home_config_data, f)

    yield tmp_path
