import json
from pathlib import Path

import numpy as np
import pytest
from pyobjcryst.crystal import create_crystal_from_cif
from pyobjcryst.powderpattern import PowderPattern
from pyobjcryst.reflectionprofile import ReflectionProfilePseudoVoigt

from diffpy.cmirietveld.component import Component

TESTDATA_DIR = Path(__file__).parent / "testdata"
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


class PbSO4Pattern:
    """A plain pyobjcryst pattern for PbSO4, on a 16-110 degree grid at
    1.5406 angstroms, with its diffraction component."""

    def __init__(self):
        crystal = create_crystal_from_cif(
            str(TESTDATA_DIR / "PbSO4-Wyckoff.cif")
        )
        tth = np.arange(16.0, 110.0, 0.025)
        self.pattern = PowderPattern()
        self.pattern.SetWavelength(1.5406)
        self.pattern.SetPowderPatternX(np.radians(tth))
        self.pattern.SetPowderPatternObs(np.zeros_like(tth))
        self.diffraction = self.pattern.AddPowderPatternDiffraction(crystal)

    def calculate(self):
        self.pattern.Prepare()
        calculated_pattern = np.array(self.pattern.GetPowderPatternCalc())
        return calculated_pattern


@pytest.fixture
def pbso4_pattern():
    pbso4_pattern = PbSO4Pattern()
    return pbso4_pattern


@pytest.fixture
def other_pbso4_pattern():
    other_pbso4_pattern = PbSO4Pattern()
    return other_pbso4_pattern


@pytest.fixture
def pseudo_voigt_component():
    component = Component("peak_profile")
    component.create_parameters_from(
        ReflectionProfilePseudoVoigt(), PSEUDO_VOIGT_PARAMETER_NAMES
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
