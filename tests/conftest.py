import json
from pathlib import Path

import numpy as np
import pytest
from pyobjcryst.crystal import create_crystal_from_cif

TESTDATA_DIR = Path(__file__).parent / "testdata"


class NotificationCounter:
    """Count the notifications sent to an srfit observer."""

    def __init__(self):
        self.count = 0

    def record_notification(self, observed_objects):
        self.count += 1


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


@pytest.fixture
def notification_counter():
    counter = NotificationCounter()
    return counter


@pytest.fixture
def pbso4_crystal():
    crystal = create_crystal_from_cif(str(TESTDATA_DIR / "PbSO4-Wyckoff.cif"))
    return crystal


@pytest.fixture
def pbso4_tth():
    tth = np.arange(16.0, 110.0, 0.025)
    return tth


@pytest.fixture
def pbso4_generator(pbso4_crystal):
    """Return a RietveldGenerator for PbSO4 with Cu Kalpha1 x-rays.

    The peak width W is set to 0.01 degrees squared so that the peaks
    span several points of a 0.025 degree grid.
    """
    from diffpy.cmirietveld.rietveldgenerator import RietveldGenerator

    generator = RietveldGenerator("bragg")
    generator.set_wavelength(1.5406)
    generator.set_structure(pbso4_crystal)
    generator.W.set_value(0.01)
    return generator
