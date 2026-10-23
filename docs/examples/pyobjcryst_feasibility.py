"""Check that pyobjcryst can drive a Rietveld refinement in srfit.

It refines the PbSO4 x-ray powder pattern in the repository's test data
with a small prototype generator, defined below, that calculates the
Bragg intensity with pyobjcryst. The refinement runs in the same stages
as a GSAS-II reference refinement of the same data over 16-110 degrees,
and prints both side by side.

The script settles four questions for the real generator:

1. Units. ObjCryst stores angles in radians and the Caglioti U, V, W in
   radians squared. Each RefinablePar also has a "human" value in degrees
   and degrees squared, which is what the prototype exposes.
2. The Cu Kalpha1/Kalpha2 doublet. ``PowderPattern.SetWavelength("Cu", 0.5)``
   models it, matching the 0.5 intensity ratio in INST_XRY.PRM.
3. The background. A 6-term Chebyshev polynomial registered as a function
   in the srfit equation, like the GSAS-II reference, keeps the generator
   to Bragg intensity only.
4. The x grid. The script compares the pattern on a 2x coarser grid with
   the pattern on the observed grid. They agree while the peak asymmetry
   terms are at their defaults, but once asymmetry is refined the pattern
   depends on the grid spacing, so the generator should calculate on the
   observed grid.

The script refines twice: with the pseudo-Voigt asymmetry terms fixed at
their defaults, and with asym0 and asym1 refined. In pyobjcryst 2025.1.0
the PROFILE_PSEUDO_VOIGT and PROFILE_PSEUDO_VOIGT_FINGER_COX_JEPHCOAT
profile types give identical patterns; both use the Asym0-Asym2 terms.

Two pyobjcryst behaviours the real generator must handle:

- Every value must be changed through its RefinablePar (``GetPar``), so
  that ObjCryst knows the pattern is out of date. For example,
  ``PowderPattern.SetScaleFactor`` changes the scale without invalidating
  the cached pattern, while the "Scale_" RefinablePar does invalidate it.
- A diffraction component can only be added once the pattern has a 2theta
  grid, because it needs the range to generate reflections. The prototype
  sets a placeholder grid and replaces it at the first calculation.
- pyobjcryst does not regenerate the reflection list when the 2theta grid,
  the wavelength or the lattice changes, so the prototype calls
  ``PowderPattern.Prepare`` before every calculation.

Run it from anywhere with

    python docs/examples/pyobjcryst_feasibility.py
"""

from pathlib import Path

import numpy as np
from pyobjcryst.crystal import create_crystal_from_cif
from pyobjcryst.powderpattern import PowderPattern, ReflectionProfileType
from pyobjcryst.refinableobj import RefinablePar
from scipy.optimize import least_squares

from diffpy.srfit.fitbase import (
    FitContribution,
    FitRecipe,
    FitResults,
    Profile,
    ProfileGenerator,
)
from diffpy.srfit.fitbase.parameter import ParameterAdapter
from diffpy.srfit.structure import struToParameterSet

TESTDATA_DIR = Path(__file__).resolve().parents[2] / "tests" / "testdata"
TTH_LIMITS = (16.0, 110.0)
BACKGROUND_TERMS = ["b0", "b1", "b2", "b3", "b4", "b5"]
ASYMMETRY_SETTINGS = {
    "asymmetry fixed at defaults": False,
    "asymmetry refined": True,
}

# Rwp (%) and cell (angstroms) from GSAS-II refining the same data with
# the same limits, stage by stage (GSAS-II from ~/g2main, 2026-10-15).
GSASII_REFERENCE = {
    "scale, background": (45.812, (8.48000, 5.39800, 6.95800)),
    "+ cell": (32.066, (8.48578, 5.40226, 6.96410)),
    "+ displacement": (27.398, (8.48144, 5.39943, 6.96057)),
    "+ peak profile": (13.462, (8.47997, 5.39854, 6.95982)),
    "+ atom positions, ADPs": (10.070, (8.48035, 5.39862, 6.96002)),
}


def load_gsas_std_data(data_path):
    """Load a GSAS "STD" constant-step powder data file.

    The header line ``BANK 1 <npoints> <nrecords> CONST <start> <step> ...``
    gives the start and step in centidegrees. The uncertainties are the
    square roots of the counts, as GSAS and GSAS-II assume for STD files.

    Parameters
    ----------
    data_path : Path
        The path to the data file.

    Returns
    -------
    tth : ndarray
        The 2theta values in degrees.
    counts : ndarray
        The observed counts.
    counts_uncertainty : ndarray
        The uncertainty of the counts.
    """
    lines = data_path.read_text().splitlines()
    bank_fields = lines[1].split()
    n_points = int(bank_fields[2])
    start = float(bank_fields[5]) / 100
    step = float(bank_fields[6]) / 100
    counts = np.array(
        [float(value) for line in lines[2:] for value in line.split()]
    )[:n_points]
    tth = start + step * np.arange(n_points)
    counts_uncertainty = np.sqrt(counts)
    return tth, counts, counts_uncertainty


def compute_chebyshev_background(tth, b0, b1, b2, b3, b4, b5):
    """Compute a Chebyshev polynomial background over the 2theta range.

    Parameters
    ----------
    tth : ndarray
        The 2theta values in degrees.
    b0, b1, b2, b3, b4, b5 : float
        The Chebyshev coefficients.

    Returns
    -------
    ndarray
        The background at each 2theta value.
    """
    scaled_tth = (2 * tth - tth.min() - tth.max()) / (tth.max() - tth.min())
    background = np.polynomial.chebyshev.chebval(
        scaled_tth, [b0, b1, b2, b3, b4, b5]
    )
    return background


def make_human_parameter(name, refinable_par):
    """Wrap a pyobjcryst RefinablePar as an srfit Parameter in human
    units.

    Parameters
    ----------
    name : str
        The name of the srfit Parameter.
    refinable_par : RefinablePar
        The pyobjcryst parameter to wrap.

    Returns
    -------
    ParameterAdapter
        The Parameter reading and writing the human value (degrees for
        angles, degrees squared for U, V, W).
    """
    parameter = ParameterAdapter(
        name,
        refinable_par,
        getter=RefinablePar.GetHumanValue,
        setter=RefinablePar.SetHumanValue,
    )
    return parameter


class PrototypeRietveldGenerator(ProfileGenerator):
    """Calculate a powder pattern from a structure with pyobjcryst.

    This is a minimal prototype of the planned RietveldGenerator, to check
    feasibility. It mirrors srfit's PDFGenerator: the structure comes in as
    a ParameterSet, which other generators can share.

    Parameters
    ----------
    name : str
        The name of the generator.
    phase : ObjCrystCrystalParSet
        The structure ParameterSet, made with ``struToParameterSet``.
    """

    def __init__(self, name, phase):
        ProfileGenerator.__init__(self, name)
        self.phase = phase
        self.add_parameter_set(phase)
        self._pattern = PowderPattern()
        self._pattern.SetWavelength("Cu", 0.5)
        placeholder_tth = np.linspace(5.0, 175.0, 1000)
        self._pattern.SetPowderPatternX(np.radians(placeholder_tth))
        self._pattern.SetPowderPatternObs(np.zeros_like(placeholder_tth))
        self._diffraction = self._pattern.AddPowderPatternDiffraction(
            phase.stru
        )
        self._diffraction.SetReflectionProfilePar(
            ReflectionProfileType.PROFILE_PSEUDO_VOIGT
        )
        self._last_tth = np.empty(0)
        for parameter_name, objcryst_name in [
            ("scale", "Scale_"),
            ("zero", "Zero"),
            ("displacement", "2ThetaDispl"),
        ]:
            self.add_parameter(
                make_human_parameter(
                    parameter_name, self._pattern.GetPar(objcryst_name)
                )
            )
        profile_parameter_names = [
            ("U", "U"),
            ("V", "V"),
            ("W", "W"),
            ("eta0", "Eta0"),
            ("eta1", "Eta1"),
            ("asym0", "Asym0"),
            ("asym1", "Asym1"),
        ]
        profile = self._diffraction.GetProfile()
        for parameter_name, objcryst_name in profile_parameter_names:
            self.add_parameter(
                make_human_parameter(
                    parameter_name, profile.GetPar(objcryst_name)
                )
            )

    def __call__(self, tth):
        """Calculate the Bragg intensity at the given 2theta values.

        Parameters
        ----------
        tth : ndarray
            The 2theta values in degrees.

        Returns
        -------
        ndarray
            The calculated Bragg intensity.
        """
        if not np.array_equal(tth, self._last_tth):
            self._last_tth = tth.copy()
            self._pattern.SetPowderPatternX(np.radians(tth))
            self._pattern.SetPowderPatternObs(np.zeros_like(tth))
        self._pattern.Prepare()
        bragg_intensity = np.array(self._pattern.GetPowderPatternCalc())
        return bragg_intensity


def make_recipe(refine_asymmetry):
    """Make the FitRecipe for the PbSO4 x-ray refinement.

    The starting peak profile approximates the GSAS-II instrument file:
    GSAS-II's U, V, W are Gaussian variances in centidegrees squared,
    while ObjCryst's are FWHM squared in degrees squared, so they differ by
    8 ln 2 / 10**4.

    Parameters
    ----------
    refine_asymmetry : bool
        The flag indicating if asym0 and asym1 are refined with the peak
        profile.

    Returns
    -------
    FitRecipe
        The recipe, with all variables fixed and tagged by stage.
    """
    tth, counts, counts_uncertainty = load_gsas_std_data(
        TESTDATA_DIR / "PBSO4.XRA"
    )
    within_limits = (tth >= TTH_LIMITS[0]) & (tth <= TTH_LIMITS[1])
    profile = Profile()
    profile.set_observed_profile(
        tth[within_limits],
        counts[within_limits],
        counts_uncertainty[within_limits],
    )
    phase = struToParameterSet(
        "PbSO4",
        create_crystal_from_cif(str(TESTDATA_DIR / "PbSO4-Wyckoff.cif")),
    )
    generator = PrototypeRietveldGenerator("bragg", phase)
    gaussian_variance_to_fwhm_squared = 8 * np.log(2) / 1e4
    generator.U.set_value(2.0 * gaussian_variance_to_fwhm_squared)
    generator.V.set_value(-2.0 * gaussian_variance_to_fwhm_squared)
    generator.W.set_value(5.0 * gaussian_variance_to_fwhm_squared)
    contribution = FitContribution("pbso4")
    contribution.set_profile(profile, xname="tth")
    contribution.add_profile_generator(generator)
    contribution.register_function(
        compute_chebyshev_background,
        name="background",
        argnames=["tth"] + BACKGROUND_TERMS,
    )
    contribution.set_equation("bragg + background")
    recipe = FitRecipe()
    recipe.clear_fit_hooks()
    recipe.add_contribution(contribution)
    recipe.add_variable(generator.scale, 1e-5, tags=["scale"])
    recipe.add_variable(contribution.b0, 150.0, tags=["background"])
    for term in BACKGROUND_TERMS[1:]:
        recipe.add_variable(contribution.get(term), 0.0, tags=["background"])
    for parameter in phase.sgpars.latpars:
        recipe.add_variable(parameter, tags=["cell"])
    recipe.add_variable(generator.displacement, tags=["displacement"])
    profile_parameter_names = ["U", "V", "W", "eta0", "eta1"]
    if refine_asymmetry:
        profile_parameter_names += ["asym0", "asym1"]
    for name in profile_parameter_names:
        recipe.add_variable(generator.get(name), tags=["profile"])
    for parameter in phase.sgpars.xyzpars:
        recipe.add_variable(parameter, tags=["atoms"])
    for scatterer in phase.getScatterers():
        recipe.add_variable(
            scatterer.Biso,
            0.8,
            name=f"Biso_{scatterer.name}",
            tags=["atoms"],
        )
    recipe.fix("all")
    return recipe


def check_coarse_grid(generator, tth):
    """Check that the pattern can be calculated on a coarser 2theta
    grid.

    Parameters
    ----------
    generator : PrototypeRietveldGenerator
        The generator, after refinement.
    tth : ndarray
        The observed 2theta values in degrees.

    Returns
    -------
    float
        The largest difference between the pattern on every other point
        and the full pattern at the same points, relative to its maximum.
    """
    full_pattern = generator(tth)
    coarse_pattern = generator(tth[::2])
    relative_difference = np.max(
        np.abs(coarse_pattern - full_pattern[::2])
    ) / np.max(full_pattern)
    return relative_difference


def refine_in_stages(setting_name, refine_asymmetry):
    """Refine in stages and compare each stage with GSAS-II.

    Parameters
    ----------
    setting_name : str
        The name of the asymmetry setting, for the printed table.
    refine_asymmetry : bool
        The flag indicating if asym0 and asym1 are refined with the peak
        profile.

    Returns
    -------
    FitRecipe
        The refined recipe.
    """
    recipe = make_recipe(refine_asymmetry)
    contribution = recipe.pbso4
    stages = {
        "scale, background": ["scale", "background"],
        "+ cell": ["cell"],
        "+ displacement": ["displacement"],
        "+ peak profile": ["profile"],
        "+ atom positions, ADPs": ["atoms"],
    }
    print(f"\nPeak profile: pseudo-Voigt, {setting_name}")
    print(
        f"{'stage':<24}{'srfit Rw':>9}{'GSAS-II':>9}   cell (srfit / GSAS-II)"
    )
    for stage, tags in stages.items():
        recipe.free(*tags)
        least_squares(recipe.residual, recipe.get_values(), x_scale="jac")
        results = FitResults(recipe)
        phase = contribution.bragg.phase
        srfit_cell = (phase.a.value, phase.b.value, phase.c.value)
        gsasii_rwp, gsasii_cell = GSASII_REFERENCE[stage]
        cell_text = " ".join(f"{value:.5f}" for value in srfit_cell)
        reference_text = " ".join(f"{value:.5f}" for value in gsasii_cell)
        print(
            f"{stage:<24}{100 * results.rw:>9.3f}{gsasii_rwp:>9.3f}"
            f"   {cell_text} / {reference_text}"
        )
    relative_difference = check_coarse_grid(
        contribution.bragg, contribution.profile.x
    )
    print(
        "A 2x coarser 2theta grid matches the full grid to within "
        f"{relative_difference:.1e} of the maximum intensity."
    )
    return recipe


def main():
    """Run the staged refinement with and without refining asymmetry."""
    for setting_name, refine_asymmetry in ASYMMETRY_SETTINGS.items():
        recipe = refine_in_stages(setting_name, refine_asymmetry)
        recipe.plot_recipe()
    FitResults(recipe).print_results()


if __name__ == "__main__":
    main()
