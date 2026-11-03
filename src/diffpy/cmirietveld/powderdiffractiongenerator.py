#!/usr/bin/env python
##############################################################################
#
# (c) 2026 diffpy.cmirietveld contributors.
# All rights reserved.
#
# File coded by: Caden Myers, Simon Billinge, Billinge Group members.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/diffpy/diffpy.cmirietveld/graphs/contributors
#
# See LICENSE.rst for license information.
#
##############################################################################
"""Powder diffraction profile generator.

The PowderDiffractionGenerator class calculates the Bragg intensity of
one phase of a powder diffraction pattern with pyobjcryst. It mirrors
srfit's PDFGenerator, so a structure ParameterSet can be shared between
PowderDiffractionGenerators for several datasets and a PDFGenerator.
"""

__all__ = ["PowderDiffractionGenerator"]

import warnings
from pathlib import Path

import numpy as np
from pyobjcryst import ObjCrystException
from pyobjcryst.crystal import Crystal, create_crystal_from_cif
from pyobjcryst.powderpattern import PowderPattern

from diffpy.cmirietveld.instruments import (
    ConstantWavelengthInstrument,
    Instrument,
)
from diffpy.cmirietveld.peak_profiles import PeakProfile, PseudoVoigtProfile
from diffpy.cmistructure import structure_to_parameter_set
from diffpy.cmistructure.basestructureparset import BaseStructureParSet
from diffpy.srfit.exceptions import SrFitError
from diffpy.srfit.fitbase import ProfileGenerator

PLACEHOLDER_TTH = np.linspace(5.0, 175.0, 1000)

COMPONENT_STAGES = {
    "instrument": (Instrument, "instrument", "ConstantWavelengthInstrument"),
    "peak_profile": (
        PeakProfile,
        "peak profile",
        "PseudoVoigtProfile or TCHProfile",
    ),
}


class PowderDiffractionGenerator(ProfileGenerator):
    """Calculate the Bragg intensity of one phase of a powder pattern
    with pyobjcryst.

    The generator is assembled from components, one per stage of the
    calculation: an intensity source, an instrument, zero or more
    intensity corrections and a peak profile. The Parameters of each
    component sit under its name, for example
    ``generator.peak_profile.U`` and ``generator.instrument.zero``. With
    nothing set, the generator uses a ``PseudoVoigtProfile`` and a
    ``ConstantWavelengthInstrument``.

    The generator calculates the Bragg intensity only; a background is
    added in the FitContribution equation.

    Parameters
    ----------
    name : str, optional
        The name of the generator (default "bragg", the name used in
        FitContribution equations such as "bragg + background"). It must
        be a valid attribute name.

    Attributes
    ----------
    phase : ParameterSet or None
        The structure ParameterSet, or None until a structure is set.
    peak_profile : PeakProfile
        The peak profile component.
    instrument : Instrument
        The instrument component.

    Managed Parameters
    ------------------
    scale
        The scale factor, applied to the calculated pattern.
    """

    def __init__(self, name="bragg"):
        ProfileGenerator.__init__(self, name)
        self.phase = None
        self._pattern = None
        self._diffraction = None
        self._last_tth = np.empty(0)
        self._installed_settings = {}
        self._new_parameter("scale", 1.0)
        self.add_parameter_set(ConstantWavelengthInstrument())
        self.add_parameter_set(PseudoVoigtProfile())

    def set_structure(self, structure, name=None):
        """Set the structure used to calculate the pattern.

        The structure is a pyobjcryst Crystal, the path to a CIF file, or a
        ParameterSet that adapts a Crystal. A Crystal or a path is adapted
        to a new ParameterSet, and a path is first read into a new Crystal.
        A ParameterSet is used as it is, with its own name.

        Giving the same ParameterSet to several generators shares the
        structure between them. Giving them the same path does not: each
        call reads the file again, so each generator refines its own
        structure.

        Parameters
        ----------
        structure : Crystal, pathlib.Path, str or ObjCrystCrystalParSet
            The pyobjcryst Crystal, the path to a CIF file holding it, or
            the diffpy.cmistructure ParameterSet that adapts it.
        name : str, optional
            The name of the ParameterSet made for a Crystal or a path
            (default "phase"). With a ParameterSet, it must match the
            ParameterSet's own name.

        Raises
        ------
        FileNotFoundError
            If ``structure`` is a path to a file that does not exist.
        ValueError
            If ``structure`` is a path to a file that isn't a CIF file
            holding a crystal structure, a structure or ParameterSet that
            ObjCryst can't calculate with, or a ParameterSet given with a
            ``name`` other than its own.

        Warns
        -----
        UserWarning
            If a Parameter of the replaced structure is a variable of a fit
            recipe or constrained in one, since the recipe would refine a
            Parameter that no longer changes the pattern.
        """
        if isinstance(structure, BaseStructureParSet):
            if name is not None and name != structure.name:
                raise ValueError(
                    f"set_structure was given the name '{name}' with a "
                    "ParameterSet, which keeps its own name, "
                    f"'{structure.name}'. Please leave out the name, or give "
                    "it when you create the ParameterSet."
                )
            crystal = structure.structure
        elif isinstance(structure, (str, Path)):
            crystal = _read_crystal_from_cif(Path(structure))
        else:
            crystal = structure
        if not isinstance(crystal, Crystal):
            raise ValueError(
                f"set_structure was given a {type(structure).__name__}, "
                "which the powder diffraction generator can't calculate "
                "with. Please give a pyobjcryst Crystal, the path to a CIF "
                "file, or a ParameterSet made from a Crystal."
            )
        if isinstance(structure, BaseStructureParSet):
            phase = structure
        else:
            phase = structure_to_parameter_set(name or "phase", crystal)
        self._replace_phase(phase, "set_structure")

    def set_structure_from_parset(self, phase):
        """Set the structure from an existing ParameterSet.

        This is the same as ``set_structure`` with a ParameterSet. It is
        kept so the generator has the same methods as
        diffpy.cmipdf's PDF generators.

        Parameters
        ----------
        phase : ObjCrystCrystalParSet
            The diffpy.cmistructure ParameterSet that adapts a pyobjcryst
            Crystal.

        Raises
        ------
        ValueError
            If ``phase`` does not adapt a pyobjcryst Crystal.

        Warns
        -----
        UserWarning
            If a Parameter of the replaced structure is a variable of a fit
            recipe or constrained in one.
        """
        if not isinstance(phase, BaseStructureParSet) or not isinstance(
            phase.structure, Crystal
        ):
            raise ValueError(
                f"set_structure_from_parset was given a "
                f"{type(phase).__name__}, which the powder diffraction "
                "generator can't calculate with. Please give a ParameterSet "
                "made from a pyobjcryst Crystal."
            )
        self._replace_phase(phase, "set_structure_from_parset")

    def set_peak_profile(self, peak_profile):
        """Set the peak profile component.

        The peak profile replaces the one set before, with its Parameters
        under ``peak_profile``. It is installed once the generator has a
        structure, so it can be set before or after the structure.

        Parameters
        ----------
        peak_profile : PeakProfile
            The peak profile. It can be shared with other generators.

        Raises
        ------
        ValueError
            If ``peak_profile`` is not a PeakProfile.

        Warns
        -----
        UserWarning
            If a Parameter of the replaced peak profile is a variable of a
            fit recipe or constrained in one, since the recipe would refine
            a Parameter that no longer changes the pattern.
        """
        self._set_component("peak_profile", peak_profile)

    def set_instrument(self, instrument):
        """Set the instrument component.

        The instrument replaces the one set before, with its Parameters
        under ``instrument``. It is installed once the generator has a
        structure, so it can be set before or after the structure.

        Parameters
        ----------
        instrument : Instrument
            The instrument. All the generators of one dataset share it.

        Raises
        ------
        ValueError
            If ``instrument`` is not an Instrument.

        Warns
        -----
        UserWarning
            If a Parameter of the replaced instrument is a variable of a
            fit recipe or constrained in one, since the recipe would refine
            a Parameter that no longer changes the pattern.
        """
        self._set_component("instrument", instrument)

    def add_intensity_correction(self, correction):
        """Add an intensity correction component.

        Parameters
        ----------
        correction : IntensityCorrection
            The correction. Its Parameters sit under its name.

        Raises
        ------
        ValueError
            If ``correction`` is not an IntensityCorrection.
        """
        raise NotImplementedError

    def set_intensity_source(self, intensity_source):
        """Set the intensity source component.

        Parameters
        ----------
        intensity_source : IntensitySource
            The intensity source, which replaces the structure.

        Raises
        ------
        ValueError
            If ``intensity_source`` is not an IntensitySource.
        """
        raise NotImplementedError

    def __call__(self, tth):
        """Calculate the Bragg intensity.

        Each component whose settings changed since it was last installed
        is installed again. Then each component pushes its Parameter
        values into pyobjcryst, and the pattern is calculated and
        multiplied by ``scale``.

        Parameters
        ----------
        tth : ndarray
            The 2theta values in degrees at which to calculate the
            pattern.

        Returns
        -------
        ndarray
            The calculated Bragg intensity at each 2theta value.

        Raises
        ------
        SrFitError
            If no structure is set, or the instrument has no wavelength.
        ValueError
            If a component's Parameter value is outside the limits of its
            ObjCryst ``RefinablePar``.
        """
        self._check_structure()
        components = [self.get(stage_name) for stage_name in COMPONENT_STAGES]
        for component in components:
            component._validate()
        if not np.array_equal(tth, self._last_tth):
            self._set_tth(tth)
        for component in components:
            installed_component, installed_version = (
                self._installed_settings.get(component.name, (None, None))
            )
            if (
                installed_component is not component
                or installed_version != component.settings_version
            ):
                component.install(self._pattern, self._diffraction)
                self._installed_settings[component.name] = (
                    component,
                    component.settings_version,
                )
        for component in components:
            component.update(self._pattern, self._diffraction)
        self._pattern.Prepare()
        calculated_pattern = np.array(self._pattern.GetPowderPatternCalc())
        bragg_intensity = self.scale.value * calculated_pattern
        return bragg_intensity

    def _validate(self):
        self._check_structure()
        ProfileGenerator._validate(self)

    def _check_structure(self):
        if self.phase is None:
            raise SrFitError(
                "The generator has no structure. Please set one with "
                "set_structure or set_structure_from_parset."
            )

    def _replace_phase(self, phase, setter_name):
        if self.phase is not None:
            _warn_if_parameters_used(self.phase, setter_name, "structure")
            self.remove_parameter_set(self.phase)
        self.phase = phase
        self.add_parameter_set(phase)
        self._pattern = PowderPattern()
        self._set_tth(PLACEHOLDER_TTH)
        self._diffraction = self._pattern.AddPowderPatternDiffraction(
            phase.structure
        )
        self._installed_settings = {}
        self._flush(other=(self,))

    def _set_component(self, stage_name, component):
        base_class, kind, examples = COMPONENT_STAGES[stage_name]
        setter_name = f"set_{stage_name}"
        if not isinstance(component, base_class):
            given_name = type(component).__name__
            raise ValueError(
                f"{setter_name} was given {_get_article(given_name)} "
                f"{given_name}, which is not {_get_article(kind)} {kind}. "
                f"Please give {_get_article(base_class.__name__)} "
                f"{base_class.__name__}, such as {examples}."
            )
        old_component = self.get(stage_name)
        _warn_if_parameters_used(old_component, setter_name, kind)
        self.remove_parameter_set(old_component)
        self.add_parameter_set(component)
        self._flush(other=(self,))

    def _set_tth(self, tth):
        self._last_tth = np.array(tth, dtype=float)
        self._pattern.SetPowderPatternX(np.radians(self._last_tth))
        self._pattern.SetPowderPatternObs(np.zeros_like(self._last_tth))


def _read_crystal_from_cif(cif_path):
    # create_crystal_from_cif takes no Path, and raises ObjCrystException
    # without the file name for a file with no structure
    try:
        crystal = create_crystal_from_cif(str(cif_path))
    except ObjCrystException:
        raise ValueError(
            f"No crystal structure could be read from {cif_path}. "
            "Please give a CIF file that holds a crystal structure."
        ) from None
    return crystal


def _warn_if_parameters_used(parameter_set, setter_name, kind):
    # A Parameter is observed by its ParameterSet alone, unless a recipe
    # variable proxies it, which adds the recipe's observer
    used_names = []
    for parameter in parameter_set.iterate_over_parameters():
        if len(parameter._observers) > 1 or parameter.constrained:
            if parameter.name not in used_names:
                used_names.append(parameter.name)
    if not used_names:
        return
    warnings.warn(
        f"{setter_name} replaced {_get_article(kind)} {kind} whose "
        f"Parameters are used in a fit recipe: {', '.join(used_names)}. "
        "The recipe would still refine them, but they no longer change the "
        f"pattern. Please call {setter_name} before adding the {kind}'s "
        "Parameters to the recipe.",
        UserWarning,
        stacklevel=4,
    )


def _get_article(word):
    article = "an" if word[0].lower() in "aeiou" else "a"
    return article


# End of file
