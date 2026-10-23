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
"""Rietveld profile generator.

The RietveldGenerator class calculates the Bragg intensity of a powder
diffraction pattern from a crystal structure with pyobjcryst. It mirrors
srfit's PDFGenerator, so a structure ParameterSet can be shared between
RietveldGenerators for several datasets and a PDFGenerator.
"""

__all__ = ["RietveldGenerator"]

import numpy as np
from pyobjcryst.powderpattern import PowderPattern, ReflectionProfileType
from pyobjcryst.radiation import RadiationType
from pyobjcryst.refinableobj import RefinablePar

from diffpy.srfit.exceptions import SrFitError
from diffpy.srfit.fitbase import ProfileGenerator
from diffpy.srfit.fitbase.parameter import ParameterAdapter
from diffpy.srfit.structure import struToParameterSet

RADIATION_TYPES = {
    "X": RadiationType.RAD_XRAY,
    "N": RadiationType.RAD_NEUTRON,
    "E": RadiationType.RAD_ELECTRON,
}

PATTERN_PARAMETER_NAMES = {
    "scale": "Scale_",
    "zero": "Zero",
    "displacement": "2ThetaDispl",
    "transparency": "2ThetaTransp",
}

PROFILE_PARAMETER_NAMES = {
    "U": "U",
    "V": "V",
    "W": "W",
    "eta0": "Eta0",
    "eta1": "Eta1",
    "asym0": "Asym0",
    "asym1": "Asym1",
    "asym2": "Asym2",
}

PLACEHOLDER_TTH = np.linspace(5.0, 175.0, 1000)


class RietveldGenerator(ProfileGenerator):
    """Calculate the Bragg intensity of a powder pattern with
    pyobjcryst.

    The generator holds a pyobjcryst PowderPattern with one diffraction
    component for its structure. The managed Parameters are not created
    until the structure is set, as in srfit's PDFGenerator. Angles are in
    degrees and the Caglioti U, V, W in degrees squared; pyobjcryst stores
    them internally in radians.

    The peak profile is pyobjcryst's pseudo-Voigt, whose width follows the
    Caglioti formula and whose asymmetry correction depends on the 2theta
    grid spacing, so the pattern should be calculated on the data grid.
    The reflection list is regenerated at every calculation, because
    pyobjcryst does not update it by itself when the 2theta grid, the
    wavelength or the lattice changes. This adds about 0.6 ms to a 1.4 ms
    calculation for PbSO4.
    The generator calculates the Bragg intensity only; a background can be
    added in the FitContribution equation.

    Parameters
    ----------
    name : str, optional
        The name of the generator (default "rietveld"). It must be a valid
        attribute name.

    Attributes
    ----------
    phase : ParameterSet or None
        The structure ParameterSet, or None until a structure is set.

    Managed Parameters
    ------------------
    scale
        The scale factor.
    zero
        The 2theta zero shift, in degrees.
    displacement
        The sample displacement correction, in degrees.
    transparency
        The sample transparency correction, in degrees.
    U, V, W
        The Caglioti peak width terms, in degrees squared.
    eta0, eta1
        The pseudo-Voigt mixing terms.
    asym0, asym1, asym2
        The peak asymmetry terms.
    """

    def __init__(self, name="rietveld"):
        ProfileGenerator.__init__(self, name)
        self.phase = None
        self._pattern = PowderPattern()
        self._diffraction = None
        self._last_tth = np.empty(0)
        self._set_tth(PLACEHOLDER_TTH)

    def set_structure(self, structure, name="phase"):
        """Set the structure used to calculate the pattern.

        This adapts the structure to an srfit ParameterSet with
        ``struToParameterSet``, which this generator manages.

        Parameters
        ----------
        structure : pyobjcryst.crystal.Crystal
            The crystal structure.
        name : str, optional
            The name of the ParameterSet that adapts the structure (default
            "phase").
        """
        phase = struToParameterSet(name, structure)
        self.set_structure_from_parset(phase)

    def set_structure_from_parset(self, phase):
        """Set the structure from an existing ParameterSet.

        Passing the same ParameterSet to several generators, including
        srfit's PDFGenerator through ``setPhase``, shares the structure
        between them.

        Parameters
        ----------
        phase : ObjCrystCrystalParSet
            The ParameterSet that adapts a pyobjcryst Crystal.
        """
        self.phase = phase
        self.add_parameter_set(phase)
        self._diffraction = self._pattern.AddPowderPatternDiffraction(
            phase.stru
        )
        self._diffraction.SetReflectionProfilePar(
            ReflectionProfileType.PROFILE_PSEUDO_VOIGT
        )
        for parameter_name, objcryst_name in PATTERN_PARAMETER_NAMES.items():
            self._add_objcryst_parameter(
                parameter_name, self._pattern.GetPar(objcryst_name)
            )
        profile = self._diffraction.GetProfile()
        for parameter_name, objcryst_name in PROFILE_PARAMETER_NAMES.items():
            self._add_objcryst_parameter(
                parameter_name, profile.GetPar(objcryst_name)
            )

    def set_radiation_type(self, radiation_type):
        """Set the radiation type.

        Parameters
        ----------
        radiation_type : str
            The radiation type: "X" for x-rays, "N" for neutrons or "E" for
            electrons.

        Raises
        ------
        ValueError
            If the radiation type is not supported.
        """
        if radiation_type not in RADIATION_TYPES:
            raise ValueError(
                f"'{radiation_type}' is not a supported radiation type. "
                "Please use 'X' for x-rays, 'N' for neutrons or 'E' for "
                "electrons."
            )
        self._pattern.SetRadiationType(RADIATION_TYPES[radiation_type])
        self._flush(other=(self,))

    def get_radiation_type(self):
        """Return the radiation type.

        Returns
        -------
        str
            The radiation type: "X", "N" or "E".
        """
        pyobjcryst_type = self._pattern.GetRadiationType()
        radiation_type = next(
            name
            for name, value in RADIATION_TYPES.items()
            if value == pyobjcryst_type
        )
        return radiation_type

    def set_wavelength(self, wavelength):
        """Set a single wavelength.

        Parameters
        ----------
        wavelength : float
            The wavelength in angstroms.
        """
        self._pattern.SetWavelength(wavelength)
        self._flush(other=(self,))

    def set_xray_tube(self, element, kalpha2_ratio=0.5):
        """Set the wavelengths of an x-ray tube's Kalpha1/Kalpha2
        doublet.

        Parameters
        ----------
        element : str
            The anode element, for example "Cu" or "Mo".
        kalpha2_ratio : float, optional
            The intensity ratio of Kalpha2 to Kalpha1 (default 0.5).
        """
        self._pattern.SetWavelength(element, kalpha2_ratio)
        self._flush(other=(self,))

    def get_wavelength(self):
        """Return the wavelength.

        Returns
        -------
        float
            The wavelength in angstroms. For an x-ray tube this is the
            intensity-weighted mean of the Kalpha doublet.
        """
        wavelength = self._pattern.GetWavelength()
        return wavelength

    def __call__(self, tth):
        """Calculate the Bragg intensity.

        Parameters
        ----------
        tth : ndarray
            The 2theta values in degrees at which to calculate the pattern.

        Returns
        -------
        ndarray
            The calculated Bragg intensity at each 2theta value.
        """
        if not np.array_equal(tth, self._last_tth):
            self._set_tth(tth)
        self._pattern.Prepare()
        bragg_intensity = np.array(self._pattern.GetPowderPatternCalc())
        return bragg_intensity

    def _add_objcryst_parameter(self, name, refinable_par):
        self.add_parameter(
            ParameterAdapter(
                name,
                refinable_par,
                getter=RefinablePar.GetHumanValue,
                setter=RefinablePar.SetHumanValue,
            )
        )

    def _set_tth(self, tth):
        self._last_tth = np.array(tth, dtype=float)
        self._pattern.SetPowderPatternX(np.radians(self._last_tth))
        self._pattern.SetPowderPatternObs(np.zeros_like(self._last_tth))

    def _validate(self):
        """Validate the generator.

        Raises
        ------
        SrFitError
            If no structure has been set, or the ProfileGenerator
            validation fails.
        """
        if self.phase is None:
            raise SrFitError(
                "The generator has no structure. Please set one with "
                "set_structure or set_structure_from_parset."
            )
        ProfileGenerator._validate(self)
