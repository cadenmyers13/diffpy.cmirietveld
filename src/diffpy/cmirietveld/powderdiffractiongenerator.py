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

from diffpy.srfit.fitbase import ProfileGenerator


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

    def set_structure(self, structure, name="phase"):
        """Set the structure used to calculate the pattern.

        Parameters
        ----------
        structure : pyobjcryst.crystal.Crystal
            The crystal structure.
        name : str, optional
            The name of the ParameterSet that adapts the structure
            (default "phase").
        """
        raise NotImplementedError

    def set_structure_from_parset(self, phase):
        """Set the structure from an existing ParameterSet.

        Passing the same ParameterSet to several generators, including
        srfit's PDFGenerator, shares the structure between them.

        Parameters
        ----------
        phase : ObjCrystCrystalParSet
            The ParameterSet that adapts a pyobjcryst Crystal.
        """
        raise NotImplementedError

    def set_peak_profile(self, peak_profile):
        """Set the peak profile component.

        Parameters
        ----------
        peak_profile : PeakProfile
            The peak profile. It can be shared with other generators.

        Raises
        ------
        ValueError
            If ``peak_profile`` is not a PeakProfile.
        """
        raise NotImplementedError

    def set_instrument(self, instrument):
        """Set the instrument component.

        Parameters
        ----------
        instrument : Instrument
            The instrument. All the generators of one dataset share it.

        Raises
        ------
        ValueError
            If ``instrument`` is not an Instrument.
        """
        raise NotImplementedError

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

        Parameters
        ----------
        tth : ndarray
            The 2theta values in degrees at which to calculate the
            pattern.

        Returns
        -------
        ndarray
            The calculated Bragg intensity at each 2theta value.
        """
        raise NotImplementedError


# End of file
