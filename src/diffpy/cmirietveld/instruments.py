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
"""Instrument components, which map d-spacings onto the data axis.

An instrument describes one measured dataset, so all the generators of a
dataset, one per phase, are given the same instrument object.
"""

__all__ = [
    "Instrument",
    "ConstantWavelengthInstrument",
    "TimeOfFlightInstrument",
]

from diffpy.cmirietveld.component import Component


class Instrument(Component):
    """Base class of the components that describe the instrument.

    A generator holds its instrument under the name ``instrument``, so
    the Parameters of an instrument are reached as
    ``generator.instrument.<name>``.
    """

    def __init__(self):
        Component.__init__(self, "instrument")


class ConstantWavelengthInstrument(Instrument):
    """Instrument of a constant-wavelength x-ray or neutron
    diffractometer.

    Managed Parameters
    ------------------
    zero
        The 2theta zero shift, in degrees.
    displacement
        The sample displacement correction, in degrees.
    transparency
        The sample transparency correction, in degrees.
    mu_r
        The cylinder absorption of the sample, the product of the linear
        absorption coefficient and the radius.
    """

    def set_radiation_type(self, radiation_type):
        """Set the radiation type.

        Parameters
        ----------
        radiation_type : str
            The radiation type: "X" for x-rays, "N" for neutrons or "E" for
            electrons.
        """
        raise NotImplementedError

    def set_wavelength(self, wavelength):
        """Set a single wavelength.

        Parameters
        ----------
        wavelength : float
            The wavelength in angstroms.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def set_polarization(self, polarization_rate):
        """Set the polarization rate of the x-ray beam.

        Parameters
        ----------
        polarization_rate : float
            The polarization rate used by ObjCryst, from -1 to 1.
        """
        raise NotImplementedError


class TimeOfFlightInstrument(Instrument):
    """Instrument of a time-of-flight neutron diffractometer.

    Peak positions follow t = DIFC * d + DIFA * d**2 + zero.

    Managed Parameters
    ------------------
    DIFC
        The linear conversion from d-spacing to time of flight.
    DIFA
        The quadratic conversion from d-spacing to time of flight.
    zero
        The time-of-flight zero shift.
    """


# End of file
