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
"""Parsers of instrument parameter files.

An instrument is calibrated once on a standard and then reused for every
sample measured on it. A parser reads such a calibration and creates the
components that carry it.
"""

__all__ = [
    "InstrumentParameterParser",
    "GSASInstrumentParser",
    "GSASIIInstrumentParser",
]


class InstrumentParameterParser:
    """Base class of the parsers of instrument parameter files.

    Subclasses read one format and convert its terms to the conventions
    of ObjCryst. A term with no ObjCryst counterpart gives a warning
    that names it.
    """

    def parse_file(self, path):
        """Parse an instrument parameter file.

        Parameters
        ----------
        path : str or Path
            The path to the instrument parameter file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not in the format of the parser.
        """
        raise NotImplementedError

    def get_instrument(self):
        """Return the instrument described by the file.

        Returns
        -------
        Instrument
            The instrument, with the wavelengths, zero and polarization
            of the file. The phases of a dataset share it.
        """
        raise NotImplementedError

    def get_peak_profile(self):
        """Return a new peak profile with the values of the file.

        Returns
        -------
        PeakProfile
            The peak profile. Each call returns a new object, so each
            phase gets its own, starting from the same values.
        """
        raise NotImplementedError


class GSASInstrumentParser(InstrumentParameterParser):
    """Parser of the GSAS instrument parameter file, ``.prm``."""


class GSASIIInstrumentParser(InstrumentParameterParser):
    """Parser of the GSAS-II instrument parameter file, ``.instprm``."""


# End of file
