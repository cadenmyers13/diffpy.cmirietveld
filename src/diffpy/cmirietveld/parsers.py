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
"""Parsers of powder diffraction data files and instrument parameter
files.

A data parser loads a measured pattern into an srfit ``Profile`` with
``Profile.load_parsed_data``.

An instrument is calibrated once on a standard and then reused for every
sample measured on it. An instrument parameter parser reads such a
calibration and creates the components that carry it.
"""

__all__ = [
    "PowderDataParser",
    "GSASDataParser",
    "InstrumentParameterParser",
    "GSASInstrumentParser",
    "GSASIIInstrumentParser",
]

from diffpy.srfit.fitbase import ProfileParser


class PowderDataParser(ProfileParser):
    """Base class of the parsers of powder diffraction data.

    Subclasses read one format by overriding ``_parse_metadata`` and
    ``_parse_data``. The metadata hold the type of the x axis under
    ``"xtype"``, for example ``"tth"`` for 2theta in degrees.

    Uncertainties given in a file are read as they are. srfit gives a
    point whose uncertainty is zero, negative, infinite or not a number
    no weight in a fit.

    When a file has no uncertainties, they come from counting
    statistics: the square root of the counts, divided by the number of
    counters for counts averaged over several counters. They are floored
    at 1, so that a point with zero counts keeps its weight. Negative
    counts, for example from background-subtracted data, get the square
    root of their absolute value, so that a negative point is weighted
    like a positive one of the same size. For data that aren't raw
    counts, such as normalized synchrotron data, these uncertainties are
    only approximate, and known ones can be set with
    ``Profile.set_observed_profile``.
    """

    def _compute_counting_uncertainty(self, counts, n_counters=1):
        """Return the counting uncertainty of counts averaged over
        ``n_counters`` counters, floored at 1."""
        raise NotImplementedError


class GSASDataParser(PowderDataParser):
    """Parser of GSAS powder data files.

    The first line of the file is its title, and lines before the first
    ``BANK`` line, such as comments, are skipped. Each bank starts with a
    line ``BANK <number> <points> <records> CONST <start> <step> 0 0
    <format>``, with the start and step in centidegrees, and is followed
    by its data in fields of eight characters, in one of two formats:

    - ``STD``: the counts, ten to a line. Each field holds the number of
      counters averaged (two characters, blank for one) and the counts
      averaged over them (six characters), which are the counts read.
      The uncertainties come from counting statistics.
    - ``ESD``: the counts and their uncertainty, five points to a line,
      read as they are.

    The x axis is 2theta in degrees. Only constant-step STD and ESD banks
    are read, and a file can mix the two. Files with CRLF line endings
    are read like files with LF ones.

    Examples
    --------
    Load the PbSO4 x-ray data into a ``Profile``::

        parser = GSASDataParser()
        parser.parse_file(Path("PBSO4.XRA"))
        profile = Profile()
        profile.load_parsed_data(parser)
    """

    _format = "gsas"

    def _parse_metadata(self, filename):
        """Return the metadata of a GSAS powder data file."""
        raise NotImplementedError

    def _parse_data(self, filename, column_format=None, **kwargs):
        """Append the STD and ESD banks of a GSAS powder data file."""
        raise NotImplementedError


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
