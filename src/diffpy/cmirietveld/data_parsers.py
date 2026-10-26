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
"""Parsers of powder diffraction data files.

A parser loads a measured pattern into an srfit ``Profile`` with
``Profile.load_parsed_data``.
"""

__all__ = ["PowderDataParser", "GSASStdParser"]

from diffpy.srfit.fitbase import ProfileParser


class PowderDataParser(ProfileParser):
    """Base class of the parsers of powder diffraction data.

    Subclasses read one format by overriding ``_parse_metadata`` and
    ``_parse_data``. When a file has no uncertainties, they are the
    square root of the counts, floored at 1.
    """


class GSASStdParser(PowderDataParser):
    """Parser of the GSAS "STD" powder data format."""

    _format = "gsas_std"


# End of file
