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
"""Intensity source components, which list the reflections of a
phase."""

__all__ = ["IntensitySource", "StructureIntensities", "LeBailIntensities"]

from diffpy.cmirietveld.component import Component


class IntensitySource(Component):
    """Base class of the components that list the reflections of a
    phase.

    A source gives each reflection's hkl, d-spacing and integrated
    intensity before the instrument and the corrections are applied. A
    generator holds its source under the name ``intensity_source``.
    """

    def __init__(self):
        Component.__init__(self, "intensity_source")


class StructureIntensities(IntensitySource):
    """Intensities calculated from a crystal structure.

    The multiplicity times the squared structure factor of each
    reflection comes from the structure ParameterSet of the generator.
    """


class LeBailIntensities(IntensitySource):
    """Intensities extracted iteratively from the data.

    This refines the instrument and the peak profile without a
    structure.
    """


# End of file
