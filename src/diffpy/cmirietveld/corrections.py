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
"""Intensity correction components, which scale each reflection."""

__all__ = ["IntensityCorrection", "MarchDollaseCorrection"]

from diffpy.cmirietveld.component import Component


class IntensityCorrection(Component):
    """Base class of the components that multiply the intensity of each
    reflection by a factor.

    A generator can hold several corrections. Each one is named by the
    user, so its Parameters are reached as
    ``generator.<name>.<parameter>``.

    Parameters
    ----------
    name : str
        The name of the correction. It must be a valid attribute name.
    """


class MarchDollaseCorrection(IntensityCorrection):
    """Preferred orientation described by the March-Dollase model.

    Parameters
    ----------
    name : str
        The name of the correction. It must be a valid attribute name.
    direction : tuple of int
        The Miller indices (h, k, l) of the preferred orientation.
    r : float, optional
        The starting value of the March coefficient (default 1.0, which
        is no preferred orientation).

    Managed Parameters
    ------------------
    r
        The March coefficient, bounded to [0.1, 10].
    """

    def __init__(self, name, direction, r=1.0):
        IntensityCorrection.__init__(self, name)
        self.direction = tuple(direction)
        self._start_r = r


# End of file
