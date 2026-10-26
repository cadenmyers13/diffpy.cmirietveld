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
"""Peak profile components, which set the shape of each Bragg peak."""

__all__ = [
    "PeakProfile",
    "PseudoVoigtProfile",
    "TCHProfile",
    "BackToBackExponentialProfile",
]

from diffpy.cmirietveld.component import Component


class PeakProfile(Component):
    """Base class of the components that set the peak shape.

    A generator holds its peak profile under the name ``peak_profile``,
    so the Parameters of a profile are reached as
    ``generator.peak_profile.<name>``.
    """

    def __init__(self):
        Component.__init__(self, "peak_profile")


class PseudoVoigtProfile(PeakProfile):
    """Pseudo-Voigt peak profile with Caglioti widths and asymmetry.

    This is the default peak profile of a generator. It wraps pyobjcryst's
    ``ReflectionProfilePseudoVoigt``.

    Managed Parameters
    ------------------
    U, V, W
        The Caglioti peak width terms, in degrees squared.
    eta0, eta1
        The pseudo-Voigt mixing terms.
    asym0, asym1, asym2
        The peak asymmetry terms.
    """


class TCHProfile(PeakProfile):
    """Symmetric Thompson-Cox-Hastings pseudo-Voigt peak profile.

    It wraps pyobjcryst's ``ReflectionProfilePseudoVoigtTCH``, which has
    no asymmetry terms.

    Managed Parameters
    ------------------
    U, V, W
        The Gaussian width terms, in degrees squared.
    X, Y, Z
        The Lorentzian width terms, in degrees.
    P
        The Gaussian size broadening term.
    LGmix
        The mixing of the Gaussian and Lorentzian size broadening.
    """


class BackToBackExponentialProfile(PeakProfile):
    """Peak profile of a time-of-flight pattern.

    It is a pseudo-Voigt convolved with back-to-back exponentials, and
    wraps ObjCryst's ``ReflectionProfileDoubleExponentialPseudoVoigt``.
    """


# End of file
