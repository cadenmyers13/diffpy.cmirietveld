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
"""Background calculators for the equation of a FitContribution.

A background describes a measured dataset rather than a phase. It is
registered in a FitContribution with ``register_calculator`` and added
to the Bragg intensity in the equation, for example ``"bragg +
background"``.
"""

__all__ = ["Background", "ChebyshevBackground"]

from diffpy.srfit.fitbase import Calculator


class Background(Calculator):
    """Base class of the backgrounds.

    Subclasses write ``__call__`` in NumPy and create their Parameters.

    Parameters
    ----------
    name : str
        The name of the background. It must be a valid attribute name.
    """

    def __call__(self, x):
        """Calculate the background.

        Parameters
        ----------
        x : ndarray
            The values of the data axis.

        Returns
        -------
        ndarray
            The background at each value of the data axis.
        """
        raise NotImplementedError


class ChebyshevBackground(Background):
    """Background described by a Chebyshev polynomial.

    The range of x is mapped onto [-1, 1], so changing the calculation
    range changes what the coefficients mean.

    Parameters
    ----------
    name : str
        The name of the background. It must be a valid attribute name.
    n_terms : int, optional
        The number of Chebyshev terms (default 6).

    Managed Parameters
    ------------------
    b0, b1, ...
        The Chebyshev coefficients, listed in order by ``coefficients``.
    """

    def __init__(self, name, n_terms=6):
        Background.__init__(self, name)
        self.n_terms = n_terms

    @property
    def coefficients(self):
        """The Parameters of the Chebyshev coefficients, in order."""
        raise NotImplementedError


# End of file
