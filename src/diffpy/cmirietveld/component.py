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
"""Base class of the components that configure a
PowderDiffractionGenerator.

A component describes one stage of the powder pattern calculation, such
as the peak profile or the instrument. It holds its refinable values as
srfit Parameters and configures the pyobjcryst objects of a generator.
"""

__all__ = ["Component"]

from diffpy.srfit.fitbase.parameterset import ParameterSet


class Component(ParameterSet):
    """Configure one stage of a PowderDiffractionGenerator.

    Subclasses write ``install`` and ``update`` for the pyobjcryst model
    they configure. They do not keep a reference to the pyobjcryst objects
    they are given, so one component can serve several generators.

    Parameters
    ----------
    name : str
        The name of the component. It must be a valid attribute name.
    """

    def install(self, pattern, diffraction):
        """Set the model up in one generator.

        The generator runs this once it has both the component and a
        structure, and again after a setting of the component changes, so
        it must be safe to repeat.

        Parameters
        ----------
        pattern : pyobjcryst.powderpattern.PowderPattern
            The pattern of the generator.
        diffraction : pyobjcryst.powderpattern.PowderPatternDiffraction
            The diffraction component of the generator's structure.
        """
        raise NotImplementedError

    def update(self, pattern, diffraction):
        """Push the Parameter values into the pyobjcryst objects.

        The generator runs this before every calculation.

        Parameters
        ----------
        pattern : pyobjcryst.powderpattern.PowderPattern
            The pattern of the generator.
        diffraction : pyobjcryst.powderpattern.PowderPatternDiffraction
            The diffraction component of the generator's structure.
        """
        raise NotImplementedError

    def create_parameters_from(self, refinable_obj, names):
        """Create the Parameters of the component from a pyobjcryst
        model.

        Each Parameter takes the default value of the matching
        ``RefinablePar`` and its limits as bounds.

        Parameters
        ----------
        refinable_obj : pyobjcryst.refinableobj.RefinableObj
            A fresh instance of the pyobjcryst model.
        names : dict of str to str
            The ``RefinablePar`` name for each Parameter name.
        """
        raise NotImplementedError

    def push_parameters(self, refinable_obj):
        """Push the Parameter values into a pyobjcryst model.

        Parameters
        ----------
        refinable_obj : pyobjcryst.refinableobj.RefinableObj
            The pyobjcryst model to update.

        Raises
        ------
        ValueError
            If a Parameter value is outside its bounds.
        """
        raise NotImplementedError


# End of file
