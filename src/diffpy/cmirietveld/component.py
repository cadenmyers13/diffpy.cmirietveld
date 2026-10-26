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

from diffpy.srfit.fitbase.parameter import Parameter
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

    def __init__(self, name):
        ParameterSet.__init__(self, name)
        self._refinable_par_names = {}

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
        ``RefinablePar`` and its limits as bounds. ObjCryst silently clamps
        a value outside its limits, so the bounds keep a refinement within
        the values ObjCryst actually uses.

        The Parameter names are mapped to the ``RefinablePar`` names
        rather than copied, so the names users see are this package's,
        independent of ObjCryst's. Some ObjCryst names are not valid
        Parameter names (``2ThetaDispl``). The mapping keeps ObjCryst
        parameters out of refinement scripts. The mapping is stored, so
        ``push_parameters`` writes each value back to the same
        ``RefinablePar``.

        Parameters
        ----------
        refinable_obj : pyobjcryst.refinableobj.RefinableObj
            A fresh instance of the pyobjcryst model.
        names : dict of str to str
            The ``RefinablePar`` name for each Parameter name, for example
            ``{"eta0": "Eta0"}``. Only the named ``RefinablePar``s get a
            Parameter.

        Raises
        ------
        ValueError
            If the model has no ``RefinablePar`` with one of the names.

        Examples
        --------
        In the ``__init__`` of a peak profile:

        >>> self.create_parameters_from(
        ...     ReflectionProfilePseudoVoigt(),
        ...     {"U": "U", "V": "V", "W": "W", "eta0": "Eta0"},
        ... )
        """
        available_names = [
            refinable_obj.GetPar(index).GetName()
            for index in range(refinable_obj.GetNbPar())
        ]
        for refinable_par_name in names.values():
            if refinable_par_name not in available_names:
                raise ValueError(
                    f"{type(refinable_obj).__name__} has no refinable "
                    f"parameter named '{refinable_par_name}'. Please use one "
                    f"of its parameter names: {', '.join(available_names)}."
                )
        for parameter_name, refinable_par_name in names.items():
            refinable_par = refinable_obj.GetPar(refinable_par_name)
            parameter = Parameter(
                parameter_name, refinable_par.GetHumanValue()
            )
            if refinable_par.IsLimited():
                parameter.bound_range(
                    refinable_par.GetHumanMin(), refinable_par.GetHumanMax()
                )
            self.add_parameter(parameter)
        self._refinable_par_names.update(names)

    def push_parameters(self, refinable_obj):
        """Push the Parameter values into a pyobjcryst model.

        Each value is written to the ``RefinablePar`` that
        ``create_parameters_from`` mapped it to.

        A value outside the limits of its ``RefinablePar`` raises an error
        instead of being clamped, because ObjCryst would otherwise use a
        different value from the one the Parameter reports. The check uses
        ObjCryst's limits, not the Parameter's bounds: a value outside
        narrowed bounds is still pushed, and widened bounds don't let a
        value past a limit. An optimizer run without the recipe's bounds
        stops with this error the first time it steps past a limit.

        All the values are checked before any is pushed, so an error
        leaves the model unchanged rather than half updated.

        Parameters
        ----------
        refinable_obj : pyobjcryst.refinableobj.RefinableObj
            The pyobjcryst model to update.

        Raises
        ------
        ValueError
            If a Parameter value is outside the limits of its
            ``RefinablePar``.
        """
        values = {
            parameter_name: self.get(parameter_name).value
            for parameter_name in self._refinable_par_names
        }
        refinable_pars = {
            parameter_name: refinable_obj.GetPar(refinable_par_name)
            for parameter_name, refinable_par_name in (
                self._refinable_par_names.items()
            )
        }
        for parameter_name, refinable_par in refinable_pars.items():
            if not refinable_par.IsLimited():
                continue
            value = values[parameter_name]
            lower_limit = refinable_par.GetHumanMin()
            upper_limit = refinable_par.GetHumanMax()
            if not lower_limit <= value <= upper_limit:
                used_value = min(max(value, lower_limit), upper_limit)
                raise ValueError(
                    f"The value of '{parameter_name}', {value}, is outside "
                    f"its limits [{lower_limit}, {upper_limit}]. ObjCryst "
                    f"would silently use {used_value} instead, so the refined "
                    "value would not be the one used to calculate the "
                    "pattern. Please keep the value within the limits, for "
                    "example by passing the recipe's bounds to the optimizer: "
                    "least_squares(..., bounds=recipe.get_bounds_array())."
                )
        for parameter_name, refinable_par in refinable_pars.items():
            refinable_par.SetHumanValue(values[parameter_name])


# End of file
