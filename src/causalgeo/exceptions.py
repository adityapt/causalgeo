"""Errors raised when an analysis would not be incrementality."""


class CausalGeoError(Exception):
    """Base error."""


class MissingControlError(CausalGeoError):
    """Treatment geos were provided without a control group."""


class EmptyAssignmentError(CausalGeoError):
    """No treatment or control geos to analyze."""


class MixedRoleError(CausalGeoError):
    """A geo is listed as both treatment and control."""


class OverlapError(CausalGeoError):
    """Overlay geos collide with another test's treatment or control group."""


class MissingSpendError(CausalGeoError):
    """iCPD / iROAS requested without a spend series."""
