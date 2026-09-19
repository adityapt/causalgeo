"""Intervention types — same vocabulary as Meridian GeoX."""

from enum import Enum


class Intervention(str, Enum):
    """What changes in the treatment group during the test period.

    Aligns with GeoX experiment types:

    * ``HOLDBACK`` — new activity in treatment; control stays at zero spend.
    * ``GO_DARK`` — existing activity shut off in treatment; control stays BAU.
    * ``HEAVY_UP`` — incremental activity in treatment; control stays BAU.
    """

    HOLDBACK = "holdback"
    GO_DARK = "go_dark"
    HEAVY_UP = "heavy_up"


# GeoLift-style aliases
GO_DARK = Intervention.GO_DARK
HOLDBACK = Intervention.HOLDBACK
HEAVY_UP = Intervention.HEAVY_UP
