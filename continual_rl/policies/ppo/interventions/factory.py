from __future__ import annotations

from .base import InterventionBase, InterventionContext
from .none import NoneIntervention
from .reset import ResetIntervention
from .partial_reinit import PartialReinitIntervention
from .gmp import GMPIntervention
from .set import SETIntervention
from .redo import ReDoIntervention


def make_intervention(name: str, ctx: InterventionContext) -> InterventionBase:
    name = (name or "dense").lower()

    if name in ("dense", "none", "baseline"):
        return NoneIntervention(ctx)
    if name in ("reset",):
        return ResetIntervention(ctx)
    if name in ("partial", "partial_reinit", "partial-reinit"):
        return PartialReinitIntervention(ctx)
    if name in ("gmp",):
        return GMPIntervention(ctx)
    if name in ("set",):
        return SETIntervention(ctx)
    if name in ("redo",):
        return ReDoIntervention(ctx)
    

    raise ValueError(f"Unknown intervention_type: {name}")
