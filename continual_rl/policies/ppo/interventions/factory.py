from __future__ import annotations

from .base import InterventionBase, InterventionContext
from .none import NoneIntervention
from .reset import ResetIntervention
from .partial_reinit import PartialReinitIntervention
from .gmp import GMPIntervention
from .set import SETIntervention
from .redo import ReDoIntervention

from .dormancy_monitor import DormancyMonitorIntervention
from .composite import CompositeIntervention


def make_intervention(name: str, ctx: InterventionContext) -> InterventionBase:
    name = (name or "dense").lower()

    # base intervention (the one that actually changes learning)
    if name in ("dense", "none", "baseline"):
        base = NoneIntervention(ctx)
    elif name in ("reset",):
        base = ResetIntervention(ctx)
    elif name in ("partial", "partial_reinit", "partial-reinit"):
        base = PartialReinitIntervention(ctx)
    elif name in ("gmp",):
        base = GMPIntervention(ctx)
    elif name in ("set",):
        base = SETIntervention(ctx)
    elif name in ("redo",):
        # ReDo already computes the same dormancy metric internally
        # so we avoid double-hooks by returning it as-is.
        return ReDoIntervention(ctx)
    else:
        raise ValueError(f"Unknown intervention_type: {name}")

    # wrap with dormancy monitor to track dormant fraction
    monitor = DormancyMonitorIntervention(ctx)
    return CompositeIntervention(ctx, [monitor, base])
