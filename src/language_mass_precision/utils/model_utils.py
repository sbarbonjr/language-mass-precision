"""Accept process models in the forms PM4Py users already have."""

from __future__ import annotations

from typing import Any

import pm4py
from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.process_tree.obj import ProcessTree

from language_mass_precision.objects.result import CompiledModel


def as_petri_net(*model: Any) -> tuple[PetriNet, Any, Any]:
    """Normalize ``(net, im, fm)``, a 3-tuple, a ``ProcessTree`` or a ``BPMN`` to an accepting Petri net."""
    if len(model) == 1 and isinstance(model[0], tuple):
        model = model[0]
    if len(model) == 3 and isinstance(model[0], PetriNet):
        return model[0], model[1], model[2]
    if len(model) == 1 and isinstance(model[0], (ProcessTree, BPMN)):
        return pm4py.convert_to_petri_net(model[0])
    if len(model) == 1 and isinstance(model[0], PetriNet):
        raise TypeError("A Petri net needs its initial and final marking: pass (net, im, fm)")
    raise TypeError(
        "Unsupported model; expected (net, im, fm), a ProcessTree, a BPMN or a CompiledModel, got "
        + ", ".join(type(part).__name__ for part in model)
    )


def is_compiled(*model: Any) -> bool:
    return len(model) == 1 and isinstance(model[0], CompiledModel)


def visible_labels(net: PetriNet) -> frozenset[str]:
    return frozenset(t.label for t in net.transitions if t.label is not None)
