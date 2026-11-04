"""Bounded-playout LMP baseline.

Enumerates ``L_{<=K}(M)`` with PM4Py's extensive playout and evaluates the
quotient on that finite set. The result ignores every trace longer than
``K``, so it is a truncated value with no certificate; it exists to measure
truncation error against the exact variant.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from pm4py.algo.simulation.playout.petri_net import algorithm as petri_playout
from pm4py.algo.simulation.playout.petri_net.variants import extensive

from language_mass_precision.algo.denominator import CERT_BOUNDED
from language_mass_precision.objects.automaton import Trace
from language_mass_precision.objects.priors import TracePrior
from language_mass_precision.objects.result import LMPResult, Status

DEFAULT_MAX_TRACE_LENGTH = 10


def bounded_language(net: Any, initial_marking: Any, final_marking: Any, max_trace_length: int) -> set[Trace]:
    """All visible traces of length ``<= max_trace_length`` that reach the final marking."""
    generated = petri_playout.apply(
        net,
        initial_marking,
        final_marking=final_marking,
        variant=petri_playout.Variants.EXTENSIVE,
        parameters={extensive.Parameters.MAX_TRACE_LENGTH: max_trace_length},
    )
    return {tuple(event["concept:name"] for event in trace) for trace in generated}


def apply_to_language(
    variants: Iterable[Trace],
    language: set[Trace],
    prior: TracePrior,
    max_trace_length: int,
) -> LMPResult:
    """Bounded LMP on an explicit finite language ``L_{<=K}(M)``."""
    started = time.perf_counter()
    observed = set(variants)
    fitting = observed & language
    n_mass = prior.language_mass(fitting)
    d_mass = prior.language_mass(language)
    common = {
        "certification": CERT_BOUNDED,
        "numerator_mass": n_mass,
        "denominator_lower": d_mass,
        "denominator_upper": d_mass,
        "accepted_variants": len(fitting),
        "rejected_variants": len(observed - language),
        "truncation_k": max_trace_length,
        "prior": prior.describe(),
    }
    if d_mass <= 0.0:
        return LMPResult(status=Status.UNDEFINED, reason="empty_bounded_language",
                         elapsed_seconds=time.perf_counter() - started, **common)
    value = min(n_mass / d_mass, 1.0)
    return LMPResult(status=Status.BOUNDED, value=value, lower=value, upper=value,
                     elapsed_seconds=time.perf_counter() - started, **common)


def apply(
    variants: Iterable[Trace],
    net: Any,
    initial_marking: Any,
    final_marking: Any,
    prior: TracePrior,
    max_trace_length: int = DEFAULT_MAX_TRACE_LENGTH,
) -> LMPResult:
    started = time.perf_counter()
    language = bounded_language(net, initial_marking, final_marking, max_trace_length)
    result = apply_to_language(variants, language, prior, max_trace_length)
    result.elapsed_seconds = time.perf_counter() - started
    return result
