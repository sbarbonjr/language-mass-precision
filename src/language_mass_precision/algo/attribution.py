"""Trace-level imprecision attribution.

LMP is additive over traces, so ``1 - LMP = sum over sigma in L(M) minus L~
of mu(sigma) / mu(L(M))``: every unseen accepted trace owns an exact share of
the imprecision. This module returns the largest shares by best-first search
over the DFA, which also works when ``L(M)`` is infinite.

Search order is sound because, for non-increasing length priors, a prefix's
own mass ``rho_|p| * prod pi(p)`` upper-bounds the mass of every extension.
Traces therefore come out in non-increasing mass order.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Iterable

import pandas as pd

from language_mass_precision.algo import denominator as denominator_algo
from language_mass_precision.objects.automaton import Trace, productive_states
from language_mass_precision.objects.priors import TracePrior
from language_mass_precision.objects.result import CompiledModel

DEFAULT_TOP_K = 10
DEFAULT_MAX_EXPANSIONS = 1_000_000

COLUMNS = [
    "rank", "trace", "length", "trace_mass",
    "imprecision_share", "share_lower", "share_upper", "cumulative_share",
]


def unseen_traces(
    compiled: CompiledModel,
    variants: Iterable[Trace],
    prior: TracePrior,
    top_k: int = DEFAULT_TOP_K,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
) -> list[tuple[Trace, float]]:
    """Up to ``top_k`` unseen accepted traces with their natural-log masses, heaviest first."""
    if not compiled.ok:
        raise ValueError(f"Automaton is not complete: {compiled.status}")
    dfa = compiled.dfa
    observed = set(variants)
    productive = productive_states(dfa)
    if dfa.initial not in productive:
        return []
    log_pi = {a: math.log(m) for a, m in prior.activity_mass.items() if m > 0}

    tie = itertools.count()
    heap = [(-prior.length.log_rho(0), next(tie), dfa.initial, (), 0.0)]
    found: list[tuple[Trace, float]] = []
    expansions = 0
    while heap and len(found) < top_k and expansions < max_expansions:
        neg_log_mass, _, state, prefix, symbol_sum = heapq.heappop(heap)
        expansions += 1
        if state in dfa.accepting and prefix not in observed:
            found.append((prefix, -neg_log_mass))
        length = len(prefix) + 1
        log_rho = prior.length.log_rho(length)
        for label, target in dfa.transitions[state].items():
            if target in productive and label in log_pi:
                child_sum = symbol_sum + log_pi[label]
                heapq.heappush(heap, (-(log_rho + child_sum), next(tie), target, prefix + (label,), child_sum))
    return found


def apply(
    compiled: CompiledModel,
    variants: Iterable[Trace],
    prior: TracePrior,
    top_k: int = DEFAULT_TOP_K,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
) -> pd.DataFrame:
    """Table of the ``top_k`` unseen accepted traces and their share of ``1 - LMP``.

    ``imprecision_share`` is exact when the denominator is; otherwise it is
    NaN and ``share_lower``/``share_upper`` carry the certified bounds.
    """
    variants = set(variants)
    denom = denominator_algo.compute(compiled.dfa, prior)
    rows = []
    for rank, (trace, log_mass) in enumerate(unseen_traces(compiled, variants, prior, top_k, max_expansions), start=1):
        mass = math.exp(log_mass)
        lower = mass / denom.upper if denom.upper > 0 else math.nan
        upper = mass / denom.lower if denom.lower > 0 else math.nan
        rows.append({
            "rank": rank,
            "trace": trace,
            "length": len(trace),
            "trace_mass": mass,
            "imprecision_share": lower if denom.exact else math.nan,
            "share_lower": lower,
            "share_upper": upper,
        })
    frame = pd.DataFrame(rows, columns=[c for c in COLUMNS if c != "cumulative_share"])
    frame["cumulative_share"] = frame["share_lower"].cumsum()
    frame.attrs["denominator_certification"] = denom.certification
    return frame[COLUMNS]
