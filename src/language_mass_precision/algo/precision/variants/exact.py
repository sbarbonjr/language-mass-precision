"""Full-language LMP over a compiled DFA: exact value or certified interval."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable

from scipy.special import logsumexp

from language_mass_precision.algo import denominator as denominator_algo
from language_mass_precision.objects.automaton import Trace, accepts
from language_mass_precision.objects.priors import TracePrior
from language_mass_precision.objects.result import CompiledModel, LMPResult, Status


def numerator(compiled: CompiledModel, variants: Iterable[Trace], prior: TracePrior) -> tuple[float, float, int, int]:
    """``mu(L~ intersect L(M))`` over distinct variants, accumulated in log space.

    Returns ``(mass, log_mass, accepted, rejected)``.
    """
    log_masses = []
    rejected = 0
    for trace in set(variants):
        if accepts(compiled.dfa, trace):
            log_masses.append(prior.log_mass(trace))
        else:
            rejected += 1
    if not log_masses:
        return 0.0, float("-inf"), 0, rejected
    log_mass = float(logsumexp(log_masses))
    return math.exp(log_mass), log_mass, len(log_masses), rejected


def apply(
    variants: Iterable[Trace],
    compiled: CompiledModel,
    prior: TracePrior,
    width_tolerance: float = denominator_algo.DEFAULT_WIDTH_TOLERANCE,
    max_k: int = denominator_algo.DEFAULT_MAX_K,
) -> LMPResult:
    started = time.perf_counter()
    common = {
        "dfa_states": compiled.dfa_states,
        "reachability_states": compiled.reachability_states,
        "prior": prior.describe(),
    }
    if not compiled.ok:
        return LMPResult(status=Status.INCOMPLETE, reason=compiled.status,
                         elapsed_seconds=time.perf_counter() - started, **common)

    n_mass, n_log, accepted, rejected = numerator(compiled, variants, prior)
    denom = denominator_algo.compute(compiled.dfa, prior, width_tolerance=width_tolerance, max_k=max_k)
    common.update(
        numerator_mass=n_mass,
        log_numerator_mass=n_log,
        denominator_lower=denom.lower,
        denominator_upper=denom.upper,
        accepted_variants=accepted,
        rejected_variants=rejected,
        truncation_k=denom.truncation_k,
        certification=denom.certification,
    )

    if denom.upper <= 0.0:
        return LMPResult(status=Status.UNDEFINED, reason="empty_model_language",
                         elapsed_seconds=time.perf_counter() - started, **common)

    if denom.exact:
        value = min(n_mass / denom.lower, 1.0)
        return LMPResult(status=Status.EXACT, value=value, lower=value, upper=value,
                         elapsed_seconds=time.perf_counter() - started, **common)

    # The numerator is a subset of L(M), so N <= D and the upper end caps at one.
    lower = n_mass / denom.upper
    upper = min(n_mass / denom.lower, 1.0) if denom.lower > 0.0 else 1.0
    return LMPResult(status=Status.INTERVAL, lower=lower, upper=upper,
                     elapsed_seconds=time.perf_counter() - started, **common)
