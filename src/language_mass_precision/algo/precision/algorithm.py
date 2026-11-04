"""Language-Mass Precision, PM4Py-style.

``LMP_mu(L, M) = mu(L~ intersect L(M)) / mu(L(M))`` where ``L~`` is the set of
distinct log variants and ``mu`` a fixed full-support trace prior.

Usage mirrors ``pm4py.algo.evaluation.precision.algorithm``::

    from language_mass_precision.algo.precision import algorithm as lmp_algorithm

    result = lmp_algorithm.apply(log, net, im, fm,
                                 variant=lmp_algorithm.Variants.EXACT,
                                 parameters={lmp_algorithm.Parameters.LENGTH_PRIOR: "power_law"})
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any

from pm4py.util import constants, exec_utils

from language_mass_precision.algo.conversion import to_automaton
from language_mass_precision.algo.denominator import DEFAULT_MAX_K, DEFAULT_WIDTH_TOLERANCE
from language_mass_precision.algo.precision.variants import exact, playout
from language_mass_precision.objects.priors import LengthPrior, build_trace_prior
from language_mass_precision.objects.result import CompiledModel, LMPResult
from language_mass_precision.utils import log_utils
from language_mass_precision.utils.log_utils import LogStatistics


class Variants(Enum):
    EXACT = exact
    PLAYOUT = playout


DEFAULT_VARIANT = Variants.EXACT


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    LENGTH_PRIOR = "length_prior"
    LENGTH_PARAMETER = "length_parameter"
    ACTIVITY_PRIOR = "activity_prior"
    LAPLACE_DELTA = "laplace_delta"
    ALPHABET = "alphabet"
    MAX_TRACE_LENGTH = "max_trace_length"
    STATE_LIMIT = "state_limit"
    TIME_LIMIT = "time_limit"
    WIDTH_TOLERANCE = "width_tolerance"
    MAX_K = "max_k"


class AlphabetExtensionWarning(UserWarning):
    """The model uses activities outside the declared alphabet, so ``Sigma`` was extended."""


def resolve_alphabet(
    stats: LogStatistics,
    model_labels: Iterable[str],
    alphabet: Iterable[str] | None = None,
) -> frozenset[str]:
    """``Sigma`` = declared alphabet (if any) + log activities + model-language labels.

    When an alphabet is declared, extending it means this score is not on the
    same prior as scores computed under the declared alphabet, so a warning is issued.
    """
    model_labels = frozenset(model_labels)
    base = frozenset(alphabet or ()) | stats.alphabet
    extra = model_labels - base
    if alphabet is not None and extra:
        warnings.warn(
            f"Model-language activities {sorted(extra)} are outside the declared alphabet; Sigma was "
            "extended for this model, so its score is not comparable with scores under the declared "
            "alphabet. Declare the union alphabet (or use rank_models) to compare models.",
            AlphabetExtensionWarning,
            stacklevel=3,
        )
    return base | model_labels


def evaluate(
    stats: LogStatistics,
    model: CompiledModel | tuple[Any, Any, Any],
    variant: Variants = DEFAULT_VARIANT,
    length_prior: str | LengthPrior = "geometric",
    length_parameter: float | None = None,
    activity_prior: str | Mapping[str, float] = "uniform",
    laplace_delta: float = 1.0,
    alphabet: Iterable[str] | None = None,
    max_trace_length: int = playout.DEFAULT_MAX_TRACE_LENGTH,
    state_limit: int = to_automaton.DEFAULT_STATE_LIMIT,
    time_limit: float = to_automaton.DEFAULT_TIME_LIMIT_SECONDS,
    width_tolerance: float = DEFAULT_WIDTH_TOLERANCE,
    max_k: int = DEFAULT_MAX_K,
) -> LMPResult:
    """Score one model against precomputed log statistics (the shared workhorse)."""

    def prior_for(model_labels: Iterable[str]):
        return build_trace_prior(
            resolve_alphabet(stats, model_labels, alphabet),
            length_prior=length_prior,
            length_parameter=length_parameter,
            activity_prior=activity_prior,
            median_length=stats.median_variant_length,
            activity_counts=stats.activity_counts,
            laplace_delta=laplace_delta,
        )

    if variant is Variants.PLAYOUT:
        if isinstance(model, CompiledModel):
            raise TypeError("The playout variant needs the Petri net, not a CompiledModel")
        net, im, fm = model
        language = playout.bounded_language(net, im, fm, max_trace_length)
        labels = {activity for trace in language for activity in trace}
        return playout.apply_to_language(stats.variants, language, prior_for(labels), max_trace_length)

    if isinstance(model, CompiledModel):
        compiled = model
    else:
        compiled = to_automaton.apply(*model, parameters={
            to_automaton.Parameters.STATE_LIMIT: state_limit,
            to_automaton.Parameters.TIME_LIMIT: time_limit,
        })
    prior = prior_for(compiled.labels)
    result = exact.apply(stats.variants, compiled, prior, width_tolerance=width_tolerance, max_k=max_k)
    result.elapsed_seconds += compiled.compile_seconds if not isinstance(model, CompiledModel) else 0.0
    return result


def apply(
    log: Any,
    net: Any,
    initial_marking: Any,
    final_marking: Any,
    variant: Variants = DEFAULT_VARIANT,
    parameters: dict[Any, Any] | None = None,
) -> LMPResult:
    """Compute LMP of an accepting Petri net with respect to a log.

    Parameters
    ----------
    log
        PM4Py ``EventLog``, PM4Py-format ``pandas.DataFrame``, ``{trace: count}``
        mapping, or an iterable of traces.
    net, initial_marking, final_marking
        Accepting Petri net.
    variant
        ``Variants.EXACT`` (full language: exact value or certified interval) or
        ``Variants.PLAYOUT`` (bounded baseline over traces of length ``<= MAX_TRACE_LENGTH``).
    parameters
        Keys from :class:`Parameters`. ``LENGTH_PARAMETER=None`` (default)
        calibrates the length prior from the median variant length.
    """
    parameters = parameters or {}

    def get(key: Parameters, default: Any) -> Any:
        return exec_utils.get_param_value(key, parameters, default)

    stats = log_utils.log_statistics(
        log,
        activity_key=get(Parameters.ACTIVITY_KEY, log_utils.DEFAULT_ACTIVITY_KEY),
        case_id_key=get(Parameters.CASE_ID_KEY, log_utils.DEFAULT_CASE_ID_KEY),
        timestamp_key=get(Parameters.TIMESTAMP_KEY, log_utils.DEFAULT_TIMESTAMP_KEY),
    )
    return evaluate(
        stats,
        (net, initial_marking, final_marking),
        variant=variant,
        length_prior=get(Parameters.LENGTH_PRIOR, "geometric"),
        length_parameter=get(Parameters.LENGTH_PARAMETER, None),
        activity_prior=get(Parameters.ACTIVITY_PRIOR, "uniform"),
        laplace_delta=get(Parameters.LAPLACE_DELTA, 1.0),
        alphabet=get(Parameters.ALPHABET, None),
        max_trace_length=get(Parameters.MAX_TRACE_LENGTH, playout.DEFAULT_MAX_TRACE_LENGTH),
        state_limit=get(Parameters.STATE_LIMIT, to_automaton.DEFAULT_STATE_LIMIT),
        time_limit=get(Parameters.TIME_LIMIT, to_automaton.DEFAULT_TIME_LIMIT_SECONDS),
        width_tolerance=get(Parameters.WIDTH_TOLERANCE, DEFAULT_WIDTH_TOLERANCE),
        max_k=get(Parameters.MAX_K, DEFAULT_MAX_K),
    )
