"""One-call functions in the style of ``pm4py.precision_alignments`` & co.

Every keyword argument is a :class:`~language_mass_precision.estimator.LanguageMassPrecision`
hyperparameter (``length_prior``, ``length_parameter``, ``activity_prior``,
``method``, ``activity_key``, ...).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from language_mass_precision.estimator import LanguageMassPrecision
from language_mass_precision.objects.result import CompiledModel, LMPResult


def precision_language_mass(log: Any, *model: Any, **kwargs: Any) -> LMPResult:
    """Language-Mass Precision of a model with respect to a log.

    ``model`` is ``(net, im, fm)``, a ``ProcessTree``, a ``BPMN`` or a
    :class:`CompiledModel`. Returns an :class:`LMPResult`; ``float(result)``
    gives the value when it is exact.

    >>> import language_mass_precision as lmp
    >>> log = [("a", "b", "c"), ("a", "d", "c")]
    >>> tree = pm4py.parse_process_tree("->('a', X('b', 'd', 'e'), 'c')")  # doctest: +SKIP
    >>> lmp.precision_language_mass(log, tree)                              # doctest: +SKIP
    LMPResult(status=exact, value=0.666667, certification='full_language_finite_sum')
    """
    return LanguageMassPrecision(**kwargs).fit(log).score(*model)


def imprecision_attribution(log: Any, *model: Any, top_k: int = 10, **kwargs: Any) -> pd.DataFrame:
    """The ``top_k`` unseen accepted traces and their exact share of ``1 - LMP``."""
    return LanguageMassPrecision(**kwargs).fit(log).attribution(*model, top_k=top_k)


def rank_models(log: Any, models: Mapping[str, Any] | Sequence[Any], **kwargs: Any) -> pd.DataFrame:
    """Score and rank several models under one shared prior (one row per model)."""
    return LanguageMassPrecision(**kwargs).fit(log).score_many(models)


def compile_model(*model: Any, state_limit: int | None = None, time_limit: float | None = None,
                  name: str | None = None) -> CompiledModel:
    """Compile a model once into a DFA that can be scored under many priors and logs."""
    settings = {k: v for k, v in {"state_limit": state_limit, "time_limit": time_limit}.items() if v is not None}
    return LanguageMassPrecision(**settings).compile(*model, name=name)
