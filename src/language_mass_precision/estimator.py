"""Object-oriented interface following scikit-learn conventions.

``fit`` reads the log (variant support, alphabet, median length, activity
counts); ``compile`` turns a model into a reusable DFA; ``score`` returns an
:class:`LMPResult`. Hyperparameters live in the constructor and support
``get_params``/``set_params``, so ``sklearn.base.clone`` and
``sklearn.model_selection.ParameterGrid`` work without scikit-learn being a
dependency. The prior is resolved at score time, so a parameter sweep over a
compiled model does no recompilation.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from language_mass_precision.algo import attribution as attribution_algo
from language_mass_precision.algo.conversion import to_automaton
from language_mass_precision.algo.denominator import DEFAULT_MAX_K, DEFAULT_WIDTH_TOLERANCE
from language_mass_precision.algo.precision import algorithm
from language_mass_precision.algo.precision.variants import playout
from language_mass_precision.objects.priors import LengthPrior, TracePrior, build_trace_prior, make_length_prior
from language_mass_precision.objects.result import CompiledModel, LMPResult
from language_mass_precision.utils import log_utils, model_utils

METHODS = {"exact": algorithm.Variants.EXACT, "playout": algorithm.Variants.PLAYOUT}


class NotFittedError(RuntimeError):
    """Raised when a scoring method is called before :meth:`LanguageMassPrecision.fit`."""


class LanguageMassPrecision:
    """Language-Mass Precision ``LMP_mu(L, M) = mu(L~ & L(M)) / mu(L(M))``.

    Parameters
    ----------
    length_prior : {"geometric", "power_law"} or LengthPrior, default="geometric"
        Family of the length prior ``rho``.
    length_parameter : float, optional
        ``lambda`` (geometric) or ``alpha`` (power law). ``None`` calibrates it so
        half the prior mass lies on traces no longer than the median variant length.
    activity_prior : {"uniform", "empirical"} or mapping, default="uniform"
        Activity prior ``pi``; ``"empirical"`` uses Laplace-smoothed log frequencies.
    laplace_delta : float, default=1.0
        Smoothing constant for the empirical activity prior.
    alphabet : iterable of str, optional
        Declared alphabet ``Sigma``, merged with the log activities. Declare it
        to keep the prior identical across separately scored models.
    method : {"exact", "playout"}, default="exact"
        ``"exact"`` computes the full-language value or a certified interval;
        ``"playout"`` is the bounded baseline over traces of length <= ``max_trace_length``.
    max_trace_length : int, default=10
        Truncation length ``K`` of the playout baseline.
    state_limit : int, default=200000
        Maximum reachability/DFA states before reporting ``INCOMPLETE``.
    time_limit : float, default=120.0
        Seconds allowed for compilation before reporting ``INCOMPLETE``.
    width_tolerance : float, default=1e-6
        Target relative width of certified denominator intervals (``(D_U - D_L) / D_L``).
    max_k : int, default=4096
        Largest truncation length tried for certified intervals.
    activity_key, case_id_key, timestamp_key : str
        PM4Py column/attribute names.

    Attributes
    ----------
    variants_ : frozenset of tuple
        Distinct log variants (the log support ``L~``).
    alphabet_ : frozenset of str
        Log activities merged with the declared alphabet.
    median_length_ : float
        Median length over distinct variants.
    n_cases_ : int
        Number of cases in the fitted log.
    """

    def __init__(
        self,
        length_prior: str | LengthPrior = "geometric",
        length_parameter: float | None = None,
        activity_prior: str | Mapping[str, float] = "uniform",
        laplace_delta: float = 1.0,
        alphabet: Iterable[str] | None = None,
        method: str = "exact",
        max_trace_length: int = playout.DEFAULT_MAX_TRACE_LENGTH,
        state_limit: int = to_automaton.DEFAULT_STATE_LIMIT,
        time_limit: float = to_automaton.DEFAULT_TIME_LIMIT_SECONDS,
        width_tolerance: float = DEFAULT_WIDTH_TOLERANCE,
        max_k: int = DEFAULT_MAX_K,
        activity_key: str = log_utils.DEFAULT_ACTIVITY_KEY,
        case_id_key: str = log_utils.DEFAULT_CASE_ID_KEY,
        timestamp_key: str = log_utils.DEFAULT_TIMESTAMP_KEY,
    ) -> None:
        self.length_prior = length_prior
        self.length_parameter = length_parameter
        self.activity_prior = activity_prior
        self.laplace_delta = laplace_delta
        self.alphabet = alphabet
        self.method = method
        self.max_trace_length = max_trace_length
        self.state_limit = state_limit
        self.time_limit = time_limit
        self.width_tolerance = width_tolerance
        self.max_k = max_k
        self.activity_key = activity_key
        self.case_id_key = case_id_key
        self.timestamp_key = timestamp_key

    # -- scikit-learn parameter protocol ---------------------------------

    @classmethod
    def _get_param_names(cls) -> list[str]:
        signature = inspect.signature(cls.__init__)
        return [name for name in signature.parameters if name != "self"]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._get_param_names()}

    def set_params(self, **params: Any) -> LanguageMassPrecision:
        valid = set(self._get_param_names())
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"Invalid parameter {name!r} for {type(self).__name__}; valid: {sorted(valid)}")
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(type(self).__init__).parameters.items()
            if name != "self"
        }
        changed = [f"{k}={v!r}" for k, v in self.get_params().items() if v != defaults[k]]
        return f"{type(self).__name__}({', '.join(changed)})"

    # -- fitting -----------------------------------------------------------

    def fit(self, log: Any, y: Any = None) -> LanguageMassPrecision:
        """Read the log support and the statistics used to calibrate the prior."""
        self.stats_ = log_utils.log_statistics(log, self.activity_key, self.case_id_key, self.timestamp_key)
        if not self.stats_.variant_counts:
            raise ValueError("Cannot fit LMP on an empty log")
        self.variants_ = self.stats_.variants
        self.alphabet_ = self.stats_.alphabet | frozenset(self.alphabet or ())
        self.median_length_ = self.stats_.median_variant_length
        self.n_cases_ = self.stats_.n_cases
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "stats_"):
            raise NotFittedError(f"This {type(self).__name__} instance is not fitted yet; call fit(log) first")

    @property
    def length_prior_(self) -> LengthPrior:
        """The resolved (possibly calibrated) length prior."""
        self._check_fitted()
        return make_length_prior(self.length_prior, self.length_parameter, self.median_length_)

    def prior(self, alphabet: Iterable[str] | None = None) -> TracePrior:
        """The trace prior over ``alphabet_`` (optionally extended with ``alphabet``)."""
        self._check_fitted()
        return build_trace_prior(
            self.alphabet_ | frozenset(alphabet or ()),
            length_prior=self.length_prior,
            length_parameter=self.length_parameter,
            activity_prior=self.activity_prior,
            median_length=self.median_length_,
            activity_counts=self.stats_.activity_counts,
            laplace_delta=self.laplace_delta,
        )

    # -- models --------------------------------------------------------------

    def compile(self, *model: Any, name: str | None = None) -> CompiledModel:
        """Compile ``(net, im, fm)``, a process tree or a BPMN into a reusable DFA."""
        if model_utils.is_compiled(*model):
            return model[0]
        net, im, fm = model_utils.as_petri_net(*model)
        compiled = to_automaton.apply(net, im, fm, parameters={
            to_automaton.Parameters.STATE_LIMIT: self.state_limit,
            to_automaton.Parameters.TIME_LIMIT: self.time_limit,
        })
        if name is not None:
            compiled.name = name
        return compiled

    def _variant(self) -> algorithm.Variants:
        if self.method not in METHODS:
            raise ValueError(f"Unknown method {self.method!r}; expected one of {sorted(METHODS)}")
        return METHODS[self.method]

    def _prepare(self, *model: Any) -> CompiledModel | tuple[Any, Any, Any]:
        if model_utils.is_compiled(*model):
            return model[0]
        net_model = model_utils.as_petri_net(*model)
        return net_model if self.method == "playout" else self.compile(*net_model)

    def _evaluate(self, prepared: Any, alphabet: Iterable[str] | None) -> LMPResult:
        return algorithm.evaluate(
            self.stats_,
            prepared,
            variant=self._variant(),
            length_prior=self.length_prior,
            length_parameter=self.length_parameter,
            activity_prior=self.activity_prior,
            laplace_delta=self.laplace_delta,
            alphabet=alphabet,
            max_trace_length=self.max_trace_length,
            state_limit=self.state_limit,
            time_limit=self.time_limit,
            width_tolerance=self.width_tolerance,
            max_k=self.max_k,
        )

    # -- scoring -------------------------------------------------------------

    def score(self, *model: Any) -> LMPResult:
        """LMP of one model: ``(net, im, fm)``, a process tree, a BPMN or a :class:`CompiledModel`."""
        self._check_fitted()
        return self._evaluate(self._prepare(*model), self.alphabet)

    def fit_score(self, log: Any, *model: Any) -> LMPResult:
        return self.fit(log).score(*model)

    def score_many(self, models: Mapping[str, Any] | Sequence[Any]) -> pd.DataFrame:
        """Score several models under one shared prior and rank them.

        ``Sigma`` is the union of the fitted alphabet and every model-language
        alphabet, so all scores use the same reference measure and are comparable.
        """
        self._check_fitted()
        items = models.items() if isinstance(models, Mapping) else ((f"model_{i}", m) for i, m in enumerate(models))
        prepared = {}
        labels: set[str] = set()
        for name, model in items:
            model_parts = model if isinstance(model, tuple) else (model,)
            prepared[name] = self._prepare(*model_parts)
            if isinstance(prepared[name], CompiledModel):
                labels |= prepared[name].labels
            else:
                net = prepared[name][0]
                labels |= model_utils.visible_labels(net)
        shared = self.alphabet_ | labels
        rows = []
        for name, item in prepared.items():
            result = self._evaluate(item, shared)
            rows.append({"model": name, **result.to_dict()})
        frame = pd.DataFrame(rows)
        ordering = frame["value"].where(frame["value"].notna(), (frame["lower"] + frame["upper"]) / 2)
        frame.insert(1, "rank", ordering.rank(ascending=False, method="min").astype("Int64"))
        return frame

    def attribution(self, *model: Any, top_k: int = attribution_algo.DEFAULT_TOP_K,
                    max_expansions: int = attribution_algo.DEFAULT_MAX_EXPANSIONS) -> pd.DataFrame:
        """The ``top_k`` unseen accepted traces and their exact share of ``1 - LMP``."""
        self._check_fitted()
        compiled = self.compile(*model)
        prior = self.prior(algorithm.resolve_alphabet(self.stats_, compiled.labels, self.alphabet))
        return attribution_algo.apply(compiled, self.variants_, prior, top_k=top_k, max_expansions=max_expansions)
