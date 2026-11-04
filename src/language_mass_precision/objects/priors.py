"""Trace priors ``mu_{rho,pi}(sigma) = rho_|sigma| * prod_i pi(a_i)``.

A trace prior is the fixed reference measure of LMP. It factorizes into a
*length prior* ``rho`` over ``n = 0, 1, 2, ...`` and an *activity prior* ``pi``
over a finite alphabet ``Sigma``. Both sum to one, so ``mu`` is a full-support
probability measure over ``Sigma*``. The prior is a declared measurement
choice: it is not an empirical frequency model and not a stochastic semantics
of the process model.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache

from scipy.special import zeta

Trace = tuple[str, ...]


@lru_cache(maxsize=256)
def _log_zeta(alpha: float) -> float:
    return math.log(float(zeta(alpha, 1.0)))


class LengthPrior:
    """Base class for length priors ``rho``; subclasses must be non-increasing."""

    name: str = "length_prior"

    @property
    def parameter(self) -> float:
        raise NotImplementedError

    def log_rho(self, n: int) -> float:
        """Natural log of ``rho_n``."""
        raise NotImplementedError

    def rho(self, n: int) -> float:
        return math.exp(self.log_rho(n))

    def tail(self, k: int) -> float:
        """Total length mass strictly above ``k``: ``sum_{n>k} rho_n``."""
        raise NotImplementedError

    def cdf(self, k: int) -> float:
        return 1.0 - self.tail(k)


@dataclass(frozen=True)
class GeometricLengthPrior(LengthPrior):
    """``rho_n = (1 - lambda) * lambda^n`` with ``0 < lambda < 1``."""

    lam: float
    name: str = field(default="geometric", init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.lam < 1.0:
            raise ValueError("geometric lambda must be in (0, 1)")

    @property
    def parameter(self) -> float:
        return self.lam

    def log_rho(self, n: int) -> float:
        return math.log1p(-self.lam) + n * math.log(self.lam)

    def tail(self, k: int) -> float:
        return 1.0 if k < 0 else self.lam ** (k + 1)

    @classmethod
    def from_median_length(cls, median_length: float) -> GeometricLengthPrior:
        """``lambda_50 = 2^(-1/(l50+1))``: half the mass on lengths <= ``l50``."""
        return cls(lambda_from_median_length(median_length))


@dataclass(frozen=True)
class PowerLawLengthPrior(LengthPrior):
    """``rho_n = (n + 1)^(-alpha) / zeta(alpha)`` with ``alpha > 1``."""

    alpha: float
    name: str = field(default="power_law", init=False)

    def __post_init__(self) -> None:
        if self.alpha <= 1.0:
            raise ValueError("power-law alpha must be greater than 1")

    @property
    def parameter(self) -> float:
        return self.alpha

    def log_rho(self, n: int) -> float:
        return -self.alpha * math.log(n + 1) - _log_zeta(self.alpha)

    def tail(self, k: int) -> float:
        if k < 0:
            return 1.0
        return float(zeta(self.alpha, k + 2) / zeta(self.alpha, 1.0))

    @classmethod
    def from_median_length(cls, median_length: float) -> PowerLawLengthPrior:
        """Solve ``sum_{n<=l50} rho_n = 1/2`` for ``alpha``."""
        return cls(alpha_from_median_length(median_length))


def lambda_from_median_length(median_length: float) -> float:
    if median_length < 0:
        raise ValueError("median_length must be non-negative")
    return 2.0 ** (-1.0 / (median_length + 1.0))


def power_law_length_cdf(k: int, alpha: float) -> float:
    if alpha <= 1.0:
        raise ValueError("alpha must be greater than 1")
    if k < 0:
        return 0.0
    return sum((n + 1) ** (-alpha) for n in range(k + 1)) / float(zeta(alpha, 1.0))


def alpha_from_median_length(median_length: float) -> float:
    if median_length < 0:
        raise ValueError("median_length must be non-negative")
    k = int(round(median_length))
    low, high = 1.000001, 100.0
    for _ in range(120):
        mid = 0.5 * (low + high)
        if power_law_length_cdf(k, mid) < 0.5:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


LENGTH_PRIORS: dict[str, type[LengthPrior]] = {
    "geometric": GeometricLengthPrior,
    "power_law": PowerLawLengthPrior,
}


def make_length_prior(
    prior: str | LengthPrior,
    parameter: float | None = None,
    median_length: float | None = None,
) -> LengthPrior:
    """Resolve a length prior from a name and parameter, or calibrate it.

    With ``parameter=None`` the prior is calibrated so that half of its mass
    lies on traces no longer than ``median_length`` (Section 5 of the paper).
    """
    if isinstance(prior, LengthPrior):
        return prior
    if prior not in LENGTH_PRIORS:
        raise ValueError(f"Unknown length prior {prior!r}; expected one of {sorted(LENGTH_PRIORS)}")
    cls = LENGTH_PRIORS[prior]
    if parameter is not None:
        return cls(parameter)
    if median_length is None:
        raise ValueError("Either a prior parameter or a median length for calibration is required")
    return cls.from_median_length(median_length)


@dataclass(frozen=True)
class TracePrior:
    """The reference measure ``mu_{rho,pi}`` over ``Sigma*``."""

    length: LengthPrior
    activity_mass: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.activity_mass:
            raise ValueError("The activity prior needs a non-empty alphabet")
        if any(mass <= 0.0 for mass in self.activity_mass.values()):
            raise ValueError("The activity prior must have full support (all masses > 0)")
        total = math.fsum(self.activity_mass.values())
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"The activity prior must sum to one (got {total})")

    @property
    def alphabet(self) -> frozenset[str]:
        return frozenset(self.activity_mass)

    @property
    def alphabet_size(self) -> int:
        return len(self.activity_mass)

    @property
    def is_uniform(self) -> bool:
        values = list(self.activity_mass.values())
        return all(math.isclose(v, values[0], rel_tol=1e-12) for v in values)

    def log_mass(self, trace: Iterable[str]) -> float:
        """Natural-log trace mass; stable where :meth:`mass` would underflow."""
        trace = tuple(trace)
        try:
            symbols = math.fsum(math.log(self.activity_mass[a]) for a in trace)
        except KeyError as missing:
            raise ValueError(f"Activity {missing.args[0]!r} is outside the prior alphabet") from None
        return self.length.log_rho(len(trace)) + symbols

    def mass(self, trace: Iterable[str]) -> float:
        return math.exp(self.log_mass(trace))

    def language_mass(self, traces: Iterable[Trace]) -> float:
        """``mu(X)`` for an explicit finite set of traces."""
        return math.fsum(self.mass(trace) for trace in set(map(tuple, traces)))

    def describe(self) -> dict[str, object]:
        return {
            "length_prior": self.length.name,
            "length_parameter": self.length.parameter,
            "activity_prior": "uniform" if self.is_uniform else "custom",
            "alphabet_size": self.alphabet_size,
        }

    @classmethod
    def uniform(cls, length: LengthPrior, alphabet: Iterable[str]) -> TracePrior:
        """``pi(a) = 1/|Sigma|``, the instantiation used for the paper's main results."""
        alphabet = sorted(set(alphabet))
        if not alphabet:
            raise ValueError("The alphabet must not be empty")
        return cls(length, {a: 1.0 / len(alphabet) for a in alphabet})

    @classmethod
    def empirical(
        cls,
        length: LengthPrior,
        activity_counts: Mapping[str, int],
        alphabet: Iterable[str] | None = None,
        delta: float = 1.0,
    ) -> TracePrior:
        """Laplace-smoothed activity frequencies ``(c(a) + delta) / (N + delta |Sigma|)``.

        Smoothing keeps full support for activities the log never shows
        (``delta > 0`` is required for that).
        """
        symbols = sorted(set(activity_counts) | set(alphabet or ()))
        if not symbols:
            raise ValueError("The alphabet must not be empty")
        if delta < 0:
            raise ValueError("delta must be non-negative")
        total = sum(activity_counts.get(a, 0) for a in symbols)
        denominator = total + delta * len(symbols)
        return cls(length, {a: (activity_counts.get(a, 0) + delta) / denominator for a in symbols})


def build_trace_prior(
    alphabet: Iterable[str],
    length_prior: str | LengthPrior = "geometric",
    length_parameter: float | None = None,
    activity_prior: str | Mapping[str, float] = "uniform",
    median_length: float | None = None,
    activity_counts: Mapping[str, int] | None = None,
    laplace_delta: float = 1.0,
) -> TracePrior:
    """Assemble ``mu_{rho,pi}`` from declarative settings.

    ``activity_prior`` is ``"uniform"``, ``"empirical"`` (Laplace-smoothed log
    frequencies) or an explicit mapping ``{activity: mass}`` over ``alphabet``.
    """
    length = make_length_prior(length_prior, length_parameter, median_length)
    alphabet = sorted(set(alphabet))
    if isinstance(activity_prior, Mapping):
        missing = set(alphabet) - set(activity_prior)
        if missing:
            raise ValueError(f"The activity prior has no mass for {sorted(missing)}")
        return TracePrior(length, {a: float(activity_prior[a]) for a in alphabet})
    if activity_prior == "uniform":
        return TracePrior.uniform(length, alphabet)
    if activity_prior == "empirical":
        if laplace_delta <= 0:
            raise ValueError("laplace_delta must be positive to keep full support")
        return TracePrior.empirical(length, activity_counts or {}, alphabet, delta=laplace_delta)
    raise ValueError(f"Unknown activity prior {activity_prior!r}; expected 'uniform', 'empirical' or a mapping")
