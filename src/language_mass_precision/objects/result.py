"""Result objects: compiled models and LMP outcomes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from language_mass_precision.objects.automaton import DeterministicAutomaton, language_alphabet


class Status(str, Enum):
    """What kind of answer an LMP computation produced.

    ``EXACT``
        The full-language value was computed exactly (sparse linear solve or
        finite sum over an acyclic productive DFA).
    ``INTERVAL``
        A certified interval ``[lower, upper]`` that contains the full-language
        value; the width comes from a rigorous length-tail bound.
    ``BOUNDED``
        A truncated value over traces of length ``<= truncation_k`` (playout
        baseline). It carries no certificate about the full language.
    ``INCOMPLETE``
        The language representation could not be built within the configured
        state/time limits; no value is reported.
    ``UNDEFINED``
        The model language is empty, so the quotient is undefined.
    """

    EXACT = "exact"
    INTERVAL = "interval"
    BOUNDED = "bounded"
    INCOMPLETE = "incomplete"
    UNDEFINED = "undefined"

    def __str__(self) -> str:
        return self.value


@dataclass
class CompiledModel:
    """A model compiled to a DFA once, reusable across priors and logs."""

    dfa: DeterministicAutomaton
    reachability_states: int = 0
    compile_seconds: float = 0.0
    name: str | None = None

    @property
    def status(self) -> str:
        return self.dfa.status

    @property
    def ok(self) -> bool:
        return self.dfa.ok

    @property
    def labels(self) -> frozenset[str]:
        """Labels occurring in the model language (dead branches excluded)."""
        return language_alphabet(self.dfa) if self.ok else frozenset()

    @property
    def dfa_states(self) -> int:
        return self.dfa.state_count

    def __repr__(self) -> str:
        label = f"{self.name!r}, " if self.name else ""
        return (
            f"CompiledModel({label}status={self.status!r}, dfa_states={self.dfa_states}, "
            f"reachability_states={self.reachability_states}, labels={len(self.labels)})"
        )


@dataclass
class LMPResult:
    """Outcome of one LMP computation for one (log, model, prior) triple.

    ``value`` is set only for ``EXACT`` and ``BOUNDED`` results. ``lower`` and
    ``upper`` are always set when a number could be produced, and coincide
    with ``value`` for exact results.
    """

    status: Status
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    certification: str = "none"
    numerator_mass: float | None = None
    log_numerator_mass: float | None = None
    denominator_lower: float | None = None
    denominator_upper: float | None = None
    accepted_variants: int | None = None
    rejected_variants: int | None = None
    truncation_k: int | None = None
    dfa_states: int | None = None
    reachability_states: int | None = None
    prior: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    elapsed_seconds: float = 0.0

    @property
    def is_exact(self) -> bool:
        return self.status is Status.EXACT

    @property
    def has_bounds(self) -> bool:
        return self.lower is not None and self.upper is not None

    @property
    def width(self) -> float | None:
        return self.upper - self.lower if self.has_bounds else None

    @property
    def midpoint(self) -> float | None:
        return 0.5 * (self.lower + self.upper) if self.has_bounds else None

    @property
    def imprecision(self) -> float | None:
        """``1 - LMP``: model-language mass not supported by the log."""
        return None if self.value is None else 1.0 - self.value

    def certainly_greater(self, other: LMPResult) -> bool:
        """True if the certified intervals prove ``self > other``."""
        if self.status not in (Status.EXACT, Status.INTERVAL) or other.status not in (Status.EXACT, Status.INTERVAL):
            return False
        return self.lower > other.upper

    def __float__(self) -> float:
        if self.value is None:
            if self.status is Status.INTERVAL:
                raise ValueError(
                    f"LMP is only certified within [{self.lower:.6g}, {self.upper:.6g}]; "
                    "use .lower, .upper or .midpoint explicitly"
                )
            raise ValueError(f"No LMP value available (status={self.status}, reason={self.reason})")
        return float(self.value)

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["status"] = self.status.value
        prior = record.pop("prior")
        record.update({f"prior_{key}": value for key, value in prior.items()})
        return record

    def __repr__(self) -> str:
        if self.status in (Status.EXACT, Status.BOUNDED) and self.value is not None:
            extra = f", truncation_k={self.truncation_k}" if self.status is Status.BOUNDED else ""
            return (
                f"LMPResult(status={self.status}, value={self.value:.6g}{extra}, "
                f"certification={self.certification!r})"
            )
        if self.status is Status.INTERVAL:
            return (
                f"LMPResult(status={self.status}, lower={self.lower:.6g}, upper={self.upper:.6g}, "
                f"width={self.width:.2g}, certification={self.certification!r})"
            )
        return f"LMPResult(status={self.status}, reason={self.reason!r})"


def nan_if_none(value: float | None) -> float:
    return math.nan if value is None else value
