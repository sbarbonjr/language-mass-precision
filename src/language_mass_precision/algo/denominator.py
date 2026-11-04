"""The LMP denominator ``mu(L(M))`` over a DFA.

Writing ``p_n`` for the ``pi``-mass of accepted traces of length exactly
``n``, the denominator is ``D = sum_n rho_n p_n``. Three computations are
available, tried in this order:

1. **Finite sum** - if the productive DFA is acyclic, ``L(M)`` is finite and
   ``D`` is a finite sum up to the longest accepted trace (any length prior).
2. **Sparse solve** - for the geometric prior, ``x = (1-lambda) f + lambda A x``
   gives ``D = x[q0]`` exactly, with ``A[q, r] = sum of pi(a) over delta(q,a)=r``.
3. **Certified interval** - otherwise ``D_K <= D <= D_K + s_{K+1} sum_{n>K} rho_n``,
   where ``s_{K+1}`` is the prior mass still on productive states after
   ``K+1`` symbols. ``K`` grows until the interval width is below
   ``width_tolerance`` relative to ``D_K`` (or ``K = max_k``), so the LMP
   interval has roughly that relative width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from language_mass_precision.objects.automaton import (
    DeterministicAutomaton,
    is_acyclic,
    language_alphabet,
    max_accepted_length,
    productive_states,
)
from language_mass_precision.objects.priors import GeometricLengthPrior, TracePrior

CERT_SPARSE_SOLVE = "full_language_sparse_solve"
CERT_FINITE_SUM = "full_language_finite_sum"
CERT_INTERVAL = "deterministic_interval"
CERT_BOUNDED = "bounded_playout"
CERT_NONE = "none"

EXACT_CERTIFICATIONS = frozenset({CERT_SPARSE_SOLVE, CERT_FINITE_SUM})

DEFAULT_WIDTH_TOLERANCE = 1e-6
DEFAULT_MAX_K = 4096
DEFAULT_RESIDUAL_TOLERANCE = 1e-8


@dataclass
class DenominatorResult:
    certification: str
    lower: float
    upper: float
    truncation_k: int | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def exact(self) -> bool:
        return self.certification in EXACT_CERTIFICATIONS


def label_mass_for(dfa: DeterministicAutomaton, prior: TracePrior) -> dict[str, float]:
    """``pi`` restricted to the DFA labels.

    Labels that only occur on dead branches never reach acceptance, so they
    may lie outside the prior alphabet and get zero mass.
    """
    missing = language_alphabet(dfa) - prior.alphabet
    if missing:
        raise ValueError(f"Model-language labels outside the prior alphabet: {sorted(missing)}")
    return {label: prior.activity_mass.get(label, 0.0) for label in dfa.alphabet}


def mass_matrix(dfa: DeterministicAutomaton, label_mass: dict[str, float]) -> sp.csr_matrix:
    """Sparse ``A[q, r] = sum of pi(a)`` over labels with ``delta(q, a) = r``.

    Alphabet letters the model never produces have no edge: their mass leaves
    the model language for good, which is exactly the implicit sink state.
    """
    n = dfa.state_count
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for state, moves in enumerate(dfa.transitions):
        for label, target in moves.items():
            rows.append(state)
            cols.append(target)
            data.append(label_mass[label])
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    matrix.sum_duplicates()
    return matrix


def accept_vector(dfa: DeterministicAutomaton) -> np.ndarray:
    vector = np.zeros(dfa.state_count)
    for state in dfa.accepting:
        vector[state] = 1.0
    return vector


def bounded_length_masses(dfa: DeterministicAutomaton, label_mass: dict[str, float], max_k: int) -> list[float]:
    """``[p_0, ..., p_max_k]``: ``pi``-mass of accepted traces of each exact length."""
    transpose = mass_matrix(dfa, label_mass).transpose().tocsr()
    accept = accept_vector(dfa)
    state_vector = np.zeros(dfa.state_count)
    state_vector[dfa.initial] = 1.0
    masses = [float(state_vector @ accept)]
    for _ in range(max_k):
        state_vector = transpose @ state_vector
        masses.append(float(state_vector @ accept))
    return masses


def compute(
    dfa: DeterministicAutomaton,
    prior: TracePrior,
    width_tolerance: float = DEFAULT_WIDTH_TOLERANCE,
    max_k: int = DEFAULT_MAX_K,
    residual_tolerance: float = DEFAULT_RESIDUAL_TOLERANCE,
) -> DenominatorResult:
    """Return ``mu(L(M))`` exactly or as a certified interval."""
    if not dfa.ok:
        raise ValueError(f"Automaton is not complete: {dfa.status}")
    label_mass = label_mass_for(dfa, prior)
    length = prior.length

    productive = productive_states(dfa)
    if not productive:
        return DenominatorResult(CERT_FINITE_SUM, 0.0, 0.0, truncation_k=0, diagnostics={"empty_language": True})

    if is_acyclic(dfa, productive):
        max_len = max_accepted_length(dfa, productive)
        masses = bounded_length_masses(dfa, label_mass, max_len)
        total = math.fsum(length.rho(n) * masses[n] for n in range(max_len + 1))
        return DenominatorResult(CERT_FINITE_SUM, total, total, truncation_k=max_len,
                                 diagnostics={"max_accepted_length": max_len})

    diagnostics: dict = {}
    if isinstance(length, GeometricLengthPrior):
        lam = length.lam
        system = (sp.identity(dfa.state_count, format="csr") - lam * mass_matrix(dfa, label_mass)).tocsc()
        rhs = (1.0 - lam) * accept_vector(dfa)
        solution = spla.spsolve(system, rhs)
        residual = float(np.linalg.norm(system @ solution - rhs))
        value = float(solution[dfa.initial])
        diagnostics.update(residual=residual, sparse_solve=value)
        if residual <= residual_tolerance and -1e-9 <= value <= 1.0 + 1e-9:
            value = min(max(value, 0.0), 1.0)
            return DenominatorResult(CERT_SPARSE_SOLVE, value, value, diagnostics=diagnostics)

    # Certified truncation. With v_n the pi-mass over states after n symbols
    # and s_n its total on productive states, p_n <= s_n and s_n is
    # non-increasing (rows of A sum to <= 1 and no mass re-enters the
    # productive region), so D - D_K <= s_{K+1} * sum_{n>K} rho_n.
    transpose = mass_matrix(dfa, label_mass).transpose().tocsr()
    accept = accept_vector(dfa)
    live = np.zeros(dfa.state_count)
    live[list(productive)] = 1.0
    state_vector = np.zeros(dfa.state_count)
    state_vector[dfa.initial] = 1.0
    terms = [length.rho(0) * float(state_vector @ accept)]
    lower = terms[0]
    k = 0
    while True:
        next_vector = transpose @ state_vector
        live_mass = float(next_vector @ live)
        tail = live_mass * length.tail(k)
        if (lower > 0.0 and tail <= width_tolerance * lower) or k >= max_k:
            break
        k += 1
        state_vector = next_vector
        terms.append(length.rho(k) * float(state_vector @ accept))
        lower += terms[-1]
    lower = math.fsum(terms)
    diagnostics.update(tail_bound=tail, live_mass=live_mass)
    return DenominatorResult(CERT_INTERVAL, lower, min(lower + tail, 1.0), truncation_k=k, diagnostics=diagnostics)
