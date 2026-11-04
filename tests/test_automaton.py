"""Regression tests for the exact engine (ported from the paper pipeline)."""

import math

import pytest
from conftest import branching_finite_model, dead_branch_model, flower_model, looping_model, silent_branching_model

from language_mass_precision.algo import denominator
from language_mass_precision.algo.conversion import to_automaton
from language_mass_precision.objects.automaton import (
    accepts,
    is_acyclic,
    language_alphabet,
    max_accepted_length,
    productive_states,
)
from language_mass_precision.objects.priors import (
    GeometricLengthPrior,
    LengthPrior,
    PowerLawLengthPrior,
    TracePrior,
    alpha_from_median_length,
    lambda_from_median_length,
)


def dfa_for(net, im, fm):
    compiled = to_automaton.apply(net, im, fm)
    assert compiled.ok
    return compiled.dfa


class GenericGeometric(LengthPrior):
    """Geometric rho without the sparse-solve shortcut, to exercise the interval path."""

    name = "generic_geometric"

    def __init__(self, lam):
        self._g = GeometricLengthPrior(lam)

    @property
    def parameter(self):
        return self._g.lam

    def log_rho(self, n):
        return self._g.log_rho(n)

    def tail(self, k):
        return self._g.tail(k)


@pytest.mark.parametrize("length", [GeometricLengthPrior(0.7), PowerLawLengthPrior(2.0)])
def test_flower_denominator_equals_one_for_both_normalized_priors(length):
    dfa = dfa_for(*flower_model("abc"))
    result = denominator.compute(dfa, TracePrior.uniform(length, "abc"), width_tolerance=1e-4, max_k=100_000)
    assert result.lower <= 1.0 <= result.upper + 1e-12
    if isinstance(length, GeometricLengthPrior):
        assert result.certification == denominator.CERT_SPARSE_SOLVE
        assert math.isclose(result.lower, 1.0, rel_tol=1e-9)


def test_finite_model_denominator_equals_explicit_language_sum():
    dfa = dfa_for(*branching_finite_model())
    prior = TracePrior.uniform(GeometricLengthPrior(0.6), "abc")
    result = denominator.compute(dfa, prior)
    assert result.certification == denominator.CERT_FINITE_SUM
    assert math.isclose(result.lower, prior.language_mass([("a", "b"), ("a", "c")]), rel_tol=1e-12)


def test_membership_through_silent_transitions():
    dfa = dfa_for(*silent_branching_model())
    assert accepts(dfa, ("a", "b"))
    assert accepts(dfa, ("a", "c"))
    assert not accepts(dfa, ("a",))
    assert not accepts(dfa, ("a", "b", "c"))


def test_bounded_length_masses_match_known_language():
    dfa = dfa_for(*looping_model())
    masses = denominator.bounded_length_masses(dfa, {a: 1 / 3 for a in "abc"}, 5)
    # exactly one accepted trace of each length n >= 2: a b^(n-2) c
    assert masses[:2] == [0.0, 0.0]
    for n in range(2, 6):
        assert math.isclose(masses[n], 3.0 ** -n, rel_tol=1e-12)


def test_certified_interval_contains_exact_geometric_value():
    dfa = dfa_for(*looping_model())
    exact = denominator.compute(dfa, TracePrior.uniform(GeometricLengthPrior(0.8), "abcd"))
    assert exact.certification == denominator.CERT_SPARSE_SOLVE
    previous_width = math.inf
    for tolerance in (1e-2, 1e-4, 1e-8):
        prior = TracePrior.uniform(GenericGeometric(0.8), "abcd")
        interval = denominator.compute(dfa, prior, width_tolerance=tolerance)
        assert interval.certification == denominator.CERT_INTERVAL
        assert interval.lower <= exact.lower <= interval.upper
        assert interval.upper - interval.lower <= previous_width
        previous_width = interval.upper - interval.lower


def test_power_law_interval_is_tight_for_cyclic_model():
    dfa = dfa_for(*looping_model())
    result = denominator.compute(dfa, TracePrior.uniform(PowerLawLengthPrior(1.5), "abcd"))
    assert result.certification == denominator.CERT_INTERVAL
    assert result.upper - result.lower <= 1e-6 * result.lower
    assert result.truncation_k < 100


def test_language_equivalent_models_have_identical_denominators():
    prior = TracePrior.uniform(GeometricLengthPrior(0.7), "abc")
    plain = denominator.compute(dfa_for(*branching_finite_model()), prior)
    silent = denominator.compute(dfa_for(*silent_branching_model()), prior)
    assert math.isclose(plain.lower, silent.lower, rel_tol=1e-12)


def test_dead_branches_do_not_enter_the_language_alphabet():
    dfa = dfa_for(*dead_branch_model())
    assert "z" in dfa.alphabet
    assert language_alphabet(dfa) == frozenset("abc")
    # the prior does not need to know about 'z'
    denominator.compute(dfa, TracePrior.uniform(GeometricLengthPrior(0.7), "abc"))


def test_productive_acyclic_structure():
    finite = dfa_for(*branching_finite_model())
    assert is_acyclic(finite, productive_states(finite))
    assert max_accepted_length(finite, productive_states(finite)) == 2
    loop = dfa_for(*looping_model())
    assert not is_acyclic(loop, productive_states(loop))


def test_calibration_puts_half_the_mass_below_the_median():
    for median in (2.0, 7.0, 12.0):
        geometric = GeometricLengthPrior(lambda_from_median_length(median))
        assert math.isclose(geometric.cdf(int(median)), 0.5, rel_tol=1e-9)
        power = PowerLawLengthPrior(alpha_from_median_length(median))
        assert math.isclose(power.cdf(int(median)), 0.5, rel_tol=1e-6)


def test_state_limit_is_reported_instead_of_a_partial_automaton():
    compiled = to_automaton.apply(*branching_finite_model(), parameters={to_automaton.Parameters.STATE_LIMIT: 1})
    assert compiled.status == "reachability_state_limit"
    with pytest.raises(ValueError):
        accepts(compiled.dfa, ("a", "b"))
