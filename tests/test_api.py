import math
import warnings

import pandas as pd
import pm4py
import pytest
from conftest import branching_finite_model, looping_model

import language_mass_precision as lmp
from language_mass_precision.algo.precision import algorithm
from language_mass_precision.algo.precision.algorithm import AlphabetExtensionWarning

LOG = [("a", "b"), ("a", "b"), ("a", "c", "c")]


def test_log_formats_are_equivalent():
    net, im, fm = branching_finite_model()
    frame = lmp.traces_to_dataframe(LOG)
    event_log = pm4py.convert_to_event_log(frame)
    counts = {("a", "b"): 2, ("a", "c", "c"): 1}
    values = {
        lmp.precision_language_mass(log, net, im, fm).value
        for log in (LOG, frame, event_log, counts)
    }
    assert len(values) == 1


def test_custom_column_names():
    frame = lmp.traces_to_dataframe(LOG).rename(columns={"concept:name": "act", "case:concept:name": "cid"})
    result = lmp.precision_language_mass(frame, *branching_finite_model(), activity_key="act", case_id_key="cid")
    assert result.is_exact


def test_strings_are_rejected_as_traces():
    with pytest.raises(TypeError):
        lmp.precision_language_mass(["ab", "ac"], *branching_finite_model())


def test_model_formats_are_equivalent():
    tree = pm4py.parse_process_tree("->('a', X('b', 'c'))")
    net_model = pm4py.convert_to_petri_net(tree)
    bpmn = pm4py.convert_to_bpmn(tree)
    values = [
        lmp.precision_language_mass(LOG, tree).value,
        lmp.precision_language_mass(LOG, *net_model).value,
        lmp.precision_language_mass(LOG, net_model).value,
        lmp.precision_language_mass(LOG, bpmn).value,
        lmp.precision_language_mass(LOG, lmp.compile_model(tree)).value,
    ]
    assert all(math.isclose(v, values[0], rel_tol=1e-12) for v in values)


def test_exact_value_matches_hand_computation():
    # L(M) = {ab, ac}; observed support in L(M) = {ab}; uniform pi over {a, b, c}
    result = lmp.precision_language_mass(LOG, *branching_finite_model(), length_parameter=0.5)
    assert result.is_exact
    assert result.value == pytest.approx(0.5)
    assert result.accepted_variants == 1 and result.rejected_variants == 1


def test_pm4py_style_algorithm_interface():
    net, im, fm = looping_model()
    result = algorithm.apply(
        LOG, net, im, fm,
        variant=algorithm.Variants.EXACT,
        parameters={algorithm.Parameters.LENGTH_PRIOR: "power_law", algorithm.Parameters.LENGTH_PARAMETER: 2.0},
    )
    assert result.status is lmp.Status.INTERVAL
    assert result.lower <= result.upper


def test_result_float_conversion():
    exact = lmp.precision_language_mass(LOG, *branching_finite_model())
    assert float(exact) == exact.value
    interval = lmp.precision_language_mass(LOG, *looping_model(), length_prior="power_law")
    with pytest.raises(ValueError, match="certified within"):
        float(interval)
    assert interval.lower <= interval.midpoint <= interval.upper


def test_incomplete_and_undefined_statuses():
    incomplete = lmp.precision_language_mass(LOG, *branching_finite_model(), state_limit=1)
    assert incomplete.status is lmp.Status.INCOMPLETE and incomplete.value is None
    net, im, fm = branching_finite_model()
    unreachable = pm4py.objects.petri_net.obj.Marking()
    unreachable[next(p for p in net.places if p.name == "p0")] = 2
    undefined = lmp.precision_language_mass(LOG, net, im, unreachable)
    assert undefined.status is lmp.Status.UNDEFINED


def test_playout_converges_to_exact_on_infinite_language():
    net, im, fm = looping_model()
    log = [("a", "c"), ("a", "b", "c")]
    exact = lmp.precision_language_mass(log, net, im, fm, length_parameter=0.6)
    errors = []
    for k in (3, 6, 12):
        bounded = lmp.precision_language_mass(
            log, net, im, fm, length_parameter=0.6, method="playout", max_trace_length=k
        )
        assert bounded.status is lmp.Status.BOUNDED
        errors.append(bounded.value - exact.value)
    assert all(e >= 0 for e in errors)  # truncation drops unseen mass, so playout over-estimates
    assert errors[0] > errors[1] > errors[2]


def test_attribution_shares_sum_to_imprecision_on_finite_language():
    net, im, fm = branching_finite_model()
    result = lmp.precision_language_mass(LOG, net, im, fm)
    table = lmp.imprecision_attribution(LOG, net, im, fm, top_k=10)
    assert list(table["trace"]) == [("a", "c")]
    assert table["cumulative_share"].iloc[-1] == pytest.approx(result.imprecision)


def test_attribution_is_mass_ordered_on_infinite_language():
    table = lmp.imprecision_attribution([("a", "c")], *looping_model(), top_k=5)
    assert list(table["length"]) == [3, 4, 5, 6, 7]
    assert table["trace_mass"].is_monotonic_decreasing


def test_declared_alphabet_extension_warns():
    estimator = lmp.LanguageMassPrecision(alphabet={"a", "b"}).fit([("a", "b")])
    with pytest.warns(AlphabetExtensionWarning):
        estimator.score(*branching_finite_model())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lmp.LanguageMassPrecision(alphabet={"a", "b", "c"}).fit([("a", "b")]).score(*branching_finite_model())


def test_rank_models_uses_one_shared_alphabet():
    frame = lmp.rank_models(LOG, {"finite": branching_finite_model(), "loop": looping_model()})
    assert isinstance(frame, pd.DataFrame)
    assert frame["prior_alphabet_size"].nunique() == 1
    assert set(frame["rank"]) == {1, 2}
