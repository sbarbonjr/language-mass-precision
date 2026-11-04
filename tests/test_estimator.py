import pytest
from conftest import branching_finite_model, looping_model

import language_mass_precision as lmp

LOG = [("a", "b"), ("a", "b", "b", "c"), ("a", "c")]


def test_unfitted_estimator_raises():
    with pytest.raises(lmp.NotFittedError):
        lmp.LanguageMassPrecision().score(*looping_model())


def test_fitted_attributes():
    estimator = lmp.LanguageMassPrecision().fit(LOG)
    assert estimator.variants_ == frozenset(LOG)
    assert estimator.alphabet_ == frozenset("abc")
    assert estimator.median_length_ == 2.0
    assert estimator.n_cases_ == 3
    assert isinstance(estimator.length_prior_, lmp.GeometricLengthPrior)


def test_get_set_params_roundtrip():
    estimator = lmp.LanguageMassPrecision(length_prior="power_law")
    params = estimator.get_params()
    assert params["length_prior"] == "power_law"
    assert estimator.set_params(length_parameter=3.0) is estimator
    assert estimator.length_parameter == 3.0
    with pytest.raises(ValueError):
        estimator.set_params(nonsense=1)
    assert "power_law" in repr(estimator)


def test_sweep_reuses_compiled_model():
    estimator = lmp.LanguageMassPrecision().fit(LOG)
    compiled = estimator.compile(*looping_model(), name="loop")
    values = []
    for lam in (0.3, 0.5, 0.7, 0.9):
        values.append(estimator.set_params(length_parameter=lam).score(compiled).value)
    assert len(set(values)) == 4


def test_score_many_ranks_models():
    estimator = lmp.LanguageMassPrecision().fit(LOG)
    frame = estimator.score_many({"finite": branching_finite_model(), "loop": looping_model()})
    assert list(frame.columns[:2]) == ["model", "rank"]
    assert len(frame) == 2


def test_sklearn_compatibility():
    base = pytest.importorskip("sklearn.base")
    model_selection = pytest.importorskip("sklearn.model_selection")
    estimator = lmp.LanguageMassPrecision(length_prior="power_law")
    clone = base.clone(estimator)
    assert clone.get_params() == estimator.get_params()
    grid = model_selection.ParameterGrid(
        {"length_prior": ["geometric", "power_law"], "activity_prior": ["uniform", "empirical"]}
    )
    compiled = lmp.compile_model(*looping_model())
    fitted = lmp.LanguageMassPrecision().fit(LOG)
    results = [fitted.set_params(**params).score(compiled) for params in grid]
    assert all(r.has_bounds for r in results)
