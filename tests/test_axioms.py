"""The controlled witness cases: LMP's behaviour on known language relations."""

import math

import pytest

import language_mass_precision as lmp

PRIORS = ["geometric", "power_law"]


def scores(toy, log=None, **kwargs):
    frame = lmp.rank_models(log if log is not None else toy.log, toy.models, **kwargs)
    return dict(zip(frame["model"], frame["lower"], strict=True))


@pytest.mark.parametrize("prior", PRIORS)
def test_a1_precision_is_separate_from_fitness(prior):
    values = scores(lmp.datasets.load_toy("precision_fitness"), length_prior=prior)
    # the under-fitting model is perfectly precise: it allows nothing unseen
    assert values["smaller"] == pytest.approx(1.0)
    assert values["exact"] == pytest.approx(1.0)
    assert values["larger"] < 1.0


@pytest.mark.parametrize("prior", PRIORS)
def test_a2_language_inclusion_is_respected_despite_decoys(prior):
    values = scores(lmp.datasets.load_toy("decoy_stress"), length_prior=prior)
    assert values["with_decoys"] == pytest.approx(1.0)
    assert values["larger"] < values["with_decoys"]


@pytest.mark.parametrize("prior", PRIORS)
def test_a3_flower_model_is_strongly_penalized(prior):
    toy = lmp.datasets.load_toy("flower")
    frame = lmp.rank_models(toy.log, toy.models, length_prior=prior).set_index("model")
    assert frame.loc["constrained", "value"] == pytest.approx(1.0)
    assert frame.loc["flower", "upper"] < 0.01


@pytest.mark.parametrize("prior", PRIORS)
def test_a4_language_equivalent_models_score_identically(prior):
    values = scores(lmp.datasets.load_toy("silent_routing"), length_prior=prior)
    assert math.isclose(values["direct"], values["silent"], rel_tol=1e-12)


@pytest.mark.parametrize("prior", PRIORS)
def test_a5_adding_fitting_behaviour_increases_lmp(prior):
    toy = lmp.datasets.load_toy("log_extension")
    model = toy.models["choice"]
    alphabet = set().union(*(set(log["concept:name"]) for log in toy.logs.values())) | {"a", "b", "c", "d", "e"}
    values = [
        lmp.precision_language_mass(
            log, *model, length_prior=prior, length_parameter=0.7 if prior == "geometric" else 2.0, alphabet=alphabet
        ).value
        for log in toy.logs.values()
    ]
    assert values[0] < values[1] < values[2]
