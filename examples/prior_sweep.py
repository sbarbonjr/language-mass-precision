"""Sweep the length prior over compiled models and track ranking stability.

    python examples/prior_sweep.py
"""

import numpy as np
import pm4py
from scipy.stats import kendalltau

import language_mass_precision as lmp

log = [("a", "b")]
models = {
    "short_extra": pm4py.parse_process_tree("X(->('a','b'), 'c')"),
    "superset_of_short": pm4py.parse_process_tree("X(->('a','b'), 'c', ->('d','e'))"),
    "many_long_extra": pm4py.parse_process_tree(
        "X(->('a','b'), ->(X('a','c','d','e'), X('a','c','d','e'), X('c','d','e')))"
    ),
}

estimator = lmp.LanguageMassPrecision(alphabet="abcde").fit(log)
compiled = {name: estimator.compile(model, name=name) for name, model in models.items()}  # compile once

reference = None
for lam in np.linspace(0.3, 0.99, 8):
    estimator.set_params(length_parameter=float(lam))
    values = {name: estimator.score(c).value for name, c in compiled.items()}
    ranking = [sorted(values, key=values.get, reverse=True).index(name) for name in compiled]
    reference = reference or ranking
    tau = kendalltau(reference, ranking).statistic
    print(f"lambda={lam:.2f}  tau={tau:+.2f}  " + "  ".join(f"{n}={v:.4f}" for n, v in values.items()))
