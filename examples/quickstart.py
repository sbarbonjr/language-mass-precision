"""Score and rank models on a toy log.

    python examples/quickstart.py
"""

import pm4py

import language_mass_precision as lmp

log = [("a", "b", "c"), ("a", "d", "c"), ("a", "b", "c")]

models = {
    "exact": pm4py.parse_process_tree("->('a', X('b', 'd'), 'c')"),
    "extra_e": pm4py.parse_process_tree("->('a', X('b', 'd', 'e'), 'c')"),
    "loop": pm4py.parse_process_tree("->('a', *(X('b', 'd'), tau), 'c')"),
    "flower": pm4py.parse_process_tree("*(tau, X('a', 'b', 'c', 'd', 'e'))"),
}

print(lmp.precision_language_mass(log, models["extra_e"]))
print()
columns = ["model", "rank", "status", "value", "lower", "upper", "certification"]
for prior in ("geometric", "power_law"):
    print(f"--- {prior} prior")
    print(lmp.rank_models(log, models, length_prior=prior)[columns].to_string(index=False))
print()
print("Heaviest unseen traces of the loop model:")
print(lmp.imprecision_attribution(log, models["loop"], top_k=5)[["trace", "imprecision_share"]].to_string(index=False))
