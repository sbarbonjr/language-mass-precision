"""Discover models from an event log and rank them by LMP and PM4Py precision.

    python examples/rank_discovered_models.py path/to/log.xes [--prior power_law]
"""

import argparse

import pm4py

import language_mass_precision as lmp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", help="XES (or .xes.gz) event log")
    parser.add_argument("--prior", default="geometric", choices=["geometric", "power_law"])
    parser.add_argument("--state-limit", type=int, default=200_000)
    parser.add_argument("--no-alignments", action="store_true", help="skip alignment precision (slow on large logs)")
    args = parser.parse_args()

    log = pm4py.read_xes(args.log, return_legacy_log_object=False)
    models = {f"IMf {n:.1f}": pm4py.discover_petri_net_inductive(log, noise_threshold=n) for n in (0.0, 0.2, 0.4)}
    models["heuristics"] = pm4py.discover_petri_net_heuristics(log)

    table = lmp.rank_models(log, models, length_prior=args.prior, state_limit=args.state_limit)
    table = table.set_index("model")[["rank", "status", "value", "lower", "upper", "certification", "dfa_states"]]
    table["token_precision"] = [pm4py.precision_token_based_replay(log, *models[m]) for m in table.index]
    if not args.no_alignments:
        table["alignment_precision"] = [pm4py.precision_alignments(log, *models[m]) for m in table.index]
    print(table.sort_values("rank").to_string())


if __name__ == "__main__":
    main()
