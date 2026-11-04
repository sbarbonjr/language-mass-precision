"""Small controlled examples with known language relations.

Each toy pairs a log with models whose language relation to the log is known
by construction, so LMP's expected behaviour can be checked by eye. They are
the witness cases of the paper's controlled evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import pandas as pd
import pm4py

_TOYS: dict[str, dict[str, Any]] = {
    "precision_fitness": {
        "description": "Precision vs fitness: 'smaller' misses a log variant, 'exact' equals the log "
                       "language, 'larger' adds unseen behaviour.",
        "logs": {"log": "log.csv"},
        "models": {"smaller": "smaller.pnml", "exact": "exact.pnml", "larger": "larger.pnml"},
    },
    "flower": {
        "description": "Overgeneralization: a model constrained to the log against a flower model "
                       "accepting every trace over {a, b, c, d}.",
        "logs": {"log": "log.csv"},
        "models": {"constrained": "constrained.pnml", "flower": "flower.pnml"},
    },
    "silent_routing": {
        "description": "Language equivalence: a direct choice and a silent-routing model with the "
                       "same visible language.",
        "logs": {"log": "log.csv"},
        "models": {"direct": "direct.pnml", "silent": "silent.pnml"},
    },
    "log_extension": {
        "description": "Adding fitting behaviour: one model, three nested logs L1 < L2 < L3.",
        "logs": {"L1": "log_L1.csv", "L2": "log_L2.csv", "L3": "log_L3.csv"},
        "models": {"choice": "choice.pnml"},
    },
    "decoy_stress": {
        "description": "Language inclusion with structural decoys: 'with_decoys' has the log language "
                       "plus dead visible branches; 'larger' is clean but strictly larger.",
        "logs": {"log": "log.csv"},
        "models": {"with_decoys": "with_decoys.pnml", "larger": "larger.pnml"},
    },
}


@dataclass
class ToyExample:
    name: str
    description: str
    logs: dict[str, pd.DataFrame]
    models: dict[str, tuple[Any, Any, Any]] = field(repr=False)

    @property
    def log(self) -> pd.DataFrame:
        """The first (or only) log of the example."""
        return next(iter(self.logs.values()))

    def __repr__(self) -> str:
        return f"ToyExample({self.name!r}, logs={list(self.logs)}, models={list(self.models)})"


def list_toys() -> dict[str, str]:
    """Names and one-line descriptions of the bundled examples."""
    return {name: spec["description"] for name, spec in _TOYS.items()}


def load_toy(name: str) -> ToyExample:
    """Load a bundled example as PM4Py DataFrames and ``(net, im, fm)`` triples."""
    if name not in _TOYS:
        raise KeyError(f"Unknown toy {name!r}; available: {sorted(_TOYS)}")
    spec = _TOYS[name]
    root = resources.files(__package__) / "data" / name
    logs = {}
    for key, filename in spec["logs"].items():
        with resources.as_file(root / filename) as path:
            frame = pd.read_csv(path)
        frame["time:timestamp"] = pd.to_datetime(frame["time:timestamp"])
        logs[key] = frame
    models = {}
    for key, filename in spec["models"].items():
        with resources.as_file(root / filename) as path:
            models[key] = pm4py.read_pnml(str(path))
    return ToyExample(name, spec["description"], logs, models)
