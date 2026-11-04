"""Turn any supported log representation into the statistics LMP needs.

LMP only uses the *support* of the log (its distinct variants); variant
frequencies never enter the score. Activity counts are kept so that an
empirical activity prior can be built on request.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import median
from typing import Any

import pandas as pd
from pm4py.objects.log.obj import EventLog
from pm4py.util import xes_constants

Trace = tuple[str, ...]

DEFAULT_ACTIVITY_KEY = xes_constants.DEFAULT_NAME_KEY
DEFAULT_CASE_ID_KEY = "case:" + xes_constants.DEFAULT_TRACEID_KEY
DEFAULT_TIMESTAMP_KEY = xes_constants.DEFAULT_TIMESTAMP_KEY


@dataclass(frozen=True)
class LogStatistics:
    variant_counts: Counter
    activity_counts: Counter

    @property
    def variants(self) -> frozenset[Trace]:
        return frozenset(self.variant_counts)

    @property
    def alphabet(self) -> frozenset[str]:
        return frozenset(self.activity_counts)

    @property
    def median_variant_length(self) -> float:
        """Median length over distinct variants, the default prior calibration target."""
        if not self.variant_counts:
            raise ValueError("Cannot compute a median length for an empty log")
        return float(median(len(trace) for trace in self.variant_counts))

    @property
    def n_cases(self) -> int:
        return sum(self.variant_counts.values())


def _variants_from_dataframe(frame: pd.DataFrame, activity_key: str, case_id_key: str, timestamp_key: str) -> Counter:
    for key in (case_id_key, activity_key):
        if key not in frame.columns:
            raise KeyError(f"Column {key!r} not found; pass activity_key/case_id_key to match your log")
    if timestamp_key in frame.columns:
        frame = frame.sort_values([case_id_key, timestamp_key], kind="mergesort")
    traces = frame.groupby(case_id_key, sort=False)[activity_key].agg(lambda s: tuple(map(str, s)))
    return Counter(traces.tolist())


def _variants_from_event_log(log: EventLog, activity_key: str) -> Counter:
    return Counter(tuple(str(event[activity_key]) for event in trace) for trace in log)


def _as_trace(trace: Any) -> Trace:
    if isinstance(trace, str):
        raise TypeError(
            f"Got the string {trace!r} as a trace; traces are sequences of activities, e.g. ('a', 'b', 'c')"
        )
    return tuple(str(activity) for activity in trace)


def variant_counts(
    log: Any,
    activity_key: str = DEFAULT_ACTIVITY_KEY,
    case_id_key: str = DEFAULT_CASE_ID_KEY,
    timestamp_key: str = DEFAULT_TIMESTAMP_KEY,
) -> Counter:
    """Variant frequencies of a log.

    Accepts a pandas DataFrame in PM4Py format, a PM4Py ``EventLog``, a mapping
    ``{trace: count}``, or an iterable of traces (each a sequence of activity names).
    """
    if isinstance(log, pd.DataFrame):
        return _variants_from_dataframe(log, activity_key, case_id_key, timestamp_key)
    if isinstance(log, EventLog):
        return _variants_from_event_log(log, activity_key)
    if isinstance(log, Mapping):
        return Counter({_as_trace(trace): int(count) for trace, count in log.items()})
    if isinstance(log, Iterable) and not isinstance(log, str):
        return Counter(_as_trace(trace) for trace in log)
    raise TypeError(f"Unsupported log type: {type(log).__name__}")


def log_statistics(
    log: Any,
    activity_key: str = DEFAULT_ACTIVITY_KEY,
    case_id_key: str = DEFAULT_CASE_ID_KEY,
    timestamp_key: str = DEFAULT_TIMESTAMP_KEY,
) -> LogStatistics:
    if isinstance(log, LogStatistics):
        return log
    counts = variant_counts(log, activity_key, case_id_key, timestamp_key)
    activities: Counter = Counter()
    for trace, count in counts.items():
        for activity in trace:
            activities[activity] += count
    return LogStatistics(variant_counts=counts, activity_counts=activities)


def traces_to_dataframe(
    traces: Iterable[Iterable[str]] | Mapping[Iterable[str], int],
    activity_key: str = DEFAULT_ACTIVITY_KEY,
    case_id_key: str = DEFAULT_CASE_ID_KEY,
    timestamp_key: str = DEFAULT_TIMESTAMP_KEY,
) -> pd.DataFrame:
    """Build a PM4Py-format DataFrame from traces, handy for toy logs.

    A mapping ``{trace: count}`` expands every trace ``count`` times.
    """
    if isinstance(traces, Mapping):
        expanded = [trace for trace, count in traces.items() for _ in range(int(count))]
    else:
        expanded = list(traces)
    base = pd.Timestamp("2026-01-01")
    rows = []
    for case_index, trace in enumerate(expanded):
        for position, activity in enumerate(_as_trace(trace)):
            rows.append({
                case_id_key: f"case_{case_index:05d}",
                activity_key: activity,
                timestamp_key: base + pd.Timedelta(days=case_index, minutes=position),
            })
    return pd.DataFrame(rows, columns=[case_id_key, activity_key, timestamp_key])
