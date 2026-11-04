"""Compile an accepting Petri net into a DFA for its visible language.

Reachable markings form an epsilon-NFA in which silent transitions are
epsilon edges and duplicate labels are ordinary nondeterminism; subset
construction then yields a DFA. Both phases are bounded by state and time
limits and report a failure status instead of returning a partial automaton,
because a silently truncated automaton would produce a wrong "exact" value.
"""

from __future__ import annotations

import time
from collections import deque
from enum import Enum
from typing import Any

from pm4py.util import exec_utils

from language_mass_precision.objects.automaton import DeterministicAutomaton, ReachabilityGraph
from language_mass_precision.objects.result import CompiledModel

CanonicalMarking = tuple[tuple[int, int], ...]

DEFAULT_STATE_LIMIT = 200_000
DEFAULT_TIME_LIMIT_SECONDS = 120.0


class Parameters(Enum):
    STATE_LIMIT = "state_limit"
    TIME_LIMIT = "time_limit"


def _canonical(counts: dict[int, int]) -> CanonicalMarking:
    return tuple(sorted((place, count) for place, count in counts.items() if count))


def build_reachability_graph(
    net: Any,
    initial_marking: Any,
    final_marking: Any,
    state_limit: int = DEFAULT_STATE_LIMIT,
    time_limit_seconds: float = DEFAULT_TIME_LIMIT_SECONDS,
) -> ReachabilityGraph:
    """Breadth-first exploration of reachable markings as an epsilon-NFA.

    A marking is accepting iff it equals the final marking exactly.
    """
    places = sorted(net.places, key=lambda place: place.name)
    place_index = {place: index for index, place in enumerate(places)}

    transitions = []
    for transition in sorted(net.transitions, key=lambda t: t.name):
        inputs = [(place_index[arc.source], arc.weight) for arc in transition.in_arcs]
        outputs = [(place_index[arc.target], arc.weight) for arc in transition.out_arcs]
        transitions.append((transition.label, inputs, outputs))

    def marking_counts(marking: Any) -> dict[int, int]:
        return {place_index[place]: count for place, count in marking.items() if count}

    start = _canonical(marking_counts(initial_marking))
    target = _canonical(marking_counts(final_marking))

    state_id: dict[CanonicalMarking, int] = {start: 0}
    edges: list[list[tuple[str | None, int]]] = [[]]
    accepting: set[int] = set()
    if start == target:
        accepting.add(0)

    started = time.perf_counter()
    frontier: deque[CanonicalMarking] = deque([start])
    while frontier:
        if time.perf_counter() - started > time_limit_seconds:
            return ReachabilityGraph(
                status="reachability_timeout", state_count=len(state_id), elapsed_seconds=time.perf_counter() - started
            )
        marking = frontier.popleft()
        source_id = state_id[marking]
        current = dict(marking)
        for label, inputs, outputs in transitions:
            if not all(current.get(place, 0) >= weight for place, weight in inputs):
                continue
            successor = dict(current)
            for place, weight in inputs:
                successor[place] -= weight
            for place, weight in outputs:
                successor[place] = successor.get(place, 0) + weight
            canonical_successor = _canonical(successor)
            target_id = state_id.get(canonical_successor)
            if target_id is None:
                if len(state_id) >= state_limit:
                    return ReachabilityGraph(
                        status="reachability_state_limit",
                        state_count=len(state_id),
                        elapsed_seconds=time.perf_counter() - started,
                    )
                target_id = len(state_id)
                state_id[canonical_successor] = target_id
                edges.append([])
                if canonical_successor == target:
                    accepting.add(target_id)
                frontier.append(canonical_successor)
            edges[source_id].append((label, target_id))

    return ReachabilityGraph(
        status="ok",
        initial=0,
        accepting=frozenset(accepting),
        edges=edges,
        state_count=len(state_id),
        elapsed_seconds=time.perf_counter() - started,
    )


def _epsilon_closures(graph: ReachabilityGraph) -> list[frozenset[int]]:
    closures: list[frozenset[int]] = [frozenset()] * graph.state_count
    for state in range(graph.state_count):
        visited = {state}
        stack = [state]
        while stack:
            current = stack.pop()
            for label, target in graph.edges[current]:
                if label is None and target not in visited:
                    visited.add(target)
                    stack.append(target)
        closures[state] = frozenset(visited)
    return closures


def determinize(
    graph: ReachabilityGraph,
    state_limit: int = DEFAULT_STATE_LIMIT,
    time_limit_seconds: float = DEFAULT_TIME_LIMIT_SECONDS,
) -> DeterministicAutomaton:
    """Subset construction over the epsilon-NFA.

    Only labels that actually leave a subset are explored, so the DFA alphabet
    is the model's own visible label set.
    """
    if graph.status != "ok":
        return DeterministicAutomaton(status=graph.status)

    closures = _epsilon_closures(graph)
    start = closures[graph.initial]

    subset_id: dict[frozenset[int], int] = {start: 0}
    transitions: list[dict[str, int]] = [{}]
    accepting: set[int] = set()
    if start & graph.accepting:
        accepting.add(0)
    alphabet: set[str] = set()

    started = time.perf_counter()
    frontier: deque[frozenset[int]] = deque([start])
    while frontier:
        if time.perf_counter() - started > time_limit_seconds:
            return DeterministicAutomaton(status="determinization_timeout", state_count=len(subset_id))
        subset = frontier.popleft()
        source_id = subset_id[subset]
        moves: dict[str, set[int]] = {}
        for nfa_state in subset:
            for label, target in graph.edges[nfa_state]:
                if label is not None:
                    moves.setdefault(label, set()).add(target)
        for label in sorted(moves):
            alphabet.add(label)
            successor = frozenset().union(*(closures[state] for state in moves[label]))
            target_id = subset_id.get(successor)
            if target_id is None:
                if len(subset_id) >= state_limit:
                    return DeterministicAutomaton(status="determinization_state_limit", state_count=len(subset_id))
                target_id = len(subset_id)
                subset_id[successor] = target_id
                transitions.append({})
                if successor & graph.accepting:
                    accepting.add(target_id)
                frontier.append(successor)
            transitions[source_id][label] = target_id

    return DeterministicAutomaton(
        status="ok",
        initial=0,
        accepting=frozenset(accepting),
        transitions=transitions,
        state_count=len(subset_id),
        alphabet=frozenset(alphabet),
        elapsed_seconds=time.perf_counter() - started,
    )


def apply(
    net: Any,
    initial_marking: Any,
    final_marking: Any,
    parameters: dict[Any, Any] | None = None,
) -> CompiledModel:
    """PM4Py-style entry point: ``(net, im, fm) -> CompiledModel``."""
    parameters = parameters or {}
    state_limit = exec_utils.get_param_value(Parameters.STATE_LIMIT, parameters, DEFAULT_STATE_LIMIT)
    time_limit = exec_utils.get_param_value(Parameters.TIME_LIMIT, parameters, DEFAULT_TIME_LIMIT_SECONDS)

    started = time.perf_counter()
    graph = build_reachability_graph(net, initial_marking, final_marking, state_limit, time_limit)
    remaining = max(time_limit - graph.elapsed_seconds, 0.0)
    dfa = determinize(graph, state_limit, remaining)
    return CompiledModel(
        dfa=dfa,
        reachability_states=graph.state_count,
        compile_seconds=time.perf_counter() - started,
        name=getattr(net, "name", None) or None,
    )
