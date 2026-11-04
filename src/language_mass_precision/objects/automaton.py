"""Finite-state representations of a model language.

A Petri net is compiled into an epsilon-NFA over its reachable markings
(:class:`ReachabilityGraph`) and then determinized into a
:class:`DeterministicAutomaton`, the object every exact LMP computation runs on.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

Trace = tuple[str, ...]


@dataclass
class ReachabilityGraph:
    """Epsilon-NFA over reachable markings; ``None`` labels are silent moves."""

    status: str
    initial: int = -1
    accepting: frozenset[int] = frozenset()
    # edges[state] = list of (label_or_None, target_state)
    edges: list[list[tuple[str | None, int]]] = field(default_factory=list)
    state_count: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class DeterministicAutomaton:
    """DFA ``(Q, Sigma_M, delta, q0, F)`` accepting the visible model language."""

    status: str
    initial: int = -1
    accepting: frozenset[int] = frozenset()
    # transitions[state][label] = target_state
    transitions: list[dict[str, int]] = field(default_factory=list)
    state_count: int = 0
    alphabet: frozenset[str] = frozenset()
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def accepts(dfa: DeterministicAutomaton, trace: Iterable[str]) -> bool:
    """Membership test ``trace in L(M)``; raises if the automaton is incomplete."""
    if dfa.status != "ok":
        raise ValueError(f"Automaton is not complete: {dfa.status}")
    state = dfa.initial
    for symbol in trace:
        successors = dfa.transitions[state]
        if symbol not in successors:
            return False
        state = successors[symbol]
    return state in dfa.accepting


def productive_states(dfa: DeterministicAutomaton) -> frozenset[int]:
    """States reachable from the initial state that can still reach acceptance.

    The subgraph induced on these states is the *productive DFA*; when it is
    acyclic the model language is finite and the denominator is a finite sum.
    """
    if dfa.status != "ok":
        return frozenset()
    reverse: dict[int, set[int]] = {state: set() for state in range(dfa.state_count)}
    for state, moves in enumerate(dfa.transitions):
        for target in moves.values():
            reverse[target].add(state)

    reachable = {dfa.initial}
    stack = [dfa.initial]
    while stack:
        current = stack.pop()
        for target in dfa.transitions[current].values():
            if target not in reachable:
                reachable.add(target)
                stack.append(target)

    co_reachable: set[int] = set(dfa.accepting)
    stack = list(dfa.accepting)
    while stack:
        current = stack.pop()
        for predecessor in reverse[current]:
            if predecessor not in co_reachable:
                co_reachable.add(predecessor)
                stack.append(predecessor)

    return frozenset(reachable & co_reachable)


def language_alphabet(dfa: DeterministicAutomaton) -> frozenset[str]:
    """Labels that occur in at least one accepted trace.

    Labels on dead branches are excluded, so language-equivalent models get
    the same alphabet (and therefore the same prior) whatever their structure.
    """
    productive = productive_states(dfa)
    return frozenset(
        label
        for state in productive
        for label, target in dfa.transitions[state].items()
        if target in productive
    )


def _restricted_topological_order(dfa: DeterministicAutomaton, states: frozenset[int]) -> list[int]:
    indegree = {state: 0 for state in states}
    adjacency: dict[int, list[int]] = {state: [] for state in states}
    for state in states:
        for target in dfa.transitions[state].values():
            if target in states:
                adjacency[state].append(target)
                indegree[target] += 1
    queue = deque(state for state in states if indegree[state] == 0)
    order: list[int] = []
    while queue:
        current = queue.popleft()
        order.append(current)
        for target in adjacency[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return order


def is_acyclic(dfa: DeterministicAutomaton, states: frozenset[int]) -> bool:
    """Kahn's algorithm restricted to ``states`` and the edges between them."""
    return len(_restricted_topological_order(dfa, states)) == len(states)


def max_accepted_length(dfa: DeterministicAutomaton, states: frozenset[int]) -> int:
    """Longest accepted trace on an acyclic productive subgraph (edge count)."""
    order = _restricted_topological_order(dfa, states)
    longest = {state: (0 if state == dfa.initial else -1) for state in states}
    best = 0
    for state in order:
        if longest[state] < 0:
            continue
        if state in dfa.accepting:
            best = max(best, longest[state])
        for target in dfa.transitions[state].values():
            if target in states and longest[state] + 1 > longest[target]:
                longest[target] = longest[state] + 1
    return best
