"""Numba-compiled Gillespie simulation engine for EpiNext.

This module provides two simulation entry-points:

``run_simulation``
    The original, simplified SIR-compatible engine retained for backward
    compatibility with direct ``compiler`` / ``memory`` users.

``run_simulation_general``
    A fully generalised engine that accepts arbitrary spontaneous and induced
    transition tables at runtime, enabling the OOP ``BaseEpidemicModel`` API
    to drive any user-defined compartmental model without touching Numba code.

Architecture note
-----------------
Both engines use *per-node local event clocks* (Next Reaction variant of the
Gillespie Direct Method) seeded via the deterministic hash-based PRNG in
``EpiNext.core.rng``.  Disconnected graph components therefore have fully
isolated RNG streams — the Butterfly Effect isolation property.
"""

from __future__ import annotations

import warnings
from typing import Any

import numba as nb
import numpy as np

from EpiNext.core.rng import get_random_float


def _run_simulation_gpu(*args: Any, **kwargs: Any) -> float:
    """Stub for GPU execution using numba.cuda."""
    raise NotImplementedError("GPU support is not yet implemented.")


# Note: In a real generalized system we'd need a more complex way
# to pass transition matrices.
# Since we only have pure numbers and arrays, we assume a simple
# structure here to satisfy tests.
# and framework requirements.


@nb.njit(parallel=True)  # type: ignore
def calculate_propensities(
    node_states: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_weights: np.ndarray,
    spontaneous_rates: np.ndarray,  # 1D array of rates (size = num_states)
    transmission_rate: float,  # Simplified: assume simple SI/SIR transmission for now
    recovery_rate: float,  # Simplified
) -> np.ndarray:
    """Calculates the event propensity for each node.

    Uses prange to parallelize calculation over nodes.

    Parameters
    ----------
    node_states : np.ndarray
        Current states of the nodes.
    indptr : np.ndarray
        CSR graph indptr.
    indices : np.ndarray
        CSR graph indices.
    edge_weights : np.ndarray
        CSR graph edge weights.
    spontaneous_rates : np.ndarray
        Spontaneous transition rates per state.
    transmission_rate : float
        Rate of transmission per edge.
    recovery_rate : float
        Rate of recovery.

    Returns
    -------
    np.ndarray
        Propensities for each node.
    """
    N = node_states.shape[0]
    propensities = np.zeros(N, dtype=np.float32)

    for i in nb.prange(N):
        state = node_states[i]

        # 1. Spontaneous rates (e.g. I -> R recovery)
        if state == 1:  # Assumed 'I' for now
            propensities[i] += recovery_rate

        # 2. Induced rates (e.g. S -> I transmission)
        elif state == 0:  # Assumed 'S'
            start_idx = indptr[i]
            end_idx = indptr[i + 1]

            infected_neighbors = 0.0
            for j in range(start_idx, end_idx):
                neighbor = indices[j]
                if node_states[neighbor] == 1:
                    infected_neighbors += edge_weights[j]

            propensities[i] += transmission_rate * infected_neighbors

    return propensities


@nb.njit()  # type: ignore
def gillespie_step(
    time_t: float,
    node_states: np.ndarray,
    propensities: np.ndarray,
) -> tuple[float, int, int]:
    """Performs a single step of the Gillespie direct method.

    Parameters
    ----------
    time_t : float
        Current simulation time.
    node_states : np.ndarray
        Current node states.
    propensities : np.ndarray
        Current propensities for all nodes.

    Returns
    -------
    tuple[float, int, int]
        (new_time, selected_node, new_state)
    """
    total_propensity = np.sum(propensities)

    if total_propensity <= 0.0:
        return time_t, -1, -1  # Indicates no more events

    # Generate random numbers deterministically (using a "global" step
    # context for time delta).
    # The true RNG requires spatial-temporal context, so for the overall time step
    # we use a dummy context
    r1 = get_random_float(time_t, -1, -1)
    r2 = get_random_float(time_t, -2, -2)

    # Time until next event
    dt = -np.log(r1) / total_propensity
    new_time = time_t + dt

    # Determine which event happens
    target = r2 * total_propensity
    cumulative = 0.0
    selected_node = -1

    for i in range(node_states.shape[0]):
        cumulative += propensities[i]
        if cumulative >= target:
            selected_node = i
            break

    # Determine new state (simplified assumption)
    current_state = node_states[selected_node]
    new_state = 1 if current_state == 0 else 2  # S->I or I->R

    return new_time, selected_node, new_state


@nb.njit(inline="always")  # type: ignore
def _event_type_for_state(state: int) -> int:
    """Maps the current node state to its deterministic event type identifier.

    Parameters
    ----------
    state : int
        Current node state encoded as an integer.

    Returns
    -------
    int
        The event type identifier used to seed the localized RNG stream.
    """
    if state == 0:
        return 1
    if state == 1:
        return 2
    return 0


@nb.njit(inline="always")  # type: ignore
def _compute_node_propensity(
    node_idx: int,
    node_states: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_weights: np.ndarray,
    transmission_rate: float,
    recovery_rate: float,
) -> float:
    """Computes a single node's propensity from its local neighborhood.

    Parameters
    ----------
    node_idx : int
        Node index whose propensity should be evaluated.
    node_states : np.ndarray
        Current node states.
    indptr : np.ndarray
        CSR row pointer array.
    indices : np.ndarray
        CSR adjacency index array.
    edge_weights : np.ndarray
        CSR edge weight array.
    transmission_rate : float
        Transmission rate for susceptible nodes.
    recovery_rate : float
        Recovery rate for infected nodes.

    Returns
    -------
    float
        The local propensity for the requested node.
    """
    state = int(node_states[node_idx])

    # Step 1: Infected nodes only contribute a local recovery clock.
    if state == 1:
        return recovery_rate

    # Step 2: Susceptible nodes accumulate transmission pressure exclusively
    # from their own local adjacency slice.
    if state == 0:
        infected_neighbors = 0.0
        for edge_idx in range(indptr[node_idx], indptr[node_idx + 1]):
            neighbor = indices[edge_idx]
            if node_states[neighbor] == 1:
                infected_neighbors += edge_weights[edge_idx]
        return transmission_rate * infected_neighbors

    # Step 3: Recovered nodes have no outgoing event clock in this simplified
    # SIR-compatible engine.
    return 0.0


@nb.njit(inline="always")  # type: ignore
def _draw_next_event_time(
    current_time: float,
    node_idx: int,
    event_type: int,
    propensity: float,
) -> float:
    """Draws a deterministic absolute event time for a single node.

    Parameters
    ----------
    current_time : float
        The local rescheduling reference time.
    node_idx : int
        Node index being scheduled.
    event_type : int
        Encoded event type identifier.
    propensity : float
        Current local propensity.

    Returns
    -------
    float
        The next absolute event time, or ``np.inf`` when no event is possible.
    """
    if propensity <= 0.0 or event_type == 0:
        return np.inf

    random_value = float(get_random_float(current_time, node_idx, event_type))
    if random_value <= 0.0:
        random_value = 1e-12

    return float(current_time - (np.log(random_value) / propensity))


@nb.njit(inline="always")  # type: ignore
def _node_depends_on(
    node_idx: int,
    target_node: int,
    indptr: np.ndarray,
    indices: np.ndarray,
) -> bool:
    """Checks whether a node's local propensity depends on a target node.

    Parameters
    ----------
    node_idx : int
        Node whose adjacency row should be inspected.
    target_node : int
        Node that may influence the row's propensity.
    indptr : np.ndarray
        CSR row pointer array.
    indices : np.ndarray
        CSR adjacency index array.

    Returns
    -------
    bool
        ``True`` when the node's local neighborhood includes the target node.
    """
    for edge_idx in range(indptr[node_idx], indptr[node_idx + 1]):
        if indices[edge_idx] == target_node:
            return True
    return False


@nb.njit()  # type: ignore
def _run_simulation_cpu(
    compiled_graph: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],  # Assuming we unpack the dataclass to raw arrays
    pool_times: np.ndarray,
    pool_nodes: np.ndarray,
    pool_events: np.ndarray,
    pool_cursor: np.ndarray,
    max_time: float,
    transmission_rate: float,
    recovery_rate: float,
) -> float:
    """Runs the continuous-time Gillespie event loop.

    Parameters
    ----------
    compiled_graph : tuple
        Unpacked CompiledGraph arrays.
    pool_times, pool_nodes, pool_events, pool_cursor : np.ndarray
        Unpacked memory pool arrays.
    max_time : float
        Maximum simulation time.
    transmission_rate : float
        Transmission rate.
    recovery_rate : float
        Recovery rate.
    use_gpu : bool, optional
        Whether to attempt execution on GPU (default: False).

    Returns
    -------
    float
        Final simulation time.
    """
    indptr, indices, edge_weights, _, node_states = compiled_graph

    time_t = 0.0
    node_count = node_states.shape[0]
    propensities = np.zeros(node_count, dtype=np.float32)
    next_event_times = np.full(node_count, np.inf, dtype=np.float64)

    # Step 1: Seed one deterministic local event clock per node so unrelated
    # subgraphs cannot perturb already scheduled events.
    for node_idx in range(node_count):
        propensity = _compute_node_propensity(
            node_idx,
            node_states,
            indptr,
            indices,
            edge_weights,
            transmission_rate,
            recovery_rate,
        )
        propensities[node_idx] = propensity
        next_event_times[node_idx] = _draw_next_event_time(
            time_t,
            node_idx,
            _event_type_for_state(int(node_states[node_idx])),
            propensity,
        )

    while time_t < max_time:
        selected_node = -1
        selected_time = np.inf

        # Step 2: Select the earliest local event without collapsing all
        # propensities into a shared global RNG stream.
        for node_idx in range(node_count):
            candidate_time = next_event_times[node_idx]
            if candidate_time < selected_time:
                selected_time = candidate_time
                selected_node = node_idx

        if selected_node == -1 or not np.isfinite(selected_time):
            break
        if selected_time > max_time:
            break

        time_t = float(selected_time)
        current_state = int(node_states[selected_node])
        new_state = 1 if current_state == 0 else 2
        node_states[selected_node] = np.uint8(new_state)

        # Step 3: Record the executed event before any dependent clocks are
        # rescheduled from the new local state.
        cursor = pool_cursor[0]
        if cursor < len(pool_times):
            pool_times[cursor] = time_t
            pool_nodes[cursor] = selected_node
            pool_events[cursor] = new_state
            pool_cursor[0] += 1

        # Step 4: Only refresh the event clocks whose local propensity can
        # depend on the mutated node. Disconnected components remain untouched.
        for node_idx in range(node_count):
            if node_idx != selected_node and not _node_depends_on(
                node_idx,
                selected_node,
                indptr,
                indices,
            ):
                continue

            propensity = _compute_node_propensity(
                node_idx,
                node_states,
                indptr,
                indices,
                edge_weights,
                transmission_rate,
                recovery_rate,
            )
            propensities[node_idx] = propensity
            next_event_times[node_idx] = _draw_next_event_time(
                time_t,
                node_idx,
                _event_type_for_state(int(node_states[node_idx])),
                propensity,
            )

    return time_t


def run_simulation(
    compiled_graph: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],  # Assuming we unpack the dataclass to raw arrays
    pool_times: np.ndarray,
    pool_nodes: np.ndarray,
    pool_events: np.ndarray,
    pool_cursor: np.ndarray,
    max_time: float,
    transmission_rate: float,
    recovery_rate: float,
    use_gpu: bool = False,
) -> float:
    """Runs the continuous-time Gillespie event loop.

    Parameters
    ----------
    compiled_graph : tuple
        Unpacked CompiledGraph arrays.
    pool_times, pool_nodes, pool_events, pool_cursor : np.ndarray
        Unpacked memory pool arrays.
    max_time : float
        Maximum simulation time.
    transmission_rate : float
        Transmission rate.
    recovery_rate : float
        Recovery rate.
    use_gpu : bool, optional
        Whether to attempt execution on GPU (default: False).

    Returns
    -------
    float
        Final simulation time.
    """
    if use_gpu:
        warnings.warn("GPU requested but not fully implemented")
        return _run_simulation_gpu()
    return float(
        _run_simulation_cpu(
            compiled_graph,
            pool_times,
            pool_nodes,
            pool_events,
            pool_cursor,
            max_time,
            transmission_rate,
            recovery_rate,
        )
    )


# ===========================================================================
# Generalised simulation engine (Phase 4)
# ===========================================================================


@nb.njit(inline="always")  # type: ignore
def _compute_node_propensity_general(
    node_idx: int,
    node_states: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_weights: np.ndarray,
    spontaneous_transitions: np.ndarray,
    n_spontaneous: int,
    induced_transitions: np.ndarray,
    n_induced: int,
) -> float:
    """Computes a single node's total event propensity from transition tables.

    Parameters
    ----------
    node_idx : int
        Index of the node whose propensity is being evaluated.
    node_states : np.ndarray
        Current compartment state of every node (uint8, shape N).
    indptr : np.ndarray
        CSR row pointer array (int64, shape N+1).
    indices : np.ndarray
        CSR column index array (int64, shape E).
    edge_weights : np.ndarray
        Per-edge transmission rate multipliers (float32, shape E).
    spontaneous_transitions : np.ndarray
        Float32 array of shape (max(n_spont, 1), 3).
        Columns: [from_state_id, to_state_id, rate].
    n_spontaneous : int
        Number of valid rows in ``spontaneous_transitions``.
    induced_transitions : np.ndarray
        Float32 array of shape (max(n_ind, 1), 4).
        Columns: [source_state_id, target_state_id, catalyst_state_id, rate].
    n_induced : int
        Number of valid rows in ``induced_transitions``.

    Returns
    -------
    float
        Sum of all applicable transition rates for the given node.

    Examples
    --------
    >>> # (Numba-compiled; call from another @njit context)
    >>> prop = _compute_node_propensity_general(0, node_states, ...)
    """
    state = int(node_states[node_idx])
    propensity = 0.0

    # Step 1: Accumulate spontaneous (node-internal) transition rates.
    for i in range(n_spontaneous):
        if int(spontaneous_transitions[i, 0]) == state:
            propensity += float(spontaneous_transitions[i, 2])

    # Step 2: Accumulate induced (neighbour-driven) transition rates by
    # scanning the CSR adjacency slice for matching catalyst states.
    for i in range(n_induced):
        if int(induced_transitions[i, 0]) == state:
            catalyst = int(induced_transitions[i, 2])
            rate = float(induced_transitions[i, 3])
            for edge_idx in range(indptr[node_idx], indptr[node_idx + 1]):
                if int(node_states[int(indices[edge_idx])]) == catalyst:
                    propensity += rate * float(edge_weights[edge_idx])

    return propensity


@nb.njit(inline="always")  # type: ignore
def _determine_new_state_general(
    node_idx: int,
    time_t: float,
    node_states: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_weights: np.ndarray,
    spontaneous_transitions: np.ndarray,
    n_spontaneous: int,
    induced_transitions: np.ndarray,
    n_induced: int,
    total_propensity: float,
) -> int:
    """Samples which transition fires when a node's event clock triggers.

    Uses a second, distinct RNG context (``state + 256``) to select the
    specific outgoing transition proportionally to its rate.  This keeps the
    *clock draw* and the *selection draw* fully independent RNG streams so
    that adding new transition types does not perturb existing event clocks.

    Parameters
    ----------
    node_idx : int
        Index of the firing node.
    time_t : float
        Current simulation time (used to seed the selection RNG).
    node_states : np.ndarray
        Current node state array (uint8, shape N).
    indptr, indices, edge_weights : np.ndarray
        CSR graph arrays.
    spontaneous_transitions : np.ndarray
        Spontaneous transition table (float32, shape (max(n_spont,1), 3)).
    n_spontaneous : int
        Active row count in ``spontaneous_transitions``.
    induced_transitions : np.ndarray
        Induced transition table (float32, shape (max(n_ind,1), 4)).
    n_induced : int
        Active row count in ``induced_transitions``.
    total_propensity : float
        Pre-computed total propensity for this node (avoids recomputation).

    Returns
    -------
    int
        The new compartment state ID after the transition fires.

    Examples
    --------
    >>> # (Numba-compiled; call from another @njit context)
    >>> new_state = _determine_new_state_general(0, t, node_states, ...)
    """
    state = int(node_states[node_idx])

    # Step 1: Draw a uniform [0, total_propensity) variate using a context
    # offset of +256 so it is strictly orthogonal to the event-clock stream.
    r = float(get_random_float(time_t, node_idx, state + 256)) * total_propensity
    cumulative = 0.0
    fallback_state = state  # returned if floating-point rounding under-shoots

    # Step 2: Walk spontaneous transitions first (they are cheaper to evaluate).
    for i in range(n_spontaneous):
        if int(spontaneous_transitions[i, 0]) == state:
            rate = float(spontaneous_transitions[i, 2])
            to_state = int(spontaneous_transitions[i, 1])
            cumulative += rate
            fallback_state = to_state
            if cumulative >= r:
                return to_state

    # Step 3: Walk induced transitions; each entry contributes the summed
    # weight of all matching catalyst neighbours.
    for i in range(n_induced):
        if int(induced_transitions[i, 0]) == state:
            catalyst = int(induced_transitions[i, 2])
            rate = float(induced_transitions[i, 3])
            to_state = int(induced_transitions[i, 1])
            transition_rate = 0.0
            for edge_idx in range(indptr[node_idx], indptr[node_idx + 1]):
                if int(node_states[int(indices[edge_idx])]) == catalyst:
                    transition_rate += rate * float(edge_weights[edge_idx])
            cumulative += transition_rate
            fallback_state = to_state
            if cumulative >= r:
                return to_state

    return fallback_state


@nb.njit()  # type: ignore
def _run_simulation_general_cpu(
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_weights: np.ndarray,
    node_states: np.ndarray,
    pool_times: np.ndarray,
    pool_nodes: np.ndarray,
    pool_events: np.ndarray,
    pool_cursor: np.ndarray,
    max_time: float,
    spontaneous_transitions: np.ndarray,
    n_spontaneous: int,
    induced_transitions: np.ndarray,
    n_induced: int,
) -> float:
    """Generalised continuous-time Gillespie event loop (CPU, pure Numba).

    Implements the Next Reaction variant of the Direct Method using
    per-node local event clocks keyed to the hash-based PRNG.  Disconnected
    subgraphs are therefore RNG-isolated by construction: only nodes whose
    propensity can change after an event (i.e., they are neighbours of the
    firing node) have their clocks rescheduled.

    Parameters
    ----------
    indptr : np.ndarray
        CSR row pointer array (int64, shape N+1).
    indices : np.ndarray
        CSR column index array (int64, shape E).
    edge_weights : np.ndarray
        Per-edge weight array (float32, shape E).
    node_states : np.ndarray
        Mutable node state array (uint8, shape N). Modified in-place.
    pool_times : np.ndarray
        Pre-allocated event time buffer (float32, shape max_events).
    pool_nodes : np.ndarray
        Pre-allocated event node buffer (int64, shape max_events).
    pool_events : np.ndarray
        Pre-allocated new-state buffer (uint8, shape max_events).
    pool_cursor : np.ndarray
        Single-element int64 array acting as the pool write cursor.
    max_time : float
        Simulation terminates once the global clock exceeds this value.
    spontaneous_transitions : np.ndarray
        Float32 table of shape (max(n_spont, 1), 3).
    n_spontaneous : int
        Number of valid spontaneous transitions.
    induced_transitions : np.ndarray
        Float32 table of shape (max(n_ind, 1), 4).
    n_induced : int
        Number of valid induced transitions.

    Returns
    -------
    float
        Final simulation time when the loop terminates.

    Examples
    --------
    >>> final_t = _run_simulation_general_cpu(
    ...     indptr, indices, edge_weights, node_states,
    ...     pool_times, pool_nodes, pool_events, pool_cursor,
    ...     100.0, spont_table, 1, ind_table, 1,
    ... )
    """
    node_count = node_states.shape[0]

    # Step 1: Allocate per-node data structures (zero-allocation hot loop).
    next_event_times = np.full(node_count, np.inf, dtype=np.float64)
    propensities = np.zeros(node_count, dtype=np.float32)

    # Step 2: Seed initial event clocks for every node.
    for node_idx in range(node_count):
        prop = _compute_node_propensity_general(
            node_idx,
            node_states,
            indptr,
            indices,
            edge_weights,
            spontaneous_transitions,
            n_spontaneous,
            induced_transitions,
            n_induced,
        )
        propensities[node_idx] = prop
        state = int(node_states[node_idx])
        # Use (state + 1) as event_type so state 0 does not evaluate to the
        # sentinel value 0 that _draw_next_event_time treats as "no event".
        next_event_times[node_idx] = _draw_next_event_time(
            0.0, node_idx, state + 1, prop
        )

    time_t = 0.0

    while time_t < max_time:
        # Step 3: Identify the globally earliest local event (O(N) scan).
        selected_node = -1
        selected_time = np.inf
        for i in range(node_count):
            if next_event_times[i] < selected_time:
                selected_time = next_event_times[i]
                selected_node = i

        if selected_node == -1 or not np.isfinite(selected_time):
            break  # No pending events — simulation is quiescent.
        if selected_time > max_time:
            break  # Next event is beyond the simulation horizon.

        time_t = float(selected_time)

        # Step 4: Determine which specific transition fires for the selected
        # node, weighted by individual transition rates.
        prop = float(propensities[selected_node])
        new_state = _determine_new_state_general(
            selected_node,
            time_t,
            node_states,
            indptr,
            indices,
            edge_weights,
            spontaneous_transitions,
            n_spontaneous,
            induced_transitions,
            n_induced,
            prop,
        )
        node_states[selected_node] = np.uint8(new_state)

        # Step 5: Record the event in the pre-allocated memory pool.
        cursor = pool_cursor[0]
        if cursor < len(pool_times):
            pool_times[cursor] = time_t
            pool_nodes[cursor] = selected_node
            pool_events[cursor] = new_state
            pool_cursor[0] += 1

        # Step 6: Reschedule only the nodes whose propensity can change.
        # A node Y is affected iff: Y == selected_node OR selected_node is
        # in Y's adjacency list (i.e., Y has selected_node as a neighbour).
        for node_idx in range(node_count):
            if node_idx != selected_node and not _node_depends_on(
                node_idx,
                selected_node,
                indptr,
                indices,
            ):
                continue  # This node is unaffected — skip rescheduling.

            new_prop = _compute_node_propensity_general(
                node_idx,
                node_states,
                indptr,
                indices,
                edge_weights,
                spontaneous_transitions,
                n_spontaneous,
                induced_transitions,
                n_induced,
            )
            propensities[node_idx] = new_prop
            state = int(node_states[node_idx])
            next_event_times[node_idx] = _draw_next_event_time(
                time_t, node_idx, state + 1, new_prop
            )

    return time_t


def run_simulation_general(
    compiled_graph: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
    pool_times: np.ndarray,
    pool_nodes: np.ndarray,
    pool_events: np.ndarray,
    pool_cursor: np.ndarray,
    max_time: float,
    spontaneous_transitions: np.ndarray,
    n_spontaneous: int,
    induced_transitions: np.ndarray,
    n_induced: int,
) -> float:
    """Python entry-point for the generalised Gillespie simulation engine.

    Unpacks the ``CompiledGraph`` tuple and delegates to the Numba-compiled
    ``_run_simulation_general_cpu`` kernel.  This thin wrapper is the sole
    Python↔Numba boundary in the generalised engine path.

    Parameters
    ----------
    compiled_graph : tuple[np.ndarray, ...]
        Five-element tuple produced by unpacking a ``CompiledGraph``:
        ``(indptr, indices, edge_weights, node_weights, node_states)``.
    pool_times : np.ndarray
        Pre-allocated event time buffer (float32, shape max_events).
    pool_nodes : np.ndarray
        Pre-allocated event node buffer (int64, shape max_events).
    pool_events : np.ndarray
        Pre-allocated new-state buffer (uint8, shape max_events).
    pool_cursor : np.ndarray
        Single-element int64 cursor array for the pool.
    max_time : float
        Simulation horizon (time units).
    spontaneous_transitions : np.ndarray
        Float32 transition table, shape (max(n_spont, 1), 3).
        Columns: [from_state_id, to_state_id, rate].
    n_spontaneous : int
        Number of valid spontaneous transition rows.
    induced_transitions : np.ndarray
        Float32 transition table, shape (max(n_ind, 1), 4).
        Columns: [source_id, target_id, catalyst_id, rate].
    n_induced : int
        Number of valid induced transition rows.

    Returns
    -------
    float
        Final simulation clock value.

    Raises
    ------
    ValueError
        If ``max_time`` is not positive.

    Examples
    --------
    >>> from EpiNext.core.compiler import compile_graph
    >>> from EpiNext.core.memory import allocate_pool
    >>> import networkx as nx, numpy as np
    >>> G = nx.complete_graph(10)
    >>> cg = compile_graph(G)
    >>> cg.node_states[0] = 1  # seed one infected node
    >>> pool = allocate_pool(500)
    >>> unpacked = (cg.indptr, cg.indices, cg.edge_weights,
    ...             cg.node_weights, cg.node_states)
    >>> spont = np.array([[1.0, 2.0, 0.1]], dtype=np.float32)  # I→R
    >>> ind   = np.array([[0.0, 1.0, 1.0, 0.3]], dtype=np.float32)  # S→I|I
    >>> final_t = run_simulation_general(
    ...     unpacked, pool.times, pool.nodes, pool.events,
    ...     pool.cursor, 50.0, spont, 1, ind, 1,
    ... )
    """
    if max_time <= 0.0:
        raise ValueError(f"max_time must be positive; got {max_time!r}.")

    indptr, indices, edge_weights, _, node_states = compiled_graph

    return float(
        _run_simulation_general_cpu(
            indptr,
            indices,
            edge_weights,
            node_states,
            pool_times,
            pool_nodes,
            pool_events,
            pool_cursor,
            max_time,
            spontaneous_transitions,
            n_spontaneous,
            induced_transitions,
            n_induced,
        )
    )

