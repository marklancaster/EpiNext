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
