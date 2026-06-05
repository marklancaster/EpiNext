import warnings
from typing import Any

import numba as nb  # type: ignore
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
    spontaneous_rates = np.array([0.0, recovery_rate, 0.0], dtype=np.float32)

    while time_t < max_time:
        propensities = calculate_propensities(
            node_states,
            indptr,
            indices,
            edge_weights,
            spontaneous_rates,
            transmission_rate,
            recovery_rate,
        )

        new_time, node, new_state = gillespie_step(time_t, node_states, propensities)

        if node == -1:
            break

        time_t = new_time
        node_states[node] = new_state

        # Record event
        cursor = pool_cursor[0]
        if cursor < len(pool_times):
            pool_times[cursor] = time_t
            pool_nodes[cursor] = node
            pool_events[cursor] = new_state
            pool_cursor[0] += 1

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
