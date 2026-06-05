from typing import Any

import networkx as nx  # type: ignore

from EpiNext.core.compiler import compile_graph
from EpiNext.core.memory import allocate_pool, extract_history
from EpiNext.core.simulator import run_simulation


def test_simulator_basic_run() -> None:
    # Simple SI graph
    G = nx.Graph()
    G.add_edge(0, 1)

    compiled = compile_graph(G)
    # Set node 0 to Infected (1)
    compiled.node_states[0] = 1

    pool = allocate_pool(10)

    # We unpack for the numba func
    unpacked_graph = (
        compiled.indptr,
        compiled.indices,
        compiled.edge_weights,
        compiled.node_weights,
        compiled.node_states,
    )

    final_time = run_simulation(
        unpacked_graph,
        pool.times,
        pool.nodes,
        pool.events,
        pool.cursor,
        1.0,  # max_time
        0.5,  # transmission_rate
        0.0,  # recovery_rate
    )

    assert final_time > 0


def _run_timeline(
    graph: nx.Graph[Any],
    infected_nodes: list[int],
    *,
    max_time: float,
    transmission_rate: float,
    recovery_rate: float,
) -> list[tuple[float, int, int]]:
    compiled = compile_graph(graph)
    compiled.node_states[:] = 0

    for node in infected_nodes:
        compiled.node_states[node] = 1

    pool = allocate_pool(32)
    unpacked_graph = (
        compiled.indptr,
        compiled.indices,
        compiled.edge_weights,
        compiled.node_weights,
        compiled.node_states,
    )

    run_simulation(
        unpacked_graph,
        pool.times,
        pool.nodes,
        pool.events,
        pool.cursor,
        max_time,
        transmission_rate,
        recovery_rate,
    )

    times, nodes, events = extract_history(pool)
    return list(zip(times.tolist(), nodes.tolist(), events.tolist(), strict=True))


def test_disconnected_component_does_not_change_tracked_node_timeline() -> None:
    base_graph: nx.Graph[int] = nx.Graph()
    base_graph.add_edge(0, 1)

    perturbed_graph: nx.Graph[int] = nx.Graph()
    perturbed_graph.add_edge(0, 1)
    perturbed_graph.add_edge(2, 3)

    base_timeline = _run_timeline(
        base_graph,
        [0],
        max_time=10.0,
        transmission_rate=1.0,
        recovery_rate=0.5,
    )
    perturbed_timeline = _run_timeline(
        perturbed_graph,
        [0, 2],
        max_time=10.0,
        transmission_rate=1.0,
        recovery_rate=0.5,
    )

    tracked_node = 1
    base_tracked_timeline = [
        event for event in base_timeline if event[1] == tracked_node
    ]
    perturbed_tracked_timeline = [
        event for event in perturbed_timeline if event[1] == tracked_node
    ]

    assert perturbed_tracked_timeline == base_tracked_timeline
