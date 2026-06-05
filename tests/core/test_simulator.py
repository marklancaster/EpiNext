import networkx as nx  # type: ignore

from EpiNext.core.compiler import compile_graph
from EpiNext.core.memory import allocate_pool
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
