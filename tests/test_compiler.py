from typing import Any

import networkx as nx
import numpy as np

from EpiNext.core.compiler import CompiledGraph, compile_graph


def test_compile_directed_graph_csr_structure() -> None:
    # A -> B, B -> C, A -> C
    G: nx.DiGraph[Any] = nx.DiGraph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(0, 2)

    compiled = compile_graph(G)

    # Expected indptr for nodes 0, 1, 2
    # node 0 out edges: 2
    # node 1 out edges: 1
    # node 2 out edges: 0
    # indptr = [0, 2, 3, 3]

    assert isinstance(compiled, CompiledGraph)
    assert np.array_equal(compiled.indptr, np.array([0, 2, 3, 3]))

    # Expected indices for nodes 0, 1, 2
    # node 0 connects to 1 and 2
    # node 1 connects to 2
    # indices = [1, 2, 2]
    # order of connections may vary depending on Graph structure internally,
    # so we sort per node
    assert set(compiled.indices[compiled.indptr[0] : compiled.indptr[1]]) == {1, 2}
    assert set(compiled.indices[compiled.indptr[1] : compiled.indptr[2]]) == {2}


def test_compile_undirected_graph_csr_structure() -> None:
    # A - B, B - C, A - C
    G: nx.Graph[Any] = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(0, 2)

    compiled = compile_graph(G)

    # Undirected translates to bidirectional edges
    # node 0 out edges: 2 (to 1, 2)
    # node 1 out edges: 2 (to 0, 2)
    # node 2 out edges: 2 (to 0, 1)
    # indptr = [0, 2, 4, 6]
    assert np.array_equal(compiled.indptr, np.array([0, 2, 4, 6]))
    assert set(compiled.indices[compiled.indptr[0] : compiled.indptr[1]]) == {1, 2}
    assert set(compiled.indices[compiled.indptr[1] : compiled.indptr[2]]) == {0, 2}
    assert set(compiled.indices[compiled.indptr[2] : compiled.indptr[3]]) == {0, 1}


def test_compile_graph_edge_weights_float32() -> None:
    G: nx.DiGraph[Any] = nx.DiGraph()
    G.add_edge(0, 1, weight=1.5)
    G.add_edge(1, 2, weight=2.5)

    compiled = compile_graph(G)

    assert compiled.edge_weights.dtype == np.float32
    assert len(compiled.edge_weights) == 2
    # We check if 1.5 and 2.5 exist in edge weights (ignoring strict order
    # if we don't fix it)
    assert 1.5 in compiled.edge_weights
    assert 2.5 in compiled.edge_weights


def test_compile_graph_node_weights_float32() -> None:
    G: nx.DiGraph[Any] = nx.DiGraph()
    G.add_node(0, weight=1.1)
    G.add_node(1, weight=2.2)
    G.add_node(2)  # default weight should be 1.0
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    compiled = compile_graph(G)

    assert compiled.node_weights.dtype == np.float32
    assert len(compiled.node_weights) == 3
    assert np.isclose(compiled.node_weights[0], 1.1)
    assert np.isclose(compiled.node_weights[1], 2.2)
    assert np.isclose(compiled.node_weights[2], 1.0)


def test_compile_graph_node_states_uint8() -> None:
    G: nx.DiGraph[Any] = nx.DiGraph()
    G.add_node(0)
    G.add_node(1)

    compiled = compile_graph(G)

    assert compiled.node_states.dtype == np.uint8
    assert len(compiled.node_states) == 2
    assert np.array_equal(compiled.node_states, np.zeros(2, dtype=np.uint8))


def test_compile_graph_caching() -> None:
    G: nx.DiGraph[Any] = nx.DiGraph()
    G.add_edge(0, 1)

    # First compilation
    compiled_1 = compile_graph(G)

    # Second compilation with the exact same graph object should return the
    # exact same CompiledGraph instance
    compiled_2 = compile_graph(G)

    assert compiled_1 is compiled_2

    # Different graph should result in a different CompiledGraph instance
    G_new: nx.DiGraph[Any] = nx.DiGraph()
    G_new.add_edge(0, 1)
    compiled_3 = compile_graph(G_new)

    assert compiled_1 is not compiled_3
