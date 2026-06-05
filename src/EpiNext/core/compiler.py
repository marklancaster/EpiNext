from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CompiledGraph:
    """A high-performance compiled representation of a network.

    Stores the network structure in Compressed Sparse Row (CSR) format,
    along with node states and weights for fast Numba execution.

    Attributes
    ----------
    indptr : np.ndarray
        CSR index pointer array of size N + 1.
    indices : np.ndarray
        CSR column indices array of size E.
    edge_weights : np.ndarray
        Weights associated with each edge, of size E.
    node_weights : np.ndarray
        Weights associated with each node, of size N.
    node_states : np.ndarray
        The state of each node, initialized to 0, of size N.
    """

    indptr: np.ndarray
    indices: np.ndarray
    edge_weights: np.ndarray
    node_weights: np.ndarray
    node_states: np.ndarray


# A simple sequential cache mechanism: bypass if the exact same networkx graph object
# is passed sequentially
_last_compiled_graph_id = None
_last_compiled_graph = None


def compile_graph(G: Any) -> CompiledGraph:
    """Compiles a NetworkX graph into a CSR array structure.

    Uses sequential caching to avoid re-compiling the exact same graph.

    Parameters
    ----------
    G : Any
        A NetworkX graph object (Graph or DiGraph).

    Returns
    -------
    CompiledGraph
        The compiled CSR representation of the graph.
    """
    global _last_compiled_graph_id, _last_compiled_graph

    current_graph_id = id(G)

    # If the exact same graph object is passed, return the cached version
    if current_graph_id == _last_compiled_graph_id and _last_compiled_graph is not None:
        return _last_compiled_graph

    # 1. Number of nodes
    N = G.number_of_nodes()

    # Extract node mapping
    # Sort nodes to ensure consistent ordering if integer based.
    # NetworkX nodes can be anything, so we'll just take them as they come or sorted.
    node_list = list(G.nodes())
    try:
        node_list.sort()
    except TypeError:
        pass

    node_to_idx = {node: i for i, node in enumerate(node_list)}

    is_directed = G.is_directed()

    # 2. Extract nodes weights
    node_weights = np.zeros(N, dtype=np.float32)
    for i, node in enumerate(node_list):
        # Default weight is 1.0 if not specified
        node_weights[i] = float(G.nodes[node].get("weight", 1.0))

    # node_states initialized to 0
    node_states = np.zeros(N, dtype=np.uint8)

    # 3. Calculate number of edges
    E = G.number_of_edges()

    # For undirected graphs, each edge appears twice in the CSR representation
    # (once for each direction)
    if not is_directed:
        E_csr = 2 * E
    else:
        E_csr = E

    # Allocate CSR arrays
    indptr = np.zeros(N + 1, dtype=np.int64)
    indices = np.zeros(E_csr, dtype=np.int64)
    edge_weights = np.zeros(E_csr, dtype=np.float32)

    # Populate CSR arrays
    current_edge_idx = 0
    for u_idx, u in enumerate(node_list):
        indptr[u_idx] = current_edge_idx

        # Get neighbors
        # For directed graphs, we only care about out-edges (successors)
        # For undirected graphs, G.neighbors gets all connected nodes
        if is_directed:
            neighbors = list(getattr(G, "successors")(u))
        else:
            neighbors = list(G.neighbors(u))

        for v in neighbors:
            v_idx = node_to_idx[v]
            indices[current_edge_idx] = v_idx

            # Get edge weight (from u to v)
            if is_directed:
                weight = G.edges[u, v].get("weight", 1.0)
            else:
                weight = G.edges[u, v].get("weight", 1.0)

            edge_weights[current_edge_idx] = float(weight)
            current_edge_idx += 1

    # Final indptr entry
    indptr[N] = current_edge_idx

    compiled_graph = CompiledGraph(
        indptr=indptr,
        indices=indices,
        edge_weights=edge_weights,
        node_weights=node_weights,
        node_states=node_states,
    )

    # Update cache
    _last_compiled_graph_id = current_graph_id
    _last_compiled_graph = compiled_graph

    return compiled_graph
