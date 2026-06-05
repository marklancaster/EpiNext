"""EpiNext core package.

Exports the low-level graph compilation and simulation engine components.

Public API
----------
compile_graph, CompiledGraph
    NetworkX → CSR compilation pipeline.
allocate_pool, extract_history, EventHistoryPool
    Zero-allocation event memory management.
run_simulation, run_simulation_general
    Numba-compiled Gillespie simulation entry-points.
"""

from __future__ import annotations

from EpiNext.core.compiler import CompiledGraph, compile_graph
from EpiNext.core.memory import EventHistoryPool, allocate_pool, extract_history
from EpiNext.core.simulator import run_simulation, run_simulation_general

__all__ = [
    "CompiledGraph",
    "compile_graph",
    "EventHistoryPool",
    "allocate_pool",
    "extract_history",
    "run_simulation",
    "run_simulation_general",
]

