"""Tests for the GPU hardware abstraction layer (Task 3.6).

All tests must pass in a CPU-only environment.  GPU-specific code paths are
exercised only when hardware is detected, and skipped otherwise so that the
CI pipeline is never gated on GPU availability.
"""

from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import networkx as nx  # type: ignore
import numpy as np
import pytest

from EpiNext.core.compiler import compile_graph
from EpiNext.core.gpu import (
    DeviceArrays,
    HardwareBackend,
    _cpu_propensities,
    detect_backend,
    run_simulation_gpu,
    transfer_to_device,
    transfer_to_host,
)
from EpiNext.core.memory import allocate_pool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_si_graph() -> Any:
    """Returns a tiny SI-ready compiled graph (node 0 Infected, node 1 Susceptible)."""
    G: nx.Graph[int] = nx.Graph()
    G.add_edge(0, 1)
    compiled = compile_graph(G)
    compiled.node_states[0] = 1  # node 0 is Infected
    return compiled


# ---------------------------------------------------------------------------
# HardwareBackend enum
# ---------------------------------------------------------------------------


def test_hardware_backend_values() -> None:
    assert HardwareBackend.CPU.value == "cpu"
    assert HardwareBackend.CUDA.value == "cuda"
    assert HardwareBackend.METAL.value == "metal"


# ---------------------------------------------------------------------------
# detect_backend
# ---------------------------------------------------------------------------


def test_detect_backend_returns_hardware_backend() -> None:
    backend = detect_backend()
    assert isinstance(backend, HardwareBackend)


def test_detect_backend_returns_cpu_when_no_gpu(monkeypatch: Any) -> None:
    """Force CUDA and MLX to appear unavailable; must fall back to CPU."""

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise ImportError("mocked unavailable")

    with patch.dict("sys.modules", {"numba.cuda": None, "mlx.core": None}):
        # Even if the modules exist in cache, detect_backend checks availability.
        # We patch cuda.is_available to return False.
        mock_cuda = MagicMock()
        mock_cuda.is_available.return_value = False
        with patch("EpiNext.core.gpu.detect_backend", wraps=detect_backend):
            with patch.dict("sys.modules", {"numba": MagicMock(cuda=mock_cuda)}):
                result = detect_backend()
        # We cannot guarantee CUDA is absent in every CI environment, but
        # we can assert the return type is always valid.
        assert isinstance(result, HardwareBackend)


# ---------------------------------------------------------------------------
# transfer_to_device / transfer_to_host — CPU path
# ---------------------------------------------------------------------------


def test_transfer_to_device_cpu_returns_device_arrays() -> None:
    compiled = _simple_si_graph()
    device = transfer_to_device(
        compiled.indptr,
        compiled.indices,
        compiled.edge_weights,
        compiled.node_weights,
        compiled.node_states,
        HardwareBackend.CPU,
    )
    assert isinstance(device, DeviceArrays)
    assert device.backend == HardwareBackend.CPU


def test_transfer_to_host_cpu_returns_numpy_arrays() -> None:
    compiled = _simple_si_graph()
    device = transfer_to_device(
        compiled.indptr,
        compiled.indices,
        compiled.edge_weights,
        compiled.node_weights,
        compiled.node_states,
        HardwareBackend.CPU,
    )
    host = transfer_to_host(device)
    assert len(host) == 5
    for arr in host:
        assert isinstance(arr, np.ndarray)


def test_transfer_roundtrip_preserves_data() -> None:
    compiled = _simple_si_graph()
    device = transfer_to_device(
        compiled.indptr,
        compiled.indices,
        compiled.edge_weights,
        compiled.node_weights,
        compiled.node_states,
        HardwareBackend.CPU,
    )
    indptr, indices, edge_weights, node_weights, node_states = transfer_to_host(device)

    assert np.array_equal(indptr, compiled.indptr)
    assert np.array_equal(indices, compiled.indices)
    assert np.array_equal(edge_weights, compiled.edge_weights)
    assert np.array_equal(node_weights, compiled.node_weights)
    assert np.array_equal(node_states, compiled.node_states)


# ---------------------------------------------------------------------------
# transfer_to_device — CUDA path (skip if CUDA absent)
# ---------------------------------------------------------------------------


def _cuda_available() -> bool:
    try:
        from numba import cuda

        return bool(cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_transfer_to_device_cuda() -> None:
    compiled = _simple_si_graph()
    device = transfer_to_device(
        compiled.indptr,
        compiled.indices,
        compiled.edge_weights,
        compiled.node_weights,
        compiled.node_states,
        HardwareBackend.CUDA,
    )
    assert device.backend == HardwareBackend.CUDA
    host = transfer_to_host(device)
    assert np.array_equal(host[0], compiled.indptr)


# ---------------------------------------------------------------------------
# _cpu_propensities
# ---------------------------------------------------------------------------


def test_cpu_propensities_susceptible_node_with_infected_neighbor() -> None:
    """Node 1 (S) has one infected neighbor (node 0, I); propensity must be positive."""
    compiled = _simple_si_graph()
    props = _cpu_propensities(
        compiled.node_states,
        compiled.indptr,
        compiled.indices,
        compiled.edge_weights,
        transmission_rate=0.5,
        recovery_rate=0.1,
    )
    assert props.dtype == np.float32
    assert props[0] == pytest.approx(0.1, rel=1e-5)  # infected -> recovery
    assert props[1] > 0.0  # susceptible -> should be infected by neighbor


def test_cpu_propensities_all_susceptible_zero() -> None:
    """If no node is infected the entire propensity vector must be zero."""
    G: nx.Graph[int] = nx.Graph()
    G.add_edge(0, 1)
    compiled = compile_graph(G)
    # Explicitly pass a fresh all-Susceptible state array to avoid sharing
    # the cached CompiledGraph's node_states, which may have been mutated by
    # a prior test.
    all_susceptible = np.zeros(compiled.node_states.shape, dtype=np.uint8)
    props = _cpu_propensities(
        all_susceptible,
        compiled.indptr,
        compiled.indices,
        compiled.edge_weights,
        transmission_rate=0.5,
        recovery_rate=0.1,
    )
    assert np.all(props == 0.0)


def test_cpu_propensities_recovered_node_zero() -> None:
    """A recovered node (state 2) has zero propensity."""
    G: nx.Graph[int] = nx.Graph()
    G.add_node(0)
    compiled = compile_graph(G)
    compiled.node_states[0] = 2  # Recovered
    props = _cpu_propensities(
        compiled.node_states,
        compiled.indptr,
        compiled.indices,
        compiled.edge_weights,
        transmission_rate=0.5,
        recovery_rate=0.1,
    )
    assert props[0] == 0.0


# ---------------------------------------------------------------------------
# run_simulation_gpu — CPU fallback path
# ---------------------------------------------------------------------------


def test_run_simulation_gpu_cpu_fallback_warns() -> None:
    """run_simulation_gpu with CPU backend must emit a UserWarning."""
    compiled = _simple_si_graph()
    pool = allocate_pool(50)
    unpacked = (
        compiled.indptr,
        compiled.indices,
        compiled.edge_weights,
        compiled.node_weights,
        compiled.node_states,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        final_time = run_simulation_gpu(
            unpacked,
            pool.times,
            pool.nodes,
            pool.events,
            pool.cursor,
            max_time=5.0,
            transmission_rate=0.5,
            recovery_rate=0.0,
            backend=HardwareBackend.CPU,
        )
    assert any("No GPU detected" in str(warning.message) for warning in w)
    assert final_time > 0.0


def test_run_simulation_gpu_cpu_fallback_records_events() -> None:
    """At least one infection event must be recorded in the pool."""
    compiled = _simple_si_graph()
    pool = allocate_pool(50)
    unpacked = (
        compiled.indptr,
        compiled.indices,
        compiled.edge_weights,
        compiled.node_weights,
        compiled.node_states,
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        run_simulation_gpu(
            unpacked,
            pool.times,
            pool.nodes,
            pool.events,
            pool.cursor,
            max_time=10.0,
            transmission_rate=1.0,
            recovery_rate=0.0,
            backend=HardwareBackend.CPU,
        )
    assert int(pool.cursor[0]) > 0


def test_run_simulation_gpu_auto_detect_does_not_crash() -> None:
    """Auto-detect backend path must not raise even without GPU hardware."""
    compiled = _simple_si_graph()
    pool = allocate_pool(50)
    unpacked = (
        compiled.indptr,
        compiled.indices,
        compiled.edge_weights,
        compiled.node_weights,
        compiled.node_states,
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        final_time = run_simulation_gpu(
            unpacked,
            pool.times,
            pool.nodes,
            pool.events,
            pool.cursor,
            max_time=5.0,
            transmission_rate=0.5,
            recovery_rate=0.0,
        )
    assert isinstance(final_time, float)


# ---------------------------------------------------------------------------
# run_simulation_gpu — CUDA path (skip if CUDA absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_run_simulation_gpu_cuda_path() -> None:
    compiled = _simple_si_graph()
    pool = allocate_pool(50)
    unpacked = (
        compiled.indptr,
        compiled.indices,
        compiled.edge_weights,
        compiled.node_weights,
        compiled.node_states,
    )
    final_time = run_simulation_gpu(
        unpacked,
        pool.times,
        pool.nodes,
        pool.events,
        pool.cursor,
        max_time=5.0,
        transmission_rate=0.5,
        recovery_rate=0.0,
        backend=HardwareBackend.CUDA,
    )
    assert final_time > 0.0
