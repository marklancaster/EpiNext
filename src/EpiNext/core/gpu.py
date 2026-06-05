"""GPU hardware abstraction layer for EpiNext.

Provides a unified interface for NVIDIA CUDA (via numba.cuda) and Apple Metal
(via MLX) acceleration, with transparent host-to-device memory transfer logic
and graceful CPU fallback when no GPU hardware is detected.
"""

from __future__ import annotations

import enum
import warnings
from typing import Any, Optional, Tuple

import numpy as np


class HardwareBackend(enum.Enum):
    """Supported hardware execution backends."""

    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"


def detect_backend() -> HardwareBackend:
    """Detects the best available hardware backend.

    Probes for NVIDIA CUDA first, then Apple Metal (MLX), and finally
    falls back to CPU.

    Returns
    -------
    HardwareBackend
        The best available backend on the current machine.
    """
    # Try CUDA first
    try:
        from numba import cuda

        if cuda.is_available():
            return HardwareBackend.CUDA
    except Exception:  # noqa: BLE001
        pass

    # Try Apple Metal / MLX
    try:
        import mlx.core as mx

        _ = mx.array([1.0])  # probe allocation
        return HardwareBackend.METAL
    except Exception:  # noqa: BLE001
        pass

    return HardwareBackend.CPU


# ---------------------------------------------------------------------------
# Memory transfer helpers
# ---------------------------------------------------------------------------


class DeviceArrays:
    """Container for arrays that have been transferred to device memory.

    Device-side arrays can be NumPy arrays (CPU), numba CUDA device arrays,
    or MLX arrays; they are typed as ``Any`` because their concrete type is
    determined at runtime.

    Attributes
    ----------
    backend : HardwareBackend
        The backend these arrays reside on.
    indptr : Any
        Device-side CSR index pointer array.
    indices : Any
        Device-side CSR column index array.
    edge_weights : Any
        Device-side edge weight array.
    node_weights : Any
        Device-side node weight array.
    node_states : Any
        Device-side node state array (mutable, modified in-place by kernel).
    """

    def __init__(
        self,
        backend: HardwareBackend,
        indptr: Any,
        indices: Any,
        edge_weights: Any,
        node_weights: Any,
        node_states: Any,
    ) -> None:
        self.backend = backend
        self.indptr: Any = indptr
        self.indices: Any = indices
        self.edge_weights: Any = edge_weights
        self.node_weights: Any = node_weights
        self.node_states: Any = node_states


def transfer_to_device(
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_weights: np.ndarray,
    node_weights: np.ndarray,
    node_states: np.ndarray,
    backend: HardwareBackend,
) -> DeviceArrays:
    """Copies host NumPy arrays to device memory.

    Parameters
    ----------
    indptr : np.ndarray
        Host CSR index pointer array.
    indices : np.ndarray
        Host CSR column index array.
    edge_weights : np.ndarray
        Host edge weight array (float32).
    node_weights : np.ndarray
        Host node weight array (float32).
    node_states : np.ndarray
        Host node state array (uint8).
    backend : HardwareBackend
        Target device backend.

    Returns
    -------
    DeviceArrays
        Container holding device-resident copies of all arrays.

    Raises
    ------
    RuntimeError
        If the requested backend is not available.
    """
    if backend == HardwareBackend.CUDA:
        try:
            from numba import cuda

            return DeviceArrays(
                backend=backend,
                indptr=cuda.to_device(indptr),
                indices=cuda.to_device(indices),
                edge_weights=cuda.to_device(edge_weights),
                node_weights=cuda.to_device(node_weights),
                node_states=cuda.to_device(node_states),
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"CUDA transfer failed: {exc}") from exc

    if backend == HardwareBackend.METAL:
        try:
            import mlx.core as mx

            return DeviceArrays(
                backend=backend,
                indptr=mx.array(indptr),
                indices=mx.array(indices),
                edge_weights=mx.array(edge_weights),
                node_weights=mx.array(node_weights),
                node_states=mx.array(node_states),
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Metal/MLX transfer failed: {exc}") from exc

    # CPU — no transfer needed; wrap as-is
    return DeviceArrays(
        backend=HardwareBackend.CPU,
        indptr=indptr,
        indices=indices,
        edge_weights=edge_weights,
        node_weights=node_weights,
        node_states=node_states,
    )


def transfer_to_host(device_arrays: DeviceArrays) -> Tuple[np.ndarray, ...]:
    """Copies device arrays back to host NumPy arrays.

    Parameters
    ----------
    device_arrays : DeviceArrays
        Container of device-resident arrays.

    Returns
    -------
    Tuple[np.ndarray, ...]
        Host copies of (indptr, indices, edge_weights, node_weights, node_states).
    """
    if device_arrays.backend == HardwareBackend.CUDA:
        # numba CUDA device arrays expose .copy_to_host(); typed as Any.
        return (
            device_arrays.indptr.copy_to_host(),
            device_arrays.indices.copy_to_host(),
            device_arrays.edge_weights.copy_to_host(),
            device_arrays.node_weights.copy_to_host(),
            device_arrays.node_states.copy_to_host(),
        )

    if device_arrays.backend == HardwareBackend.METAL:
        import mlx.core as mx

        return (
            np.array(mx.eval(device_arrays.indptr)),
            np.array(mx.eval(device_arrays.indices)),
            np.array(mx.eval(device_arrays.edge_weights)),
            np.array(mx.eval(device_arrays.node_weights)),
            np.array(mx.eval(device_arrays.node_states)),
        )

    # CPU — already NumPy arrays
    return (
        np.asarray(device_arrays.indptr),
        np.asarray(device_arrays.indices),
        np.asarray(device_arrays.edge_weights),
        np.asarray(device_arrays.node_weights),
        np.asarray(device_arrays.node_states),
    )


# ---------------------------------------------------------------------------
# CUDA kernel
# ---------------------------------------------------------------------------

# Lazy import to avoid hard dependency at module load time.
_cuda_kernel: Optional[Any] = None


def _build_cuda_kernel() -> Any:
    """Builds and caches the CUDA propensity kernel.

    Returns
    -------
    Any
        A compiled ``numba.cuda.jit`` kernel callable.

    Raises
    ------
    RuntimeError
        If CUDA is not available.
    """
    global _cuda_kernel
    if _cuda_kernel is not None:
        return _cuda_kernel

    try:
        from numba import cuda

        @cuda.jit
        def _cuda_propensities_kernel(
            node_states: np.ndarray,
            indptr: np.ndarray,
            indices: np.ndarray,
            edge_weights: np.ndarray,
            transmission_rate: float,
            recovery_rate: float,
            propensities: np.ndarray,
        ) -> None:
            """CUDA kernel: compute per-node propensity."""
            i = cuda.grid(1)
            if i >= node_states.shape[0]:
                return
            state = node_states[i]
            prop = 0.0
            if state == 1:
                prop += recovery_rate
            elif state == 0:
                start = indptr[i]
                end = indptr[i + 1]
                for k in range(start, end):
                    neighbor = indices[k]
                    if node_states[neighbor] == 1:
                        prop += transmission_rate * edge_weights[k]
            propensities[i] = prop

        _cuda_kernel = _cuda_propensities_kernel
        return _cuda_kernel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Cannot build CUDA kernel: {exc}") from exc


def run_propensities_cuda(
    node_states: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_weights: np.ndarray,
    transmission_rate: float,
    recovery_rate: float,
    threads_per_block: int = 256,
) -> np.ndarray:
    """Runs the propensity calculation on a CUDA GPU.

    Parameters
    ----------
    node_states : np.ndarray
        Node state array (host or device).
    indptr, indices, edge_weights : np.ndarray
        CSR graph arrays (host or device).
    transmission_rate : float
        Edge transmission rate.
    recovery_rate : float
        Node recovery rate.
    threads_per_block : int
        CUDA threads per block (default 256).

    Returns
    -------
    np.ndarray
        Host propensity array of shape (N,).
    """
    from numba import cuda

    kernel: Any = _build_cuda_kernel()
    N = node_states.shape[0]

    d_states = cuda.to_device(node_states)
    d_indptr = cuda.to_device(indptr)
    d_indices = cuda.to_device(indices)
    d_weights = cuda.to_device(edge_weights)
    d_prop = cuda.device_array(N, dtype=np.float32)

    blocks = (N + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block](
        d_states, d_indptr, d_indices, d_weights,
        transmission_rate, recovery_rate, d_prop,
    )
    result: np.ndarray = d_prop.copy_to_host()
    return result


# ---------------------------------------------------------------------------
# Metal / MLX kernel
# ---------------------------------------------------------------------------


def run_propensities_metal(
    node_states: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_weights: np.ndarray,
    transmission_rate: float,
    recovery_rate: float,
) -> np.ndarray:
    """Runs the propensity calculation using Apple MLX (Metal backend).

    Parameters
    ----------
    node_states : np.ndarray
        Node state array.
    indptr, indices, edge_weights : np.ndarray
        CSR graph arrays.
    transmission_rate : float
        Edge transmission rate.
    recovery_rate : float
        Node recovery rate.

    Returns
    -------
    np.ndarray
        Host propensity array of shape (N,).

    Notes
    -----
    MLX does not yet support ragged gather operations natively, so this
    implementation uses vectorised MLX operations as a best-effort GPU
    path.  For full Metal kernel control a custom C-extension would be
    required.
    """
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("MLX is not installed; Metal backend unavailable.") from exc

    N = int(node_states.shape[0])
    mx_states = mx.array(node_states.astype(np.int32))
    mx_prop = mx.zeros((N,), dtype=mx.float32)

    # Recovery contribution: nodes in state 1 get recovery_rate
    is_infected = mx_states == 1
    mx_prop = mx_prop + mx.where(is_infected, mx.array(recovery_rate), mx.array(0.0))

    # Transmission contribution: for susceptible nodes (state 0) sum
    # infected neighbour edge weights.  We expand via segment-sum over the
    # CSR indices.
    is_susceptible = mx_states == 0
    mx_edge_weights = mx.array(edge_weights)
    mx_indices = mx.array(indices)

    # Build neighbour-infected mask per edge
    neighbor_infected = mx_states[mx_indices] == 1
    weighted_contrib = mx.where(
        neighbor_infected,
        mx_edge_weights * transmission_rate,
        mx.array(0.0),
    )

    # Segment-sum: accumulate into source nodes using scatter-add
    contrib_per_node = mx.zeros((N,), dtype=mx.float32)
    contrib_per_node = contrib_per_node.at[
        mx.array(np.repeat(np.arange(N), np.diff(indptr)))
    ].add(weighted_contrib)

    mx_prop = mx_prop + mx.where(
        is_susceptible.astype(mx.float32) > 0,
        contrib_per_node,
        mx.array(0.0),
    )

    mx.eval(mx_prop)
    result: np.ndarray = np.array(mx_prop)
    return result


# ---------------------------------------------------------------------------
# Unified GPU simulation entry-point
# ---------------------------------------------------------------------------


def run_simulation_gpu(
    compiled_graph: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
    pool_times: np.ndarray,
    pool_nodes: np.ndarray,
    pool_events: np.ndarray,
    pool_cursor: np.ndarray,
    max_time: float,
    transmission_rate: float,
    recovery_rate: float,
    backend: Optional[HardwareBackend] = None,
) -> float:
    """Runs the Gillespie simulation using GPU acceleration.

    Uses the detected (or specified) hardware backend to accelerate the
    propensity calculation.  The Gillespie selection step runs on the CPU
    since it involves sequential reduction.  The most expensive inner loop
    (per-node propensity summation) is offloaded to the GPU.

    Parameters
    ----------
    compiled_graph : tuple
        Unpacked CompiledGraph arrays
        ``(indptr, indices, edge_weights, node_weights, node_states)``.
    pool_times, pool_nodes, pool_events, pool_cursor : np.ndarray
        Pre-allocated event history pool arrays.
    max_time : float
        Maximum simulation time.
    transmission_rate : float
        Edge transmission rate.
    recovery_rate : float
        Node recovery rate.
    backend : HardwareBackend, optional
        Override backend selection.  Auto-detected when ``None``.

    Returns
    -------
    float
        Final simulation time.

    Warns
    -----
    UserWarning
        If the requested GPU backend is unavailable and falls back to CPU.
    """
    from EpiNext.core.rng import get_random_float  # avoid circular at module level

    if backend is None:
        backend = detect_backend()

    if backend == HardwareBackend.CPU:
        warnings.warn(
            "No GPU detected; falling back to CPU for GPU simulation path.",
            UserWarning,
            stacklevel=2,
        )

    indptr, indices, edge_weights, _, node_states = compiled_graph

    time_t = 0.0

    while time_t < max_time:
        # -- Propensity calculation (GPU-accelerated where possible) --
        if backend == HardwareBackend.CUDA:
            try:
                propensities = run_propensities_cuda(
                    node_states, indptr, indices, edge_weights,
                    transmission_rate, recovery_rate,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"CUDA propensity failed ({exc}); using CPU fallback.",
                    UserWarning,
                    stacklevel=2,
                )
                propensities = _cpu_propensities(
                    node_states, indptr, indices, edge_weights,
                    transmission_rate, recovery_rate,
                )
        elif backend == HardwareBackend.METAL:
            try:
                propensities = run_propensities_metal(
                    node_states, indptr, indices, edge_weights,
                    transmission_rate, recovery_rate,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Metal propensity failed ({exc}); using CPU fallback.",
                    UserWarning,
                    stacklevel=2,
                )
                propensities = _cpu_propensities(
                    node_states, indptr, indices, edge_weights,
                    transmission_rate, recovery_rate,
                )
        else:
            propensities = _cpu_propensities(
                node_states, indptr, indices, edge_weights,
                transmission_rate, recovery_rate,
            )

        total_propensity = float(np.sum(propensities))
        if total_propensity <= 0.0:
            break

        # -- Gillespie step (CPU) --
        r1 = float(get_random_float(time_t, -1, -1))
        r2 = float(get_random_float(time_t, -2, -2))

        dt = -np.log(r1) / total_propensity
        new_time = time_t + dt

        target = r2 * total_propensity
        cumulative = 0.0
        selected_node = -1
        for i in range(node_states.shape[0]):
            cumulative += propensities[i]
            if cumulative >= target:
                selected_node = i
                break

        if selected_node == -1:
            break

        current_state = int(node_states[selected_node])
        new_state = 1 if current_state == 0 else 2

        time_t = new_time
        node_states[selected_node] = np.uint8(new_state)

        cursor = int(pool_cursor[0])
        if cursor < len(pool_times):
            pool_times[cursor] = time_t
            pool_nodes[cursor] = selected_node
            pool_events[cursor] = new_state
            pool_cursor[0] += 1

    return time_t


def _cpu_propensities(
    node_states: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_weights: np.ndarray,
    transmission_rate: float,
    recovery_rate: float,
) -> np.ndarray:
    """Pure-NumPy propensity calculation used as the CPU fallback.

    Parameters
    ----------
    node_states : np.ndarray
        Current node states (uint8).
    indptr, indices, edge_weights : np.ndarray
        CSR graph arrays.
    transmission_rate : float
        Edge transmission rate.
    recovery_rate : float
        Node recovery rate.

    Returns
    -------
    np.ndarray
        Propensity array of shape (N,) and dtype float32.
    """
    N = node_states.shape[0]
    propensities = np.zeros(N, dtype=np.float32)

    infected_mask = (node_states == 1).astype(np.float32)

    for i in range(N):
        state = int(node_states[i])
        if state == 1:
            propensities[i] = recovery_rate
        elif state == 0:
            start = int(indptr[i])
            end = int(indptr[i + 1])
            neighbors = indices[start:end]
            weights = edge_weights[start:end]
            propensities[i] = transmission_rate * float(
                np.dot(infected_mask[neighbors], weights)
            )

    return propensities
