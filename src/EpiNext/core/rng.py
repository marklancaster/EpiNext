from typing import Any

import numba as nb  # type: ignore
import numpy as np


@nb.njit(inline="always")  # type: ignore
def hash_context(time_t: float, node_id: int, event_type: int) -> Any:
    """Computes a deterministic integer hash based on context.

    Parameters
    ----------
    time_t : float
        The current simulation time.
    node_id : int
        The ID of the node undergoing the event.
    event_type : int
        The type of event (an integer identifier).

    Returns
    -------
    int
        A deterministic hash.
    """
    # Simple hash combining mechanism based on FNV-1a principles for bits.
    # Convert float to its raw bit representation
    t_bits = np.float64(time_t).view(np.int64)

    # We mix these 64-bit integers.
    h = np.int64(1469598103934665603)  # FNV offset basis
    prime = np.int64(1099511628211)

    # Mix time
    h = h ^ t_bits
    h = h * prime

    # Mix node
    h = h ^ np.int64(node_id)
    h = h * prime

    # Mix event_type
    h = h ^ np.int64(event_type)
    h = h * prime

    return h


@nb.njit(inline="always")  # type: ignore
def get_random_float(time_t: float, node_id: int, event_type: int) -> Any:
    """Generates a pseudo-random float in [0.0, 1.0) deterministically.

    Uses a hash of the given spatial-temporal context.

    Parameters
    ----------
    time_t : float
        The current simulation time.
    node_id : int
        The ID of the node undergoing the event.
    event_type : int
        The type of event.

    Returns
    -------
    float
        A deterministic pseudo-random float in [0.0, 1.0).
    """
    h = hash_context(time_t, node_id, event_type)
    # We want a float in [0, 1).
    # Use the upper 53 bits for a float64 fraction.
    # Unsigned right shift to get 53 bits
    val = np.uint64(h) >> np.uint64(11)
    # 53 bits means max val is 2**53 - 1
    # Divide by 2**53
    return val * (1.0 / 9007199254740992.0)
