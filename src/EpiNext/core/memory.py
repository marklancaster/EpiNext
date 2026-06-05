from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class EventHistoryPool:
    """Pre-allocated memory pool for storing simulation event history.

    Designed to avoid Numba allocations and garbage collection during the
    inner simulation loops. Size should be estimated before simulation run.

    Attributes
    ----------
    times : np.ndarray
        Array of size (max_events,) storing the time of each event.
    nodes : np.ndarray
        Array of size (max_events,) storing the node ID that transitioned.
    events : np.ndarray
        Array of size (max_events,) storing the event type integer.
    cursor : np.ndarray
        A size-1 array acting as a mutable counter for the next open index.
    """

    times: np.ndarray
    nodes: np.ndarray
    events: np.ndarray
    cursor: np.ndarray


def allocate_pool(max_events: int) -> EventHistoryPool:
    """Allocates a zeroed-out EventHistoryPool.

    Parameters
    ----------
    max_events : int
        The estimated maximum number of events that will occur.

    Returns
    -------
    EventHistoryPool
        The initialized memory pool.
    """
    return EventHistoryPool(
        times=np.zeros(max_events, dtype=np.float32),
        nodes=np.zeros(max_events, dtype=np.int64),
        events=np.zeros(max_events, dtype=np.uint8),
        cursor=np.zeros(1, dtype=np.int64),
    )


def extract_history(
    pool: EventHistoryPool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts the recorded history from the pool up to the current cursor.

    Parameters
    ----------
    pool : EventHistoryPool
        The memory pool to extract from.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple of (times, nodes, events) arrays, truncated to the valid data length.
    """
    current_size = pool.cursor[0]
    return (
        pool.times[:current_size].copy(),
        pool.nodes[:current_size].copy(),
        pool.events[:current_size].copy(),
    )
