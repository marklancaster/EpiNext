from typing import Any

from numba import njit  # type: ignore

from EpiNext.core.rng import get_random_float, hash_context


def test_hash_context_consistency() -> None:
    h1 = hash_context(0.1, 42, 1)
    h2 = hash_context(0.1, 42, 1)
    assert h1 == h2


def test_hash_context_uniqueness() -> None:
    h1 = hash_context(0.1, 42, 1)
    h2 = hash_context(0.2, 42, 1)
    h3 = hash_context(0.1, 43, 1)
    h4 = hash_context(0.1, 42, 2)

    assert len({h1, h2, h3, h4}) == 4


@njit  # type: ignore
def wrapper(time_t: float, node_id: int, event_type: int) -> Any:
    return get_random_float(time_t, node_id, event_type)


def test_rng_determinism() -> None:
    r1 = wrapper(0.1, 42, 1)
    r2 = wrapper(0.1, 42, 1)
    assert r1 == r2


def test_butterfly_effect_isolation() -> None:
    # If a disconnected node experiences an event, it should not change the rng sequence
    # of an isolated node.
    r_node1 = wrapper(0.1, 1, 1)

    # "Another node" (disconnected in simulation) has an event
    _ = wrapper(0.1, 2, 1)

    # Node 1 has its second event
    r_node1_second = wrapper(0.2, 1, 1)

    # If node 2 never had an event, node 1 should still get the exact same sequence.
    # We can test this just by rerunning the exact context query.
    assert wrapper(0.1, 1, 1) == r_node1
    assert wrapper(0.2, 1, 1) == r_node1_second
