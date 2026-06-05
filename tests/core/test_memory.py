from EpiNext.core.memory import allocate_pool, extract_history


def test_allocate_pool() -> None:
    pool = allocate_pool(10)
    assert pool.times.shape == (10,)
    assert pool.nodes.shape == (10,)
    assert pool.events.shape == (10,)
    assert pool.cursor[0] == 0


def test_extract_history() -> None:
    pool = allocate_pool(10)
    pool.times[0] = 0.1
    pool.nodes[0] = 5
    pool.events[0] = 1
    pool.cursor[0] = 1

    times, nodes, events = extract_history(pool)
    assert len(times) == 1
    assert times[0] == 0.1
    assert nodes[0] == 5
    assert events[0] == 1
