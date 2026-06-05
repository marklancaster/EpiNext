"""TDD test suite for BaseEpidemicModel and SimulationResult (Phase 4, Tasks 4.1–4.4).

These tests establish the mathematical and API contracts that the production
code must satisfy.  They are written *before* the implementation per the
strict TDD mandate in AGENTS.md.

Tests cover:
- Compartment-to-integer translation (Task 4.3)
- Transition definition storage (Task 4.2)
- Internal transition table construction
- Initial condition wiring (Task 4.4)
- End-to-end ``run()`` contract (Task 4.1)
- ``SimulationResult`` time-series extraction
"""

from __future__ import annotations

from typing import Any

import networkx as nx  # type: ignore
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: concrete subclass used throughout the test suite
# ---------------------------------------------------------------------------
# NOTE: these imports are deferred to avoid import errors before production
# code exists; pytest will report a collection error if the module is absent.
from EpiNext.models.base import BaseEpidemicModel, SimulationResult


class _SIModel(BaseEpidemicModel):
    """Minimal SI model (no recovery) used as the simplest test fixture.

    Compartments
    ------------
    S (0) → I (1) via induced transmission from I neighbours.
    """

    def define_transitions(self) -> None:
        """Define the SI compartments and single induced transition."""
        # Step 1: Register compartments (S=0, I=1).
        self.add_compartments(["S", "I"])
        # Step 2: Susceptible nodes become infected when an infected neighbour
        # exists; there is no spontaneous recovery in this model.
        self.add_induced_transition(
            source="S",
            target="I",
            catalyst="I",
            rate=float(self.params["tau"]),
        )


class _SIRModel(BaseEpidemicModel):
    """Minimal SIR model used as a second-level test fixture.

    Compartments
    ------------
    S (0) → I (1) via induced transmission.
    I (1) → R (2) via spontaneous recovery.
    """

    def define_transitions(self) -> None:
        """Define SIR compartments and transitions."""
        self.add_compartments(["S", "I", "R"])
        self.add_spontaneous_transition(
            from_state="I",
            to_state="R",
            rate=float(self.params["gamma"]),
        )
        self.add_induced_transition(
            source="S",
            target="I",
            catalyst="I",
            rate=float(self.params["tau"]),
        )


# ---------------------------------------------------------------------------
# Task 4.3: Compartment-to-integer translation
# ---------------------------------------------------------------------------


def test_add_compartments_creates_forward_mapping() -> None:
    """add_compartments() must map each string to a unique uint8-compatible int."""
    G: Any = nx.path_graph(3)
    model = _SIRModel(graph=G, params={"tau": 0.3, "gamma": 0.1})

    # Verify forward mapping (string → int)
    assert model.compartment_to_id["S"] == 0
    assert model.compartment_to_id["I"] == 1
    assert model.compartment_to_id["R"] == 2


def test_add_compartments_creates_reverse_mapping() -> None:
    """add_compartments() must also populate the id_to_compartment reverse map."""
    G: Any = nx.path_graph(3)
    model = _SIRModel(graph=G, params={"tau": 0.3, "gamma": 0.1})

    assert model.id_to_compartment[0] == "S"
    assert model.id_to_compartment[1] == "I"
    assert model.id_to_compartment[2] == "R"


def test_add_compartments_ids_are_unique() -> None:
    """No two compartment names may share the same integer ID."""
    G: Any = nx.path_graph(2)
    model = _SIModel(graph=G, params={"tau": 0.5})
    ids = list(model.compartment_to_id.values())
    assert len(ids) == len(set(ids)), "Compartment IDs must be unique."


# ---------------------------------------------------------------------------
# Task 4.2: Spontaneous and induced transition storage
# ---------------------------------------------------------------------------


def test_add_spontaneous_transition_is_stored() -> None:
    """add_spontaneous_transition() must record the (from, to, rate) triple."""
    G: Any = nx.path_graph(3)
    model = _SIRModel(graph=G, params={"tau": 0.3, "gamma": 0.1})

    # _SIRModel registers I→R at rate gamma=0.1
    spontaneous_names = [
        (t.from_state, t.to_state) for t in model._spontaneous
    ]
    assert ("I", "R") in spontaneous_names


def test_add_spontaneous_transition_rate_is_correct() -> None:
    """Stored spontaneous rate must match the value passed to the method."""
    G: Any = nx.path_graph(3)
    model = _SIRModel(graph=G, params={"tau": 0.3, "gamma": 0.17})

    sir_spont = [t for t in model._spontaneous if t.from_state == "I"]
    assert len(sir_spont) == 1
    assert pytest.approx(sir_spont[0].rate, rel=1e-5) == 0.17


def test_add_induced_transition_is_stored() -> None:
    """add_induced_transition() must record the (source, target, catalyst, rate) quadruple."""
    G: Any = nx.path_graph(3)
    model = _SIRModel(graph=G, params={"tau": 0.3, "gamma": 0.1})

    induced_names = [
        (t.source, t.target, t.catalyst) for t in model._induced
    ]
    assert ("S", "I", "I") in induced_names


def test_add_induced_transition_rate_is_correct() -> None:
    """The stored induced rate must match the value passed to add_induced_transition."""
    G: Any = nx.path_graph(3)
    model = _SIRModel(graph=G, params={"tau": 0.42, "gamma": 0.1})

    sir_ind = [t for t in model._induced if t.source == "S"]
    assert len(sir_ind) == 1
    assert pytest.approx(sir_ind[0].rate, rel=1e-5) == 0.42


# ---------------------------------------------------------------------------
# Internal transition table construction
# ---------------------------------------------------------------------------


def test_build_transition_tables_shapes_sir() -> None:
    """_build_transition_tables() must return correctly shaped float32 arrays.

    SIR model has 1 spontaneous (I→R) and 1 induced (S→I) transition.
    """
    G: Any = nx.path_graph(3)
    model = _SIRModel(graph=G, params={"tau": 0.3, "gamma": 0.1})
    spont, n_spont, ind, n_ind = model._build_transition_tables()

    # Spontaneous: shape (≥1, 3) float32
    assert spont.dtype == np.float32
    assert spont.ndim == 2
    assert spont.shape[1] == 3
    assert n_spont == 1

    # Induced: shape (≥1, 4) float32
    assert ind.dtype == np.float32
    assert ind.ndim == 2
    assert ind.shape[1] == 4
    assert n_ind == 1


def test_build_transition_tables_values_sir() -> None:
    """Transition table entries must correctly encode compartment IDs and rates."""
    G: Any = nx.path_graph(3)
    model = _SIRModel(graph=G, params={"tau": 0.3, "gamma": 0.1})
    spont, n_spont, ind, n_ind = model._build_transition_tables()

    # Spontaneous row 0: I(1) → R(2) at rate 0.1
    assert int(spont[0, 0]) == 1  # from_state = I
    assert int(spont[0, 1]) == 2  # to_state = R
    assert pytest.approx(float(spont[0, 2]), rel=1e-4) == 0.1

    # Induced row 0: S(0) → I(1) catalysed by I(1) at rate 0.3
    assert int(ind[0, 0]) == 0  # source = S
    assert int(ind[0, 1]) == 1  # target = I
    assert int(ind[0, 2]) == 1  # catalyst = I
    assert pytest.approx(float(ind[0, 3]), rel=1e-4) == 0.3


def test_build_transition_tables_si_no_spontaneous() -> None:
    """SI model has no spontaneous transitions; n_spontaneous must be 0."""
    G: Any = nx.path_graph(2)
    model = _SIModel(graph=G, params={"tau": 0.5})
    _, n_spont, _, n_ind = model._build_transition_tables()

    assert n_spont == 0
    assert n_ind == 1


# ---------------------------------------------------------------------------
# Task 4.4: set_initial_conditions
# ---------------------------------------------------------------------------


def test_set_initial_conditions_stores_nodes() -> None:
    """set_initial_conditions() must store the provided node indices."""
    G: Any = nx.path_graph(5)
    model = _SIRModel(graph=G, params={"tau": 0.3, "gamma": 0.1})
    model.set_initial_conditions(initial_infecteds=[0, 2])

    assert model._initial_infecteds == [0, 2]


def test_set_initial_conditions_empty_list() -> None:
    """An empty initial_infecteds list must be accepted without error."""
    G: Any = nx.path_graph(5)
    model = _SIRModel(graph=G, params={"tau": 0.3, "gamma": 0.1})
    model.set_initial_conditions(initial_infecteds=[])

    assert model._initial_infecteds == []


# ---------------------------------------------------------------------------
# Task 4.1: BaseEpidemicModel.run() — structural contract
# ---------------------------------------------------------------------------


def test_run_returns_simulation_result() -> None:
    """run() must return a SimulationResult instance."""
    G: Any = nx.path_graph(4)
    model = _SIRModel(graph=G, params={"tau": 1.0, "gamma": 0.5})
    model.set_initial_conditions(initial_infecteds=[0])
    result = model.run(t_max=10.0)

    assert isinstance(result, SimulationResult)


def test_run_result_has_nonnegative_times() -> None:
    """All recorded event times must be non-negative."""
    G: Any = nx.path_graph(4)
    model = _SIRModel(graph=G, params={"tau": 1.0, "gamma": 0.5})
    model.set_initial_conditions(initial_infecteds=[0])
    result = model.run(t_max=10.0)

    assert np.all(result.times >= 0.0)


def test_run_result_times_are_monotone() -> None:
    """Recorded event times must be non-decreasing."""
    G: Any = nx.path_graph(6)
    model = _SIRModel(graph=G, params={"tau": 1.0, "gamma": 0.5})
    model.set_initial_conditions(initial_infecteds=[0])
    result = model.run(t_max=20.0)

    if len(result.times) > 1:
        diffs = np.diff(result.times.astype(np.float64))
        assert np.all(diffs >= 0.0), "Event times must be monotonically non-decreasing."


def test_run_result_times_within_tmax() -> None:
    """No recorded event time may exceed t_max."""
    G: Any = nx.path_graph(4)
    model = _SIRModel(graph=G, params={"tau": 1.0, "gamma": 0.5})
    model.set_initial_conditions(initial_infecteds=[0])
    t_max = 5.0
    result = model.run(t_max=t_max)

    assert np.all(result.times <= t_max + 1e-9)


# ---------------------------------------------------------------------------
# SimulationResult time-series extraction
# ---------------------------------------------------------------------------


def test_simulation_result_compartment_map_preserved() -> None:
    """SimulationResult must store the compartment map from the model."""
    G: Any = nx.path_graph(4)
    model = _SIRModel(graph=G, params={"tau": 1.0, "gamma": 0.5})
    model.set_initial_conditions(initial_infecteds=[0])
    result = model.run(t_max=10.0)

    assert result.compartment_map[0] == "S"
    assert result.compartment_map[1] == "I"
    assert result.compartment_map[2] == "R"


def test_compartment_counts_conservation() -> None:
    """S(t) + I(t) + R(t) = N must hold at every recorded time point."""
    G: Any = nx.path_graph(6)
    model = _SIRModel(graph=G, params={"tau": 1.5, "gamma": 0.5})
    model.set_initial_conditions(initial_infecteds=[0])
    result = model.run(t_max=30.0)

    t_pts, counts = result.compartment_counts()
    N = G.number_of_nodes()

    totals = counts["S"] + counts["I"] + counts["R"]
    assert np.all(totals == N), (
        f"Compartment conservation violated: totals range "
        f"[{totals.min()}, {totals.max()}], expected {N}."
    )


def test_compartment_counts_initial_state() -> None:
    """At t=0 (the first entry), exactly one node is infected."""
    G: Any = nx.path_graph(6)
    model = _SIRModel(graph=G, params={"tau": 1.5, "gamma": 0.5})
    model.set_initial_conditions(initial_infecteds=[0])
    result = model.run(t_max=30.0)

    _, counts = result.compartment_counts()
    assert counts["I"][0] == 1, "Exactly one initially infected node expected."
    assert counts["S"][0] == G.number_of_nodes() - 1


def test_si_model_infection_spreads() -> None:
    """SI model on a complete graph must infect all nodes within t_max."""
    N = 8
    G: Any = nx.complete_graph(N)
    model = _SIModel(graph=G, params={"tau": 2.0})
    model.set_initial_conditions(initial_infecteds=[0])
    result = model.run(t_max=100.0)

    # All nodes in state I at the end
    _, counts = result.compartment_counts()
    assert counts["I"][-1] == N, (
        f"Expected all {N} nodes infected at end; got {counts['I'][-1]}."
    )


def test_sir_model_no_susceptibles_remain_high_rate() -> None:
    """SIR model on complete graph with high tau must exhaust susceptibles."""
    N = 10
    G: Any = nx.complete_graph(N)
    model = _SIRModel(graph=G, params={"tau": 5.0, "gamma": 0.5})
    model.set_initial_conditions(initial_infecteds=[0])
    result = model.run(t_max=100.0)

    _, counts = result.compartment_counts()
    # With very high transmission, S should reach 0
    assert counts["S"][-1] == 0, (
        f"Expected 0 susceptibles at end, got {counts['S'][-1]}."
    )
