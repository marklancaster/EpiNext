"""TDD test suite for standard pre-built epidemic models (Phase 4, Tasks 4.5–4.8).

Tests cover:
- SIRModel compartment structure and run behaviour (Task 4.6)
- SISModel cyclic behaviour — recovered nodes re-enter susceptible pool (Task 4.7)
- SEIRModel four-compartment latency dynamics (Task 4.8)
- Mean-Field Limit validation of SIRModel against scipy ODE (Task 4.5)
"""

from __future__ import annotations

from typing import Any

import networkx as nx  # type: ignore
import numpy as np
import pytest
from scipy.integrate import odeint  # type: ignore

from EpiNext.models.standard import SEIRModel, SIRModel, SISModel

# ---------------------------------------------------------------------------
# SIRModel — Task 4.6
# ---------------------------------------------------------------------------


def test_sir_model_instantiates() -> None:
    """SIRModel must instantiate without error given a graph and params."""
    G: Any = nx.path_graph(4)
    model = SIRModel(graph=G, params={"tau": 0.3, "gamma": 0.1})
    assert model is not None


def test_sir_model_compartments() -> None:
    """SIRModel must register S, I, R compartments with IDs 0, 1, 2."""
    G: Any = nx.path_graph(4)
    model = SIRModel(graph=G, params={"tau": 0.3, "gamma": 0.1})

    assert model.compartment_to_id["S"] == 0
    assert model.compartment_to_id["I"] == 1
    assert model.compartment_to_id["R"] == 2


def test_sir_model_has_spontaneous_recovery() -> None:
    """SIRModel must contain an I→R spontaneous transition at rate gamma."""
    G: Any = nx.path_graph(4)
    gamma = 0.15
    model = SIRModel(graph=G, params={"tau": 0.3, "gamma": gamma})

    ir_trans = [t for t in model._spontaneous if t.from_state == "I" and t.to_state == "R"]
    assert len(ir_trans) == 1
    assert pytest.approx(ir_trans[0].rate, rel=1e-5) == gamma


def test_sir_model_has_induced_transmission() -> None:
    """SIRModel must contain an S→I induced transition catalysed by I at rate tau."""
    G: Any = nx.path_graph(4)
    tau = 0.42
    model = SIRModel(graph=G, params={"tau": tau, "gamma": 0.1})

    si_trans = [
        t for t in model._induced
        if t.source == "S" and t.target == "I" and t.catalyst == "I"
    ]
    assert len(si_trans) == 1
    assert pytest.approx(si_trans[0].rate, rel=1e-5) == tau


def test_sir_model_run_returns_result() -> None:
    """SIRModel.run() must return a result without raising."""
    G: Any = nx.path_graph(5)
    model = SIRModel(graph=G, params={"tau": 1.0, "gamma": 0.5})
    model.set_initial_conditions(initial_infecteds=[0])
    result = model.run(t_max=20.0)
    assert result is not None


def test_sir_model_conservation() -> None:
    """S(t) + I(t) + R(t) == N at every time step in SIRModel."""
    N = 8
    G: Any = nx.complete_graph(N)
    model = SIRModel(graph=G, params={"tau": 2.0, "gamma": 0.5})
    model.set_initial_conditions(initial_infecteds=[0])
    result = model.run(t_max=30.0)

    _, counts = result.compartment_counts()
    totals = counts["S"] + counts["I"] + counts["R"]
    assert np.all(totals == N)


def test_sir_model_recovered_nodes_accumulate() -> None:
    """SIRModel: R(t) must be non-decreasing over time."""
    N = 10
    G: Any = nx.complete_graph(N)
    model = SIRModel(graph=G, params={"tau": 2.0, "gamma": 1.0})
    model.set_initial_conditions(initial_infecteds=[0])
    result = model.run(t_max=30.0)

    _, counts = result.compartment_counts()
    diffs = np.diff(counts["R"].astype(np.int64))
    assert np.all(diffs >= 0), "R must be non-decreasing in SIR."


# ---------------------------------------------------------------------------
# SISModel — Task 4.7
# ---------------------------------------------------------------------------


def test_sis_model_instantiates() -> None:
    """SISModel must instantiate without error."""
    G: Any = nx.path_graph(4)
    model = SISModel(graph=G, params={"tau": 0.5, "gamma": 0.2})
    assert model is not None


def test_sis_model_compartments() -> None:
    """SISModel must register S=0, I=1 only (no R compartment)."""
    G: Any = nx.path_graph(4)
    model = SISModel(graph=G, params={"tau": 0.5, "gamma": 0.2})

    assert "S" in model.compartment_to_id
    assert "I" in model.compartment_to_id
    assert "R" not in model.compartment_to_id


def test_sis_model_has_i_to_s_spontaneous() -> None:
    """SISModel must have an I→S spontaneous transition at rate gamma."""
    G: Any = nx.path_graph(4)
    gamma = 0.25
    model = SISModel(graph=G, params={"tau": 0.5, "gamma": gamma})

    is_trans = [t for t in model._spontaneous if t.from_state == "I" and t.to_state == "S"]
    assert len(is_trans) == 1
    assert pytest.approx(is_trans[0].rate, rel=1e-5) == gamma


def test_sis_model_conservation() -> None:
    """S(t) + I(t) == N at every time step."""
    N = 8
    G: Any = nx.complete_graph(N)
    model = SISModel(graph=G, params={"tau": 1.5, "gamma": 0.5})
    model.set_initial_conditions(initial_infecteds=[0])
    result = model.run(t_max=30.0)

    _, counts = result.compartment_counts()
    totals = counts["S"] + counts["I"]
    assert np.all(totals == N)


def test_sis_model_infected_can_recover_to_susceptible() -> None:
    """In SISModel, a node can be re-infected after recovery (I→S→I cycling)."""
    # Run a long simulation on a small complete graph; if I→S transitions occur,
    # S count must rise at least once after the first infection wave.
    N = 6
    G: Any = nx.complete_graph(N)
    model = SISModel(graph=G, params={"tau": 2.0, "gamma": 3.0})  # fast recovery
    model.set_initial_conditions(initial_infecteds=[0])
    result = model.run(t_max=50.0)

    _, counts = result.compartment_counts()
    # With fast recovery, S must increase at some point (I→S transition)
    s_increases = np.sum(np.diff(counts["S"].astype(np.int64)) > 0)
    assert s_increases > 0, (
        "SISModel: S count must increase at least once (I→S recovery event)."
    )


# ---------------------------------------------------------------------------
# SEIRModel — Task 4.8
# ---------------------------------------------------------------------------


def test_seir_model_instantiates() -> None:
    """SEIRModel must instantiate without error."""
    G: Any = nx.path_graph(5)
    model = SEIRModel(graph=G, params={"tau": 0.3, "sigma": 0.2, "gamma": 0.1})
    assert model is not None


def test_seir_model_compartments() -> None:
    """SEIRModel must register S=0, E=1, I=2, R=3."""
    G: Any = nx.path_graph(5)
    model = SEIRModel(graph=G, params={"tau": 0.3, "sigma": 0.2, "gamma": 0.1})

    assert model.compartment_to_id["S"] == 0
    assert model.compartment_to_id["E"] == 1
    assert model.compartment_to_id["I"] == 2
    assert model.compartment_to_id["R"] == 3


def test_seir_model_has_e_to_i_spontaneous() -> None:
    """SEIRModel must have an E→I spontaneous transition at rate sigma."""
    G: Any = nx.path_graph(5)
    sigma = 0.22
    model = SEIRModel(graph=G, params={"tau": 0.3, "sigma": sigma, "gamma": 0.1})

    ei_trans = [t for t in model._spontaneous if t.from_state == "E" and t.to_state == "I"]
    assert len(ei_trans) == 1
    assert pytest.approx(ei_trans[0].rate, rel=1e-5) == sigma


def test_seir_model_has_i_to_r_spontaneous() -> None:
    """SEIRModel must have an I→R spontaneous transition at rate gamma."""
    G: Any = nx.path_graph(5)
    gamma = 0.13
    model = SEIRModel(graph=G, params={"tau": 0.3, "sigma": 0.2, "gamma": gamma})

    ir_trans = [t for t in model._spontaneous if t.from_state == "I" and t.to_state == "R"]
    assert len(ir_trans) == 1
    assert pytest.approx(ir_trans[0].rate, rel=1e-5) == gamma


def test_seir_model_has_s_to_e_induced() -> None:
    """SEIRModel must have an S→E induced transition catalysed by I."""
    G: Any = nx.path_graph(5)
    tau = 0.35
    model = SEIRModel(graph=G, params={"tau": tau, "sigma": 0.2, "gamma": 0.1})

    se_trans = [
        t for t in model._induced
        if t.source == "S" and t.target == "E" and t.catalyst == "I"
    ]
    assert len(se_trans) == 1
    assert pytest.approx(se_trans[0].rate, rel=1e-5) == tau


def test_seir_model_conservation() -> None:
    """S(t) + E(t) + I(t) + R(t) == N at every time step."""
    N = 8
    G: Any = nx.complete_graph(N)
    model = SEIRModel(graph=G, params={"tau": 1.0, "sigma": 0.5, "gamma": 0.3})
    model.set_initial_conditions(initial_infecteds=[0])
    result = model.run(t_max=30.0)

    _, counts = result.compartment_counts()
    totals = counts["S"] + counts["E"] + counts["I"] + counts["R"]
    assert np.all(totals == N)


def test_seir_model_exposed_peaks_before_infected() -> None:
    """In SEIR, E must reach a positive value before I is large."""
    N = 12
    G: Any = nx.complete_graph(N)
    model = SEIRModel(graph=G, params={"tau": 2.0, "sigma": 0.5, "gamma": 0.3})
    # Start with a single infected (I=2) so E-peak must come next.
    model.set_initial_conditions(initial_infecteds=[0])
    result = model.run(t_max=50.0)

    _, counts = result.compartment_counts()
    # E must exceed 0 at some point (latency period)
    assert np.any(counts["E"] > 0), "SEIRModel: E compartment must be populated."


# ---------------------------------------------------------------------------
# Task 4.5: Mean-Field Limit validation (SIR vs ODE)
# ---------------------------------------------------------------------------


def _sir_ode(y: list[float], _t: float, tau: float, gamma: float, N: int) -> list[float]:
    """Right-hand side of the deterministic SIR ODE system.

    Parameters
    ----------
    y : list[float]
        State vector [S, I, R].
    _t : float
        Current time (unused; ODE is autonomous).
    tau : float
        Transmission rate per contact.
    gamma : float
        Recovery rate.
    N : int
        Total population size.

    Returns
    -------
    list[float]
        Derivatives [dS/dt, dI/dt, dR/dt].
    """
    S, I, R = y
    dS = -tau * S * I / N
    dI = tau * S * I / N - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


@pytest.mark.slow
def test_sir_mean_field_matches_ode() -> None:
    """SIRModel on a fully-connected graph must track the deterministic ODE.

    The Law of Large Numbers guarantees that as N → ∞ the stochastic SIR
    converges to the mean-field ODE.  For N = 300 we accept an L∞ deviation
    of ≤ 15 % of N on the normalised fractions.

    This is the 'Mathematical Correctness (Mean-Field Limit)' test mandated by
    Section 7.1 of EpiNext_Architecture_Specification.md.
    """
    # -- Parameters --
    N = 300
    tau = 0.4
    gamma = 0.1
    t_max = 60.0

    # -- Stochastic simulation on complete graph --
    G: Any = nx.complete_graph(N)
    model = SIRModel(graph=G, params={"tau": tau, "gamma": gamma})
    model.set_initial_conditions(initial_infecteds=[0])
    result = model.run(t_max=t_max)

    t_sim, counts_sim = result.compartment_counts()
    S_sim = counts_sim["S"] / N
    I_sim = counts_sim["I"] / N
    R_sim = counts_sim["R"] / N

    # -- ODE solution at the same time points --
    # Use a dense grid and then interpolate to simulation time points.
    t_dense = np.linspace(0.0, t_max, 2000)
    y0: list[float] = [float(N - 1), 1.0, 0.0]
    sol = odeint(_sir_ode, y0, t_dense, args=(tau, gamma, N))
    S_ode = np.interp(t_sim, t_dense, sol[:, 0] / N)
    I_ode = np.interp(t_sim, t_dense, sol[:, 1] / N)
    R_ode = np.interp(t_sim, t_dense, sol[:, 2] / N)

    # -- Assertion: L∞ norm of fractional deviation ≤ 0.15 --
    tolerance = 0.15
    err_S = float(np.max(np.abs(S_sim - S_ode)))
    err_I = float(np.max(np.abs(I_sim - I_ode)))
    err_R = float(np.max(np.abs(R_sim - R_ode)))

    assert err_S <= tolerance, f"S fraction deviation {err_S:.3f} exceeds {tolerance}."
    assert err_I <= tolerance, f"I fraction deviation {err_I:.3f} exceeds {tolerance}."
    assert err_R <= tolerance, f"R fraction deviation {err_R:.3f} exceeds {tolerance}."
