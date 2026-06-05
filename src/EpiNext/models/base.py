"""BaseEpidemicModel and SimulationResult — Phase 4, Tasks 4.1–4.4.

This module provides the PyTorch-style Object-Oriented API layer that sits
on top of the raw Numba/CSR execution engine.  Researchers interact only
with this layer; the array plumbing is entirely hidden.

Design Principles
-----------------
* **Subclassability first** — users override ``define_transitions()`` to
  describe their compartmental model using plain strings, never raw integers.
* **Translation layer** — ``compartment_to_id`` / ``id_to_compartment``
  bridge Python strings to the ``uint8`` state IDs consumed by the engine.
* **Zero-allocation execution** — ``run()`` pre-allocates the memory pool
  and then calls the compiled Numba kernel; no Python loops during simulation.
* **AGENTS.md compliance** — no global ``random``, no Python for-loops over
  the graph during execution, 100 % NumPy-style docstrings, strict typing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from EpiNext.core.compiler import compile_graph
from EpiNext.core.memory import allocate_pool, extract_history
from EpiNext.core.simulator import run_simulation_general

# ---------------------------------------------------------------------------
# Internal transition record types
# ---------------------------------------------------------------------------


@dataclass
class _SpontaneousTransition:
    """A node-internal rate process (e.g. I → R recovery).

    Attributes
    ----------
    from_state : str
        Compartment name before the transition.
    to_state : str
        Compartment name after the transition.
    rate : float
        Constant transition rate (events per unit time).
    """

    from_state: str
    to_state: str
    rate: float


@dataclass
class _InducedTransition:
    """An edge-driven transmission process (e.g. S → I induced by I neighbour).

    Attributes
    ----------
    source : str
        Compartment name of the node that can be transitioned.
    target : str
        Compartment name the node transitions into.
    catalyst : str
        Compartment name of the neighbouring node that triggers the transition.
    rate : float
        Per-edge rate multiplier applied to each matching neighbour edge weight.
    """

    source: str
    target: str
    catalyst: str
    rate: float


# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------


class SimulationResult:
    """Encapsulates the output of a single epidemic simulation run.

    Stores the raw event history together with enough metadata to reconstruct
    macroscopic compartment counts at every recorded time point.  This object
    is returned by ``BaseEpidemicModel.run()`` and will serve as the input to
    the persistence layer in Phase 6 (``to_sqlite``, ``to_csv``, ``to_json``).

    Parameters
    ----------
    times : np.ndarray
        Monotone float32 array of event timestamps (shape K).
    nodes : np.ndarray
        Integer array of the node index involved in each event (shape K).
    events : np.ndarray
        uint8 array of the *new* compartment state after each event (shape K).
    node_states : np.ndarray
        Final uint8 node state array (shape N) — state of each node at end.
    compartment_map : dict[int, str]
        Maps integer compartment IDs back to string names.
    n_nodes : int
        Total number of nodes in the network.
    initial_states : np.ndarray
        uint8 array of the compartment state assigned to every node at t=0.

    Attributes
    ----------
    times, nodes, events, node_states, compartment_map, n_nodes,
    initial_states
        As above.

    Examples
    --------
    >>> result = model.run(t_max=50.0)
    >>> t_pts, counts = result.compartment_counts()
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t_pts, counts['I'])
    """

    def __init__(
        self,
        times: np.ndarray,
        nodes: np.ndarray,
        events: np.ndarray,
        node_states: np.ndarray,
        compartment_map: dict[int, str],
        n_nodes: int,
        initial_states: np.ndarray,
    ) -> None:
        self.times: np.ndarray = times
        self.nodes: np.ndarray = nodes
        self.events: np.ndarray = events
        self.node_states: np.ndarray = node_states
        self.compartment_map: dict[int, str] = compartment_map
        self.n_nodes: int = n_nodes
        self.initial_states: np.ndarray = initial_states

    def t(self) -> np.ndarray:
        """Returns the recorded event time array.

        Returns
        -------
        np.ndarray
            Float array of event times in ascending order.

        Examples
        --------
        >>> result.t()
        array([0.123, 0.456, ...], dtype=float32)
        """
        return self.times

    def compartment_counts(
        self,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Computes macroscopic compartment counts over time.

        Replays the event history from the recorded initial states to build
        a time series of per-compartment node counts.  The first entry
        (index 0) corresponds to the initial configuration at t = 0.

        Returns
        -------
        tuple[np.ndarray, dict[str, np.ndarray]]
            A 2-tuple ``(time_points, counts)`` where:

            * ``time_points`` is a float64 array of length K + 1 (the initial
              snapshot at 0.0 prepended to the K recorded event times).
            * ``counts`` is a dict mapping each compartment name (e.g.
              ``'S'``, ``'I'``, ``'R'``) to an int64 count array of the same
              length.

        Examples
        --------
        >>> t_pts, counts = result.compartment_counts()
        >>> counts['I']  # infected count at each event
        array([1, 2, 3, ...], dtype=int64)

        Notes
        -----
        The function runs in O(K · C) time where K is the event count and C
        is the number of compartments.  For very long runs, prefer streaming
        directly from the raw ``times`` / ``nodes`` / ``events`` arrays.
        """
        n_comp = len(self.compartment_map)
        K = int(self.times.shape[0])

        # Step 1: Initialise counts from the stored initial state snapshot.
        current_counts = np.zeros(n_comp, dtype=np.int64)
        for state_val in self.initial_states:
            current_counts[int(state_val)] += 1

        # Step 2: Allocate output arrays (K + 1 entries — one per event plus
        # the t = 0 snapshot).
        time_points = np.zeros(K + 1, dtype=np.float64)
        counts_history = np.zeros((K + 1, n_comp), dtype=np.int64)
        counts_history[0] = current_counts.copy()

        # Step 3: Maintain a running per-node state tracker so we can
        # correctly decrement the *old* compartment when an event fires.
        current_node_states = self.initial_states.copy().astype(np.int64)

        for i in range(K):
            node_idx = int(self.nodes[i])
            old_state = int(current_node_states[node_idx])
            new_state = int(self.events[i])

            # Step 4: Update count deltas.
            current_counts[old_state] -= 1
            current_counts[new_state] += 1
            current_node_states[node_idx] = new_state

            time_points[i + 1] = float(self.times[i])
            counts_history[i + 1] = current_counts.copy()

        # Step 5: Build the output dictionary keyed by compartment name.
        result: dict[str, np.ndarray] = {}
        for state_id, state_name in self.compartment_map.items():
            result[state_name] = counts_history[:, state_id]

        return time_points, result


# ---------------------------------------------------------------------------
# BaseEpidemicModel
# ---------------------------------------------------------------------------


class BaseEpidemicModel(ABC):
    """Abstract base class for all epidemic models in EpiNext.

    Provides the PyTorch-style subclassing API described in
    ``EpiNext_Architecture_Specification.md`` §3.1.  Users create custom
    models by subclassing and implementing ``define_transitions()`` using the
    ``add_compartments``, ``add_spontaneous_transition``, and
    ``add_induced_transition`` helper methods.

    Parameters
    ----------
    graph : Any
        A ``networkx.Graph`` or ``networkx.DiGraph`` instance representing the
        contact network.  Typed as ``Any`` to avoid importing networkx stubs.
    params : dict[str, float]
        Dictionary of model parameters (e.g. ``{'tau': 0.3, 'gamma': 0.1}``).
    n_cores : int, optional
        Number of CPU cores to use for parallel ensemble runs.  Pass ``-1`` to
        use all available cores.  Default is ``1``.
    use_gpu : bool, optional
        If ``True``, attempt GPU acceleration via the ``gpu`` hardware
        abstraction layer.  Default is ``False``.

    Attributes
    ----------
    graph : Any
        The networkx graph provided at construction.
    params : dict[str, float]
        The parameter dictionary provided at construction.
    n_cores : int
        Requested CPU core count.
    use_gpu : bool
        GPU acceleration flag.
    compartment_to_id : dict[str, int]
        Forward mapping from compartment string name to uint8-compatible int.
    id_to_compartment : dict[int, str]
        Reverse mapping from uint8-compatible int to compartment string.
    _spontaneous : list[_SpontaneousTransition]
        List of registered spontaneous transitions.
    _induced : list[_InducedTransition]
        List of registered induced transitions.
    _initial_infecteds : list[int]
        Node indices seeded as infected at simulation start.

    Examples
    --------
    >>> import networkx as nx
    >>> from EpiNext.models.base import BaseEpidemicModel
    >>>
    >>> class MySIRModel(BaseEpidemicModel):
    ...     def define_transitions(self):
    ...         self.add_compartments(['S', 'I', 'R'])
    ...         self.add_spontaneous_transition('I', 'R', rate=self.params['gamma'])
    ...         self.add_induced_transition('S', 'I', catalyst='I',
    ...                                    rate=self.params['tau'])
    >>>
    >>> G = nx.barabasi_albert_graph(1000, 3)
    >>> model = MySIRModel(graph=G, params={'tau': 0.3, 'gamma': 0.1})
    >>> model.set_initial_conditions(initial_infecteds=[0, 1, 2])
    >>> result = model.run(t_max=100.0)
    """

    def __init__(
        self,
        graph: Any,
        params: dict[str, float],
        n_cores: int = 1,
        use_gpu: bool = False,
    ) -> None:
        self.graph: Any = graph
        self.params: dict[str, float] = params
        self.n_cores: int = n_cores
        self.use_gpu: bool = use_gpu

        # Step 1: Initialise the compartment translation maps.
        self.compartment_to_id: dict[str, int] = {}
        self.id_to_compartment: dict[int, str] = {}

        # Step 2: Initialise empty transition registries.
        self._spontaneous: list[_SpontaneousTransition] = []
        self._induced: list[_InducedTransition] = []

        # Step 3: Initialise empty initial conditions.
        self._initial_infecteds: list[int] = []

        # Step 4: Invoke subclass transition definition immediately so that
        # the model is fully configured after __init__ returns.
        self.define_transitions()

    # -----------------------------------------------------------------------
    # Abstract interface
    # -----------------------------------------------------------------------

    @abstractmethod
    def define_transitions(self) -> None:
        """Defines the compartments and transition rules for this model.

        Subclasses must implement this method and call:

        * ``add_compartments()`` exactly once,
        * ``add_spontaneous_transition()`` for each internal clock process,
        * ``add_induced_transition()`` for each edge-driven process.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.

        Examples
        --------
        >>> def define_transitions(self):
        ...     self.add_compartments(['S', 'I', 'R'])
        ...     self.add_spontaneous_transition('I', 'R', self.params['gamma'])
        ...     self.add_induced_transition('S', 'I', 'I', self.params['tau'])
        """

    # -----------------------------------------------------------------------
    # Transition registration helpers (Task 4.2)
    # -----------------------------------------------------------------------

    def add_compartments(self, names: list[str]) -> None:
        """Registers a list of compartment names and assigns uint8-compatible IDs.

        The first name in the list is assigned ID 0, the second ID 1, and so
        on.  IDs must fit in a ``uint8`` (0–255), which supports models with
        up to 256 distinct compartments.

        Parameters
        ----------
        names : list[str]
            Ordered list of compartment name strings (e.g. ``['S', 'I', 'R']``).

        Raises
        ------
        ValueError
            If ``names`` contains duplicates.
        ValueError
            If the number of compartments exceeds 255.

        Examples
        --------
        >>> model.add_compartments(['S', 'I', 'R'])
        >>> model.compartment_to_id['S']
        0
        """
        if len(names) != len(set(names)):
            raise ValueError(
                f"Compartment names must be unique; got duplicates in {names!r}."
            )
        if len(names) > 255:
            raise ValueError(
                f"At most 255 compartments are supported (uint8 limit); "
                f"got {len(names)}."
            )
        for idx, name in enumerate(names):
            self.compartment_to_id[name] = idx
            self.id_to_compartment[idx] = name

    def add_spontaneous_transition(
        self,
        from_state: str,
        to_state: str,
        rate: float,
    ) -> None:
        """Registers a node-internal (spontaneous) transition.

        A spontaneous transition fires independently of the node's neighbours
        (e.g. recovery I → R).

        Parameters
        ----------
        from_state : str
            Compartment that must be active for the transition to be possible.
        to_state : str
            Compartment the node enters when the transition fires.
        rate : float
            Constant hazard rate (events per unit time, must be > 0).

        Raises
        ------
        ValueError
            If ``from_state`` or ``to_state`` are not registered compartments.
        ValueError
            If ``rate`` is not strictly positive.

        Examples
        --------
        >>> model.add_spontaneous_transition('I', 'R', rate=0.1)
        """
        self._validate_compartment(from_state)
        self._validate_compartment(to_state)
        if rate <= 0.0:
            raise ValueError(
                f"Spontaneous transition rate must be > 0; got {rate!r}."
            )
        self._spontaneous.append(
            _SpontaneousTransition(from_state=from_state, to_state=to_state, rate=rate)
        )

    def add_induced_transition(
        self,
        source: str,
        target: str,
        catalyst: str,
        rate: float,
    ) -> None:
        """Registers an edge-driven (induced) transition.

        An induced transition fires when a node in ``source`` state has one or
        more neighbours in ``catalyst`` state.  The per-edge propensity is
        ``rate × edge_weight``, so heterogeneous contact rates are
        automatically captured by the edge weight attributes of the graph.

        Parameters
        ----------
        source : str
            Compartment state of the node that can be transitioned.
        target : str
            Compartment state the node enters when the transition fires.
        catalyst : str
            Compartment state of the neighbouring node that induces the event.
        rate : float
            Per-edge transmission rate (must be > 0).

        Raises
        ------
        ValueError
            If any of ``source``, ``target``, or ``catalyst`` are not
            registered compartments.
        ValueError
            If ``rate`` is not strictly positive.

        Examples
        --------
        >>> model.add_induced_transition('S', 'I', catalyst='I', rate=0.3)
        """
        self._validate_compartment(source)
        self._validate_compartment(target)
        self._validate_compartment(catalyst)
        if rate <= 0.0:
            raise ValueError(
                f"Induced transition rate must be > 0; got {rate!r}."
            )
        self._induced.append(
            _InducedTransition(
                source=source, target=target, catalyst=catalyst, rate=rate
            )
        )

    # -----------------------------------------------------------------------
    # Initial conditions (Task 4.4)
    # -----------------------------------------------------------------------

    def set_initial_conditions(self, initial_infecteds: list[int]) -> None:
        """Sets the nodes that start the simulation in the infectious state.

        Infected nodes are placed in the compartment with ID 1 (the second
        compartment registered via ``add_compartments``), which by convention
        is the primary infectious class (e.g. ``'I'`` in SIR / SIS / SEIR).

        Parameters
        ----------
        initial_infecteds : list[int]
            List of node indices (0-based integers from the compiled graph
            ordering) to seed as initially infected.  An empty list starts
            the simulation with no infected nodes.

        Raises
        ------
        ValueError
            If any node index is out of range for the current graph.

        Examples
        --------
        >>> model.set_initial_conditions(initial_infecteds=[0, 1, 2])
        """
        n_nodes = self.graph.number_of_nodes()
        for idx in initial_infecteds:
            if idx < 0 or idx >= n_nodes:
                raise ValueError(
                    f"Node index {idx} is out of range for a graph with "
                    f"{n_nodes} nodes."
                )
        self._initial_infecteds = list(initial_infecteds)

    # -----------------------------------------------------------------------
    # Intervention registration stub (Phase 5 preparation)
    # -----------------------------------------------------------------------

    def add_intervention(
        self,
        time: float,
        action: Any,
    ) -> None:
        """Registers a time-triggered intervention callback (Phase 5 stub).

        Full implementation — including the Numba pause/resume interrupt
        layer — is deferred to Phase 5 (InterventionEngine).  This stub
        accepts the call without error so that the Phase 6 example code
        in the architecture spec can be imported and tested today.

        Parameters
        ----------
        time : float
            Simulation time at which the callback fires.
        action : callable
            Function that receives the engine state and applies a
            modification (e.g. vaccination, parameter change).

        Examples
        --------
        >>> model.add_intervention(time=30.0, action=mass_vaccination)
        """
        # Phase 5 will store these in an InterventionEngine registry and wire
        # them into the Numba event loop via a pause/resume interrupt.
        # For now we accept the call silently.
        _ = time
        _ = action

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _validate_compartment(self, name: str) -> None:
        """Raises ValueError if ``name`` is not a registered compartment.

        Parameters
        ----------
        name : str
            Compartment name to validate.

        Raises
        ------
        ValueError
            If the compartment has not been registered.
        """
        if name not in self.compartment_to_id:
            raise ValueError(
                f"Compartment {name!r} has not been registered. "
                f"Call add_compartments() first."
            )

    def _build_transition_tables(
        self,
    ) -> tuple[np.ndarray, int, np.ndarray, int]:
        """Converts registered transitions into Numba-ready float32 arrays.

        Produces two flat tables suitable for direct ingestion by
        ``run_simulation_general``.  Array shapes use ``max(..., 1)`` to
        guarantee that Numba always receives 2-D arrays with at least one
        row, avoiding shape-0 type-inference issues.

        Returns
        -------
        tuple[np.ndarray, int, np.ndarray, int]
            A 4-tuple ``(spont_table, n_spont, ind_table, n_ind)`` where:

            * ``spont_table`` — float32 array, shape (max(n_spont, 1), 3).
              Columns: [from_state_id, to_state_id, rate].
            * ``n_spont`` — number of valid spontaneous rows.
            * ``ind_table`` — float32 array, shape (max(n_ind, 1), 4).
              Columns: [source_id, target_id, catalyst_id, rate].
            * ``n_ind`` — number of valid induced rows.

        Examples
        --------
        >>> spont, n_s, ind, n_i = model._build_transition_tables()
        >>> spont.shape
        (1, 3)
        """
        n_spont = len(self._spontaneous)
        n_ind = len(self._induced)

        # Step 1: Build spontaneous table — shape (max(n_spont, 1), 3).
        spont_table = np.zeros((max(n_spont, 1), 3), dtype=np.float32)
        for i, tr in enumerate(self._spontaneous):
            spont_table[i, 0] = float(self.compartment_to_id[tr.from_state])
            spont_table[i, 1] = float(self.compartment_to_id[tr.to_state])
            spont_table[i, 2] = float(tr.rate)

        # Step 2: Build induced table — shape (max(n_ind, 1), 4).
        ind_table = np.zeros((max(n_ind, 1), 4), dtype=np.float32)
        for i, tr in enumerate(self._induced):
            ind_table[i, 0] = float(self.compartment_to_id[tr.source])
            ind_table[i, 1] = float(self.compartment_to_id[tr.target])
            ind_table[i, 2] = float(self.compartment_to_id[tr.catalyst])
            ind_table[i, 3] = float(tr.rate)

        return spont_table, n_spont, ind_table, n_ind

    # -----------------------------------------------------------------------
    # Simulation entry-point (Task 4.1)
    # -----------------------------------------------------------------------

    def run(self, t_max: float, track_history: bool = True) -> SimulationResult:
        """Compiles the graph and runs the stochastic Gillespie simulation.

        This is the single public entry-point that bridges the OOP API to the
        raw Numba engine.  The execution path is:

        1. Compile the networkx graph into CSR arrays (cached if identical).
        2. Apply initial conditions to the ``node_states`` array.
        3. Translate registered transitions into float32 Numba-compatible tables.
        4. Pre-allocate the zero-allocation event history memory pool.
        5. Invoke the compiled ``_run_simulation_general_cpu`` kernel.
        6. Extract the event history and wrap it in a ``SimulationResult``.

        Parameters
        ----------
        t_max : float
            Simulation horizon in time units.  The loop terminates when the
            global clock exceeds this value or all events are exhausted.
        track_history : bool, optional
            If ``True`` (default), record the full event history.  If
            ``False``, the pool size is set to 1 to minimise memory usage
            (only the final node states are accessible).

        Returns
        -------
        SimulationResult
            Encapsulated simulation output with full event history and
            compartment time-series extraction methods.

        Raises
        ------
        ValueError
            If no compartments have been registered (``define_transitions()``
            was not called or is empty).
        ValueError
            If ``t_max`` is not positive.

        Examples
        --------
        >>> model.set_initial_conditions(initial_infecteds=[0, 1])
        >>> result = model.run(t_max=100.0)
        >>> t_pts, counts = result.compartment_counts()
        """
        if not self.compartment_to_id:
            raise ValueError(
                "No compartments registered. Call add_compartments() "
                "inside define_transitions()."
            )
        if t_max <= 0.0:
            raise ValueError(f"t_max must be positive; got {t_max!r}.")

        # Step 1: Compile the networkx graph to CSR arrays.
        compiled = compile_graph(self.graph)

        # Step 2: Apply initial conditions — default all nodes to state 0 (S).
        compiled.node_states[:] = 0

        # Infectious compartment is ID 1 by convention.
        infectious_id = 1
        for node_idx in self._initial_infecteds:
            compiled.node_states[node_idx] = np.uint8(infectious_id)

        # Snapshot the initial state for later time-series reconstruction.
        initial_states = compiled.node_states.copy()

        # Step 3: Build Numba-compatible transition tables.
        spont_table, n_spont, ind_table, n_ind = self._build_transition_tables()

        # Step 4: Pre-allocate the event memory pool.
        n_nodes = int(compiled.node_states.shape[0])
        pool_capacity = n_nodes * 20 if track_history else 1
        pool = allocate_pool(max_events=pool_capacity)

        # Step 5: Unpack CompiledGraph into the tuple expected by the kernel.
        unpacked: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ] = (
            compiled.indptr,
            compiled.indices,
            compiled.edge_weights,
            compiled.node_weights,
            compiled.node_states,
        )

        # Step 6: Execute the Numba kernel.
        run_simulation_general(
            unpacked,
            pool.times,
            pool.nodes,
            pool.events,
            pool.cursor,
            t_max,
            spont_table,
            n_spont,
            ind_table,
            n_ind,
        )

        # Step 7: Extract recorded events and wrap in SimulationResult.
        times, nodes, events = extract_history(pool)

        return SimulationResult(
            times=times,
            nodes=nodes,
            events=events,
            node_states=compiled.node_states.copy(),
            compartment_map=dict(self.id_to_compartment),
            n_nodes=n_nodes,
            initial_states=initial_states,
        )
