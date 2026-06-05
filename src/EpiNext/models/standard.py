"""Standard pre-built epidemic models for EpiNext (Phase 4, Tasks 4.6–4.8).

Provides three canonical compartmental models as ready-to-use subclasses of
``BaseEpidemicModel``.  Each class overrides only ``define_transitions()``
to register its compartments and transition rules; all heavy-lifting is
delegated to the base class and the Numba engine.

Models
------
SIRModel
    Susceptible → Infectious → Removed.  Classic irreversible epidemic model.
SISModel
    Susceptible → Infectious → Susceptible.  Cyclic model with no immunity.
SEIRModel
    Susceptible → Exposed → Infectious → Removed.  Adds a latency period.

Usage
-----
>>> import networkx as nx
>>> from EpiNext.models.standard import SIRModel
>>>
>>> G = nx.barabasi_albert_graph(n=10_000, m=3)
>>> model = SIRModel(graph=G, params={'tau': 0.15, 'gamma': 0.1})
>>> model.set_initial_conditions(initial_infecteds=[0, 1, 2])
>>> result = model.run(t_max=150.0)
>>> t_pts, counts = result.compartment_counts()
"""

from __future__ import annotations

from EpiNext.models.base import BaseEpidemicModel


class SIRModel(BaseEpidemicModel):
    """Susceptible–Infectious–Removed epidemic model.

    The classic Kermack–McKendrick (1927) SIR model implemented as a
    ``BaseEpidemicModel`` subclass.  The three compartments are:

    * **S (0)** — Susceptible: can be infected by infectious neighbours.
    * **I (1)** — Infectious: spreads the disease and recovers spontaneously.
    * **R (2)** — Removed: immune; takes no further part in transmission.

    Parameters
    ----------
    graph : Any
        Contact network (``networkx.Graph`` or ``networkx.DiGraph``).
    params : dict[str, float]
        Must contain:

        * ``'tau'`` — per-edge transmission rate (S → I, > 0).
        * ``'gamma'`` — spontaneous recovery rate (I → R, > 0).

    n_cores : int, optional
        CPU parallelism for ensemble runs (default 1).
    use_gpu : bool, optional
        GPU acceleration flag (default False).

    Examples
    --------
    >>> import networkx as nx
    >>> from EpiNext.models.standard import SIRModel
    >>> G = nx.complete_graph(100)
    >>> model = SIRModel(graph=G, params={'tau': 0.3, 'gamma': 0.1})
    >>> model.set_initial_conditions(initial_infecteds=[0])
    >>> result = model.run(t_max=50.0)
    >>> t_pts, counts = result.compartment_counts()
    >>> print(counts['R'][-1], 'nodes recovered')
    """

    def define_transitions(self) -> None:
        """Registers the three SIR compartments and their two transitions.

        Compartments
        ------------
        S = 0, I = 1, R = 2

        Transitions
        -----------
        * Spontaneous: I → R at rate ``params['gamma']`` (recovery).
        * Induced:     S → I (catalyst I) at rate ``params['tau']``
          (infection via an infectious neighbour).

        Raises
        ------
        KeyError
            If ``'tau'`` or ``'gamma'`` are missing from ``self.params``.

        Examples
        --------
        >>> model = SIRModel(graph=G, params={'tau': 0.3, 'gamma': 0.1})
        >>> model.compartment_to_id
        {'S': 0, 'I': 1, 'R': 2}
        """
        # Step 1: Register compartments in the canonical SIR order so that
        # the infectious class maps to ID 1 (the engine's default seed state).
        self.add_compartments(["S", "I", "R"])

        # Step 2: Spontaneous recovery clock — every infectious node ticks
        # independently at rate gamma.
        self.add_spontaneous_transition(
            from_state="I",
            to_state="R",
            rate=float(self.params["gamma"]),
        )

        # Step 3: Edge-driven infection — susceptible nodes accumulate
        # transmission pressure proportional to the number and weight of
        # their infectious neighbours.
        self.add_induced_transition(
            source="S",
            target="I",
            catalyst="I",
            rate=float(self.params["tau"]),
        )


class SISModel(BaseEpidemicModel):
    """Susceptible–Infectious–Susceptible epidemic model.

    The cyclic SIS model (Weiss & Dishon, 1971) in which recovered nodes
    return immediately to the susceptible pool, enabling persistent endemic
    states.  Appropriate for diseases that confer no lasting immunity (e.g.
    common cold, seasonal influenza-like illnesses).

    Compartments
    ------------
    * **S (0)** — Susceptible.
    * **I (1)** — Infectious; recovers and becomes susceptible again.

    Parameters
    ----------
    graph : Any
        Contact network.
    params : dict[str, float]
        Must contain:

        * ``'tau'`` — per-edge transmission rate (S → I, > 0).
        * ``'gamma'`` — spontaneous return-to-susceptible rate (I → S, > 0).

    n_cores : int, optional
        Default 1.
    use_gpu : bool, optional
        Default False.

    Examples
    --------
    >>> import networkx as nx
    >>> from EpiNext.models.standard import SISModel
    >>> G = nx.watts_strogatz_graph(200, 4, 0.1)
    >>> model = SISModel(graph=G, params={'tau': 0.6, 'gamma': 0.3})
    >>> model.set_initial_conditions(initial_infecteds=[0, 1])
    >>> result = model.run(t_max=100.0)
    """

    def define_transitions(self) -> None:
        """Registers the two SIS compartments and their transitions.

        Compartments
        ------------
        S = 0, I = 1

        Transitions
        -----------
        * Spontaneous: I → S at rate ``params['gamma']`` (recovery to susceptible).
        * Induced:     S → I (catalyst I) at rate ``params['tau']``
          (infection via infectious neighbour).

        Raises
        ------
        KeyError
            If ``'tau'`` or ``'gamma'`` are missing from ``self.params``.

        Examples
        --------
        >>> model = SISModel(graph=G, params={'tau': 0.5, 'gamma': 0.2})
        >>> model.compartment_to_id
        {'S': 0, 'I': 1}
        """
        # Step 1: Only two compartments — no permanent removal class.
        self.add_compartments(["S", "I"])

        # Step 2: Recovery returns node to susceptible (cyclic, no immunity).
        self.add_spontaneous_transition(
            from_state="I",
            to_state="S",
            rate=float(self.params["gamma"]),
        )

        # Step 3: Infection via infectious neighbours.
        self.add_induced_transition(
            source="S",
            target="I",
            catalyst="I",
            rate=float(self.params["tau"]),
        )


class SEIRModel(BaseEpidemicModel):
    """Susceptible–Exposed–Infectious–Removed epidemic model.

    The SEIR model (Anderson & May, 1979) extends SIR with an *Exposed*
    latency class.  Newly infected nodes first enter the non-infectious ``E``
    compartment and progress to ``I`` at a rate ``sigma`` (the inverse of the
    mean incubation period).  This is appropriate for diseases with a
    significant incubation period (e.g. COVID-19, measles, Ebola).

    Compartments
    ------------
    * **S (0)** — Susceptible.
    * **E (1)** — Exposed (infected but not yet infectious).
    * **I (2)** — Infectious.
    * **R (3)** — Removed.

    Parameters
    ----------
    graph : Any
        Contact network.
    params : dict[str, float]
        Must contain:

        * ``'tau'`` — per-edge transmission rate (S → E, > 0).
        * ``'sigma'`` — progression rate from latent to infectious
          (E → I, > 0).  Equal to 1 / mean_incubation_period.
        * ``'gamma'`` — recovery rate (I → R, > 0).

    n_cores : int, optional
        Default 1.
    use_gpu : bool, optional
        Default False.

    Notes
    -----
    Because ``E`` has ID 1, the ``set_initial_conditions()`` helper seeds
    nodes into state I=1 (Exposed) by default.  For SEIR it is more natural
    to seed nodes as *Infectious* (ID 2).  Pass ``initial_infecteds`` to
    ``set_initial_conditions()`` and then manually set those node states to 2
    before calling ``run()`` — or override ``run()`` in a subclass.

    In practice, seeding a small number of Exposed nodes is epidemiologically
    equivalent to seeding Infectious nodes for large networks.

    Examples
    --------
    >>> import networkx as nx
    >>> from EpiNext.models.standard import SEIRModel
    >>> G = nx.barabasi_albert_graph(500, 3)
    >>> model = SEIRModel(graph=G, params={'tau': 0.3, 'sigma': 0.2, 'gamma': 0.1})
    >>> model.set_initial_conditions(initial_infecteds=[0])
    >>> result = model.run(t_max=100.0)
    >>> t_pts, counts = result.compartment_counts()
    >>> print('Peak I:', counts['I'].max())
    """

    def define_transitions(self) -> None:
        """Registers the four SEIR compartments and their three transitions.

        Compartments
        ------------
        S = 0, E = 1, I = 2, R = 3

        Transitions
        -----------
        * Spontaneous: E → I at rate ``params['sigma']`` (progression).
        * Spontaneous: I → R at rate ``params['gamma']`` (recovery).
        * Induced:     S → E (catalyst I) at rate ``params['tau']``
          (latent infection via an infectious neighbour).

        Raises
        ------
        KeyError
            If ``'tau'``, ``'sigma'``, or ``'gamma'`` are missing from
            ``self.params``.

        Examples
        --------
        >>> model = SEIRModel(graph=G,
        ...     params={'tau': 0.3, 'sigma': 0.2, 'gamma': 0.1})
        >>> model.compartment_to_id
        {'S': 0, 'E': 1, 'I': 2, 'R': 3}
        """
        # Step 1: Register four compartments; E is at ID 1 so that the base
        # class's initial-conditions helper seeds nodes to the Exposed class.
        self.add_compartments(["S", "E", "I", "R"])

        # Step 2: Latency progression — exposed nodes become infectious
        # independently of their neighbours.
        self.add_spontaneous_transition(
            from_state="E",
            to_state="I",
            rate=float(self.params["sigma"]),
        )

        # Step 3: Recovery — infectious nodes are removed.
        self.add_spontaneous_transition(
            from_state="I",
            to_state="R",
            rate=float(self.params["gamma"]),
        )

        # Step 4: Exposure via contact — susceptible nodes exposed to
        # infectious neighbours enter the latent compartment.
        self.add_induced_transition(
            source="S",
            target="E",
            catalyst="I",
            rate=float(self.params["tau"]),
        )
