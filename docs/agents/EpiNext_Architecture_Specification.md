# **EpiNext Software Architecture & Design Specification**

**Project:** Modernization of Epidemics on Networks (EpiNext)

**Document Version:** 2.0 (Extended HPC & Web-UI Specification)

**Document Date:** 5th June, 2026

**Target Audience:** Jules (Lead AI Software Engineer / HPC Architect)

## **1\. Executive Summary**

The original `EoN` (Epidemics on Networks) library is a foundational tool in network epidemiology, providing algorithms for simulating continuous-time and discrete-time infectious disease dynamics over `networkx` graphs. However, the legacy codebase suffers from heavy procedural design, brittle global state management (specifically concerning the Python global `random` module), and a lack of modern object-oriented extensibility.

The primary mandate of **EpiNext** is to completely rewrite the stochastic simulation engine from the ground up. This modernization effort must achieve six primary objectives:

1. **Uncompromising Extensibility (The PyTorch Pattern):** Provide a highly Object-Oriented API where researchers can define complex disease models (SIR, SEIR, SIRS, or custom multi-stage diseases) via intuitive subclassing, without touching the underlying array logic.  
2. **Deterministic & Reproducible Timelines:** Eradicate global random state. Implement a localized, hash-based Random Number Generator (RNG) stream approach to ensure that local graph perturbations do not cause cascading butterfly effects across the entire simulation timeline.  
3. **Blazing Fast Execution (CPU & GPU):** Abstract away the slow Python-level `networkx` graph representations during execution. The engine must compile graphs into flat, cache-optimized NumPy arrays (Compressed Sparse Row \- CSR format) and execute the core Gillespie/discrete event loops using `numba` Just-In-Time (JIT) compilation and optional CUDA GPU offloading.  
4. **Dynamic Interventions & Callbacks:** Support mid-simulation state changes. Researchers must be able to inject policies (e.g., lockdowns, targeted vaccinations, quarantine) dynamically based on time or simulation state thresholds.  
5. **Advanced Memory Management:** Optimize for extreme scale (millions of nodes) using state bit-packing, memory pooling, and minimized garbage collection in hot loops.  
6. **Web-UI & Persistence Ready:** Architect the backend to seamlessly support a robust web-based frontend. This requires built-in capabilities to save simulation states to a structured SQLite database, export data to CSV/JSON, and serve a REST API for a Bootstrap-powered dashboard.

This document serves as the absolute, non-negotiable source of truth and execution plan for Jules.

## **2\. Background and Mathematical Motivation**

### **2.1 The "Math vs. Software" Debt**

The v1 codebase prioritized mathematical correctness over software engineering principles. Functions like `fast_SIR` and `Gillespie_SIR` are monolithic procedural blocks spanning hundreds of lines. Adding a new disease compartment (e.g., `E` for Exposed) required either writing a completely new monolithic function or using an extremely clunky generic network builder. This lack of modularity makes the library nearly impossible to maintain or extend.

### **2.2 The Global RNG Timeline Desynchronization Problem**

In standard continuous-time Gillespie algorithms (specifically the Direct Method or Next Reaction Method), the algorithm calculates a total system rate $R$, draws a time step $\Delta t \sim \text{Exp}(R)$, and selects an event with probability proportional to its rate.

Because the legacy EoN relies on the global Python `random` module, a change in one corner of the network creates a catastrophic desynchronization of the RNG sequence:

* **Scenario:** A researcher runs a simulation. At $t=5$, Node A infects Node B.  
* **Intervention:** The researcher rewinds, and artificially "vaccinates" Node Z (a node 10 degrees of separation away from A) at $t=0$.  
* **The Legacy Failure:** Because Node Z's events are removed from the global queue, the total rate $R$ changes, altering the $\Delta t$ drawn for *all* subsequent events. Consequently, the RNG rolls for Node A and B shift. Node A might no longer infect Node B, entirely due to a pseudo-random sequence shift, not because of the network topology or disease mechanics.

**The Fix:** EpiNext must decouple the event timeline from a global state, utilizing a strict, context-aware hash-based PRNG algorithm.

## **3\. Core Architectural Pillars & High-Performance Computing (HPC) Constraints**

Jules must adhere strictly to the following architectural pillars to ensure the library can scale to national-level population networks.

### **3.1 Object-Oriented Subclassing API**

EpiNext must hide the extreme complexity of continuous-time event queues, memory management, and CSR arrays behind a clean, subclassable `BaseEpidemicModel`.

**Mandates:**

* Standard models (`SIRModel`, `SISModel`, `SEIRModel`) must be shipped out-of-the-box as subclasses.  
* Researchers define custom models by inheriting from `BaseEpidemicModel` and overriding `define_transitions()`.  
* The API must accept standard `networkx.Graph` and `networkx.DiGraph` objects, parsing node and edge attributes dynamically.

### **3.2 High-Performance "Numba \+ CSR" Execution Engine**

Python `for`-loops and dictionary lookups are strictly forbidden within the simulation execution loop.

**Data Structure Mandates (The Graph Compiler):** When a networkx graph is ingested, it must be compiled into the following flat 1D arrays:

1. `indptr` (Array of `int32`): Size $N + 1$. Points to the start of a node's edges in the indices array.  
2. `indices` (Array of `int32`): Size $E$. The target nodes for all edges.  
3. `edge_weights` (Array of `float32`): Size $E$. The transmission probability/rate multipliers for each edge.  
4. `node_states` (Array of `uint8`): Size $N$. The current compartment state of each node. By using `uint8`, we can pack up to 256 distinct states (e.g., S=0, I=1, R=2), drastically reducing cache misses during iteration.

**Execution Mandates:**

* **JIT Compilation:** The actual stochastic event loop must be a pure function compiled using `@numba.njit`.  
* **Zero Allocation Hot-Loops:** Inside the Numba compiled loop, there must be zero memory allocations (no `np.zeros` or list appends). Pre-allocate a large output buffer (Memory Pool) for the event history and truncate it at the end of the simulation.  
* **Hardware Acceleration:** \* **Multi-Core (CPU):** Use `numba.prange` for bulk synchronous state updates (if using discrete time) and Python `concurrent.futures.ProcessPoolExecutor` for running parallel Monte Carlo simulation ensembles.  
  * **GPU Offloading (CUDA & Metal):** Expose a `use_gpu=True` flag. When active, utilize @numba.cuda.jit to dispatch heavy array operations to NVIDIA GPUs. Additionally, since MacBooks are highly prevalent in the scientific community, the engine must support Apple Metal acceleration. Architect a hardware abstraction layer so that Apple Silicon users can leverage Metal Performance Shaders (e.g., integrating with Apple's MLX framework or equivalent Apple Silicon backends) alongside the standard CUDA implementation.

### **3.3 The Reproducible Hash-Based PRNG**

To solve the global RNG desynchronization problem, the stochastic roll for any given event must rely exclusively on its localized context.

**Mandates:**

* **No `import random`.** The standard library `random` is forbidden.  
* **The Hash Function:** To determine the outcome of an interaction (e.g., transmission from $u$ to $v$ at time $t$, the engine must generate a deterministic seed: `event_seed = hash((base_simulation_seed, node_u, node_v, event_type_id, current_time_step))`
* Use this integer seed to instantiate a lightweight, localized PRNG (e.g., utilizing an inline XOR-shift algorithm or Numba's internal thread-safe RNG generation constrained by the seed).

### **3.4 Dynamic Interventions & Callbacks**

Epidemics are not static. The library must support runtime modifications to the simulation parameters and graph structure.

**Mandates:**

* Implement an `InterventionEngine`.  
* Support **Time-Triggered Interventions:** e.g., `model.add_intervention(time=30.0, action=lockdown_func)` where `lockdown_func` multiplies all `edge_weights` by 0.2.  
* Support **State-Triggered Interventions:** e.g., `model.add_intervention(condition=lambda state: state.I > 1000, action=vaccinate_func)`.  
* Support **Time-Varying Parameters:** e.g., passing a function `tau(t) = 0.5 * (1 + sin(t))` instead of a static float for transmission rates, allowing for seasonality modeling.

## **4\. Web-UI & Persistence Architecture (The "EoN-Dash")**

While Jules is primarily building the core simulation engine, the architecture must seamlessly plug into a future Web-based User Interface. Researchers should be able to launch a local web server to run, manage, and visualize simulations without writing Python code.

### **4.1 Persistence Layer (SQLite)**

The `SimulationResult` object must contain a robust export mechanism.

* **Method:** `result.to_sqlite(db_path="epinext_experiments.db", run_name="Run_1")` 
* **Schema Definition:** Jules must define the following tables via SQLAlchemy or raw parameterized SQL queries:  
  * `experiments` (id, run\_name, timestamp, graph\_nodes, graph\_edges, parameters\_json)  
  * `time_series` (id, experiment\_id, time\_t, state\_S, state\_E, state\_I, state\_R, ...)  
  * `interventions_log` (id, experiment\_id, time\_t, intervention\_description)

### **4.2 REST API Specification (FastAPI Prep)**

The engine must expose interface classes that can be easily wrapped by FastAPI endpoints:

* `POST /api/v1/simulate`: Accepts JSON payload with model type, network parameters (e.g., Erdos-Renyi params), and disease parameters. Returns a task ID.  
* `GET /api/v1/results/{task_id}`: Retrieves the macroscopic time series data in JSON format for the frontend to plot.

### **4.3 Frontend Vision (Bootstrap 5\)**

The future UI will be built using plain HTML, vanilla JavaScript, and **Bootstrap 5**. No complex React/Angular toolchains.

* **Layout:** A clean, responsive dashboard using Bootstrap Grids (`container-fluid`, `row`, `col-md-3` for settings sidebar, `col-md-9` for visualization canvas).  
* **Components:** Use Bootstrap Cards (`card`) to group simulation parameters, Offcanvas (`offcanvas`) for advanced settings/interventions, and Modals (`modal`) for export confirmations.  
* **Visualization:** The JSON payload exported by the backend will be ingested by a frontend plotting library (like Chart.js or Plotly.js) to render the $S(t), I(t), R(t)$ curves interactively.

## **5\. Technology Stack & Tooling Requirements**

The project will strictly utilize a modern, high-performance Python ecosystem.

* **Package Management:** `uv`. Project must be initialized via `uv init`. The `pyproject.toml` is the sole source of truth.  
* **Linting & Formatting:** `ruff`. Jules must configure `ruff` to enforce strict formatting (88 character line length, `I` for import sorting, strict PEP8).  
* **Type Hinting:** Extensive use of Python's modern typing (`list[int], dict[str, float]`). The codebase must pass `mypy --strict`.  
* **Testing:** `pytest`. Property-based testing via the `hypothesis` library is highly encouraged to test algorithmic invariants.  
* **Core Dependencies:**  
  * `numpy` \>= 1.24 (Array manipulation, RNG).  
  * `numba` \>= 0.58 (JIT compilation and CUDA GPU support).  
  * `networkx` \>= 3.0 (Public-facing graph structures).  
  * `pandas` (For clean CSV exporting).  
  * `sqlite3` (Built-in, for persistence).  
* **Documentation:** Standard Numpy style docstrings for all public classes.

## **6\. Proposed User-Facing API**

Jules must build the internal logic to support the following exact User Experience (UX). Notice how the public-facing code completely abstracts the complex Numba arrays and event queues.

### **Example 1: Creating an Extensible Custom Model (SEIR) with GPU & Interventions**

```python
import networkx as nx  
import numpy as np  
from EpiNext.core import BaseEpidemicModel  
from EpiNext.interventions import TargetNodes

class SEIRModel(BaseEpidemicModel):  
    def define_transitions(self):  
        # 1. Define states (Mapped to uint8 0, 1, 2, 3 internally)  
        self.add_compartments(['S', 'E', 'I', 'R'])  
          
        # 2. Spontaneous transitions (Internal node clock)  
        self.add_spontaneous_transition('E', 'I', rate=self.params['sigma'])  
        self.add_spontaneous_transition('I', 'R', rate=self.params['gamma'])  
          
        # 3. Induced transitions (Edge-based transmission)  
        self.add_induced_transition(source='S', target='E', catalyst='I', rate=self.params['tau'])

# A. Generate a massive NetworkX graph  
G = nx.barabasi_albert_graph(n=100_000, m=3)

# B. Instantiate with Multi-Core and GPU acceleration  
model = SEIRModel(  
    graph=G,   
    params={'tau': 0.15, 'sigma': 0.2, 'gamma': 0.1},   
    n_cores=-1,   
    use_gpu=True  
)

# C. Define an intervention: Vaccinate 20% of Susceptibles at day 30  
def mass_vaccination(engine_state):  
    susceptible_nodes = engine_state.get_nodes_in_state('S')  
    targets = np.random.choice(susceptible_nodes, size=int(len(susceptible_nodes)*0.2), replace=False)  
    engine_state.update_nodes(targets, 'R')

model.add_intervention(time=30.0, action=mass_vaccination)

# D. Run, track histories, and persist to SQLite  
model.set_initial_conditions(initial_infecteds=[0, 1, 2, 3, 4])  
results = model.run(t_max=150.0, track_history=True)

# E. Save to local SQLite database for the future Web UI dashboard  
results.to_sqlite("local_simulations.db", run_name="SEIR_Vaccination_Scenario")  
results.to_csv("seir_results.csv")
```

## **7\. Quality Assurance & Test-Driven Development (TDD) Mandate**

Jules is strictly forbidden from writing core logic before defining the test conditions. **TDD is mandatory.** The `pytest` suite must establish the mathematical and algorithmic contracts first.

### **7.1 Required Test Categories**

1. **RNG Isolation Tests (The Butterfly Effect Check \- Crucial):**  
   * **Setup:** Simulate an SIR model on a graph of 1,000 nodes. Log the exact state transition timeline of Node 500\.  
   * **Perturbation:** Run the exact same model, but artificially introduce a completely disconnected component of 10 nodes undergoing rapid, chaotic infections.  
   * **Assertion:** Node 500's exact timeline (infection time, recovery time) *must be bit-for-bit identical* in both tests. This mathematically proves the Hash-based PRNG is working and global state contamination is eradicated.  
2. **Graph Compilation Tests:**  
   * Verify that NetworkX edge lists accurately map to the `indptr` and `indices` arrays.  
   * Assert that node attributes correctly map to the `node_weights` float arrays.  
3. **Mathematical Correctness (Mean-Field Limit):**  
   * Simulate an SIR model on a massive Complete Graph (Fully Connected) with $N = 10,000$.  
   * Assert that the resulting macroscopic curve ($S(t), I(t), R(t)$) tightly matches the numerical integration of the standard SIR Ordinary Differential Equations (ODEs) using `scipy.integrate.odeint`, within an acceptable stochastic margin of error (e.g., $L@$ norm $< 0.05$).  
4. **Memory Leak Tests:**  
   * Run 10,000 iterations of a small graph simulation. Assert that system memory consumption remains stable, proving the internal Memory Pool array pre-allocation is working and the Python garbage collector is not being thrashed.

## **8\. Iterative Execution Plan for Jules**

Jules must not attempt to generate the entire library in a single response (which leads to context collapse and hallucinated code). Jules must strictly follow this phased iterative plan. The user will prompt Jules to proceed to the next phase upon successful completion of the current one.

### **Phase 1: Project Scaffolding & CI/CD**

* Check the initialised `uv` project structure.  
* Check the `pyproject.toml` with strict `ruff` configurations, `mypy` settings, and dependencies (`numba`, `networkx`, `pytest`, `hypothesis`). If configurations are missing, configure them.

### **Phase 2: Graph Compilation & Memory Management**

* Write tests for `CompiledGraph` mapping.  
* Implement `src/core/compiler.py`.  
* Implement the $N + 1$ `indptr` and $E$ `indices` mapping logic, heavily utilizing `uint8` and `float32` arrays for cache optimization.

### **Phase 3: The Hash-Based RNG & Numba Engine**

* Write the RNG Butterfly Effect isolation tests.  
* Implement the pure-function, Numba-compiled static event loop (`src/core/simulator.py`).  
* Implement the CUDA and Metal (`use_gpu=True`) device allocation and kernel dispatches for bulk state updates.

### **Phase 4: The PyTorch-Style OOP Wrapper & Standard Models**

* Implement `BaseEpidemicModel`. It must serve as the seamless bridge linking Python strings/objects to the raw Numba integers/arrays.  
* Implement `SIRModel` and `SISModel` as subclasses.  
* Write tests validating the stochastic outputs against ODE limits.

### **Phase 5: Dynamic Interventions Engine**

* Implement the callback system allowing users to pass Python functions that safely interrupt the Numba event loop, modify array states (e.g., vaccination), and resume the engine.

### **Phase 6: Outputs, Persistence, and Web-UI Prep**

* Build `SimulationResult` object.  
* Implement `to_sqlite()`, mapping the data structures to the defined SQL schema.  
* Implement `to_csv()` and `to_json()`.

### **Phase 7: Final Documentation & Type Verification**

* Generate Numpy style compliant docstrings for every method.  
* Run a final `mypy --strict` and `ruff check --fix` pass over the entire codebase to ensure production readiness.

***End of Specification.*** Jules, acknowledge your understanding of these constraints. Wait for the user's prompt before beginning Phase 1\.