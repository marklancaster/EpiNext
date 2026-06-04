# **EoNv2 Project Task Breakdown (WBS)**

**Project:** Modernization of Epidemics on Networks (EoNv2)

**Document Version:** 2.0

**Document Date:** 5th June, 2026

**Reference:** `EpiNext_Architecture_Specification.md` (Version 2.0)

This document provides a granular task breakdown for Jules, expanding on the 7-phase iterative execution plan. Each phase must be fully completed and tested before moving to the next.

## **Phase 1: Project Scaffolding & CI/CD**

**Goal:** Establish a robust, modern Python development environment.

* \[ \] **Task 1.1:** Check the project initialisation.  
* \[ \] **Task 1.2:** Configure `pyproject.toml` with project metadata and core dependencies (`numpy`, `numba`, `networkx`, `pandas`, `scipy`) if needed.  
* \[ \] **Task 1.3:** Add development dependencies to `pyproject.toml` (`pytest`, `hypothesis`, `ruff`, `mypy`) if needed.  
* \[ \] **Task 1.4:** Configure strict `ruff` rules in `pyproject.toml` (line-length=88, `I` for import sorting, PEP8 compliance).  
* \[ \] **Task 1.5:** Configure `mypy` for strict typing (`strict = true`).  
* \[ \] **Task 1.6:** Scaffold the target directory structure:  
  * `src/core/`  
  * `src/models/`  
  * `src/interventions/`  
  * `src/utils/`  
  * `tests/`  
* \[ \] **Task 1.7:** Set up a basic `pytest` configuration file and verify the test runner executes correctly on an empty suite.

## **Phase 2: Graph Compilation & Memory Management**

**Goal:** Abstract NetworkX graphs into high-performance, cache-friendly flat arrays.

* \[ \] **Task 2.1:** Define the `CompiledGraph` Python Dataclass structure (Type-hinted for NumPy arrays).  
* \[ \] **Task 2.2:** **(TDD)** Write `pytest` assertions verifying that simple NetworkX edge lists accurately map to expected CSR arrays (`indptr` and `indices`).  
* \[ \] **Task 2.3:** **(TDD)** Write tests to ensure node/edge weights accurately map to the correct indices in `float32` arrays.  
* \[ \] **Task 2.4:** Implement `src/core/compiler.py` to parse NetworkX directed and undirected graphs.  
* \[ \] **Task 2.5:** Implement the core CSR mapping logic (Size $N + 1$ for `indptr`, Size $E$ for `indices`).  
* \[ \] **Task 2.6:** Implement strict memory typing: allocate `node_states` as `uint8`, and `edge_weights` as `float32` to minimize cache misses.  
* \[ \] **Task 2.7:** Implement the Graph Caching mechanism to bypass the expensive compilation step if the same graph is passed sequentially.

## **Phase 3: The Hash-Based RNG & Numba Engine**

**Goal:** Build the blazing-fast execution loop and eradicate global RNG state.

* \[ \] **Task 3.1:** **(TDD \- Crucial)** Write the "Butterfly Effect Check" tests. Assert that perturbing a disconnected sub-graph does not alter the exact bit-for-bit timeline of an isolated node.  
* \[ \] **Task 3.2:** Implement the localized, hash-based deterministic PRNG algorithm (using Numba-compatible random initialization).  
* \[ \] **Task 3.3:** Design the Numba zero-allocation Memory Pool strategy (pre-allocating large arrays for tracking the event history to avoid garbage collection).  
* \[ \] **Task 3.4:** Implement the continuous-time Gillespie (Next Reaction/Direct Method) event loop using `@numba.njit`.  
* \[ \] **Task 3.5:** Implement `numba.prange` support for bulk state updates where mathematically applicable.  
* \[ \] **Task 3.6:** Implement CUDA and Metal GPU hooks (`use_gpu=True flag`) using `@numba.cuda.jit` for NVIDIA, and architect a hardware abstraction layer for Apple Metal (e.g., using Apple's MLX framework or similar) for macOS users, including host-to-device memory transfer logic.

## **Phase 4: The PyTorch-Style OOP Wrapper & Standard Models**

**Goal:** Build the user-friendly API that abstracts the Numba engine.

* \[ \] **Task 4.1:** Implement the `BaseEpidemicModel` class.  
* \[ \] **Task 4.2:** Implement transition definitions: `add_compartments()`, `add_spontaneous_transition()`, and `add_induced_transition()`.  
* \[ \] **Task 4.3:** Implement the translation layer (mapping string-based states like 'S', 'I' to `uint8` integers 0, 1 for the Numba engine).  
* \[ \] **Task 4.4:** Implement `set_initial_conditions()`.  
* \[ \] **Task 4.5:** **(TDD)** Write Mean-Field Limit tests comparing a fully connected graph's stochastic simulation to a `scipy.integrate.odeint` curve.  
* \[ \] **Task 4.6:** Create the `SIRModel` subclass utilizing the new API.  
* \[ \] **Task 4.7:** Create the `SISModel` subclass utilizing the new API.  
* \[ \] **Task 4.8:** Create the `SEIRModel` subclass utilizing the new API.

## **Phase 5: Dynamic Interventions Engine**

**Goal:** Allow users to pause, modify, and resume the simulation mid-flight.

* \[ \] **Task 5.1:** Create the `InterventionEngine` class.  
* \[ \] **Task 5.2:** Implement Time-Triggered callbacks (e.g., executing a lockdown function exactly at $t = 30).  
* \[ \] **Task 5.3:** Implement State-Triggered callbacks (e.g., executing a function when the `I` compartment exceeds a threshold).  
* \[ \] **Task 5.4:** Build the Numba pause/resume interrupt layer (allowing Python callbacks to safely modify the underlying flat arrays mid-simulation).  
* \[ \] **Task 5.5:** Implement support for Time-Varying Parameters (e.g., sine wave transmission rates) within the Numba engine.

## **Phase 6: Outputs, Persistence, and Web-UI Prep**

**Goal:** Format simulation results for data analysis and the future Web Dashboard.

* \[ \] **Task 6.1:** Build the `SimulationResult` object to encapsulate output arrays.  
* \[ \] **Task 6.2:** Implement macroscopic time-series extraction (e.g., `result.t()`, `result.S()`).  
* \[ \] **Task 6.3:** Define the SQLite database schema (`experiments`, `time_series`, `interventions_log`).  
* \[ \] **Task 6.4:** Implement `result.to_sqlite()` to map the results directly into local database tables.  
* \[ \] **Task 6.5:** Implement `result.to_csv()` and `result.to_json()` helper methods using `pandas`.  
* \[ \] **Task 6.6:** Build internal data structure representations compatible with a future FastAPI/Pydantic ingestion.

## **Phase 7: Final Documentation & Type Verification**

**Goal:** Ensure production readiness and flawless developer experience.

* \[ \] **Task 7.1:** Write Sphinx/Google-style docstrings for all classes and methods in `core`.  
* \[ \] **Task 7.2:** Write comprehensive docstrings and usage examples for the pre-built `models`.  
* \[ \] **Task 7.3:** Run a final, comprehensive `mypy --strict` pass across the entire codebase; resolve all `Any` and missing type hints.  
* \[ \] **Task 7.4:** Run `ruff check --fix` and `ruff format` to finalize styling.  
* \[ \] **Task 7.5:** Execute the Memory Leak stress test (10,000+ iterations) to ensure stable garbage collection behavior.