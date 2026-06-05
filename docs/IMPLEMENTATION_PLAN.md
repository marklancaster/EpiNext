# EpiNext Implementation Plan

**Project:** Modernization of Epidemics on Networks (EpiNext)  
**Document Version:** 1.0  
**Last Updated:** June 5, 2026  
**Status:** Ready for Phase 1 Verification

---

## Table of Contents

1. [Overview](#overview)
2. [Current Status](#current-status)
3. [Phase-by-Phase Breakdown](#phase-by-phase-breakdown)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Key Constraints](#key-constraints)
6. [FAQ & Clarifications](#faq--clarifications)

---

## Overview

EpiNext is a complete rewrite of the legacy Epidemics on Networks library, transforming it from a procedural codebase into a modern, high-performance OOP framework. The implementation follows a **7-phase iterative approach**, with each phase building upon the previous one.

**Key Goals:**
- ✅ Modern object-oriented API (PyTorch/Keras style)
- ✅ Deterministic, reproducible timelines (no global RNG state)
- ✅ Blazing-fast execution via Numba JIT compilation and GPU support
- ✅ Dynamic interventions and callbacks for mid-simulation modifications
- ✅ SQLite persistence and JSON export for web UI integration

---

## Current Status

### Project Structure ✅ ~90% Complete

| Component | Status | Notes |
|-----------|--------|-------|
| Project initialization (`uv`) | ✅ Complete | Modern Python environment |
| `pyproject.toml` configuration | ✅ Complete | Strict `ruff` + `mypy --strict` |
| Core dependencies | ✅ Complete | numpy, numba, networkx, pandas, scipy |
| Dev dependencies | ✅ Complete | pytest, hypothesis, ruff, mypy |
| Directory scaffolding | ✅ Complete | `src/EpiNext/{core,models,interventions,utils}` |
| Legacy reference (`EoN/`) | ✅ Complete | Do not modify; use for math reference only |
| Architecture documentation | ✅ Complete | See `docs/agents/EpiNext_Architecture_Specification.md` |

### Next Steps
- [ ] Phase 1: Final pytest configuration verification
- [ ] Phase 2: Begin graph compilation implementation

---

## Phase-by-Phase Breakdown

### Phase 1: Project Scaffolding & CI/CD
**Duration:** 1-2 hours  
**Status:** ~90% Complete

This phase establishes the development foundation.

#### Tasks

- [ ] **Task 1.1:** Verify `uv` environment setup and `pyproject.toml` completeness
- [ ] **Task 1.2:** Confirm all required dependencies installed (numpy, numba, networkx, pandas, scipy)
- [ ] **Task 1.3:** Verify development dependencies (pytest, hypothesis, ruff, mypy)
- [ ] **Task 1.4:** Check `ruff` configuration: line-length=88, import sorting, PEP8
- [ ] **Task 1.5:** Check `mypy` strict mode enabled
- [ ] **Task 1.6:** Verify directory structure:
  - `src/EpiNext/core/`
  - `src/EpiNext/models/`
  - `src/EpiNext/interventions/`
  - `src/EpiNext/utils/`
  - `tests/`
- [ ] **Task 1.7:** Configure pytest and run empty test suite

#### Expected Output
✅ A production-ready Python development environment with passing CI/CD pipeline

---

### Phase 2: Graph Compilation & Memory Management
**Duration:** 4-6 hours  
**Difficulty:** Medium

Abstracts NetworkX graphs into high-performance, cache-friendly flat arrays (CSR format).

#### Key Concepts

- **Compressed Sparse Row (CSR):** A memory-efficient representation of graphs
  - `indptr` (int32): Pointers to adjacency lists
  - `indices` (int32): Target node indices
  - `edge_weights` (float32): Transmission rates
  - `node_states` (uint8): Compartment states (S=0, E=1, I=2, R=3)

#### Tasks

- [ ] **Task 2.1:** Write comprehensive test suite (`test_compiler.py`)
  - Test simple path/complete/random graphs
  - Test directed vs. undirected graphs
  - Test node/edge attributes and weights
  - Test disconnected components and isolated nodes

- [ ] **Task 2.2:** Implement `CompiledGraph` dataclass
  - Type-hinted NumPy arrays
  - Metadata fields (n_nodes, n_edges, state_names)

- [ ] **Task 2.3:** Implement graph compiler in `src/core/compiler.py`
  - NetworkX → CSR conversion
  - Dynamic weight extraction
  - Handle directed/undirected graphs

- [ ] **Task 2.4:** Optimize memory usage
  - Use `uint8` for node states (cache-optimal)
  - Use `float32` for weights (minimize footprint)

- [ ] **Task 2.5:** Add graph caching mechanism
  - Hash-based memoization to avoid recompiling identical graphs

#### Expected Output
✅ A fully tested graph compiler that converts NetworkX graphs into optimized flat arrays

---

### Phase 3: Hash-Based RNG & Numba Simulation Engine
**Duration:** 6-8 hours  
**Difficulty:** High (Core Engine)

Implements the deterministic, reproducible event-driven simulation loop.

#### Key Concepts

- **Deterministic Seeding:** `event_seed = hash((base_seed, node_u, node_v, time_t, event_type))`
- **Gillespie Algorithm:** Continuous-time stochastic simulation via exponential time steps
- **Memory Pool:** Pre-allocate output buffers to eliminate garbage collection in hot loops
- **No Global State:** Strict prohibition on Python's `random` module

#### Tasks (Test-First Development Required)

- [ ] **Task 3.1:** Write "Butterfly Effect" isolation test (`test_rng_isolation.py`)
  - Simulate SIR on 1,000-node graph; log Node 500's timeline
  - Re-run with disconnected 10-node component added
  - **Critical Assertion:** Node 500's timeline is bit-for-bit identical (proves no global RNG contamination)

- [ ] **Task 3.2:** Implement localized PRNG in `src/core/rng.py`
  - Hash-based deterministic seeding
  - Thread-safe random initialization
  - Zero dependency on Python's `random` module

- [ ] **Task 3.3:** Implement Numba event loop in `src/core/simulator.py`
  - Continuous-time Gillespie Direct Method
  - Pre-allocated event history buffers (Memory Pool pattern)
  - Zero allocations in hot loop
  - Handle state transitions and rate updates

- [ ] **Task 3.4:** Add multi-core support
  - Use `numba.prange` for bulk state updates
  - Support `n_cores` parameter for parallel ensemble runs

- [ ] **Task 3.5:** Add GPU acceleration framework
  - CUDA hooks (`@numba.cuda.jit`)
  - Apple Metal abstraction layer (hardware abstraction design)
  - Device memory transfer optimization

#### Expected Output
✅ A blazing-fast Numba-compiled simulation engine with proven RNG isolation and optional GPU support

---

### Phase 4: PyTorch-Style OOP Wrapper & Standard Models
**Duration:** 5-7 hours  
**Difficulty:** Medium

Builds the user-friendly API abstracting raw Numba internals.

#### Key Concepts

- **API Philosophy:** Hide complexity behind clean Python inheritance
- **State Translation:** Map string labels ('S', 'I') to uint8 integers (0, 1)
- **Seamless Integration:** Combine CompiledGraph + Numba simulator + OOP wrapper

#### Tasks (Test-First Development Required)

- [ ] **Task 4.1:** Write Mean-Field Limit test (`test_ode_limit.py`)
  - Simulate SIR on fully-connected 10,000-node graph
  - Compare macroscopic curve to numerical ODE solution (scipy.integrate.odeint)
  - Assert L2 norm error < 0.05 (within stochastic margin)

- [ ] **Task 4.2:** Implement `BaseEpidemicModel` in `src/core/model.py`
  - `add_compartments(labels)`: Map string states to uint8
  - `add_spontaneous_transition(source, target, rate)`: Internal transitions
  - `add_induced_transition(source, target, catalyst, rate)`: Edge-based transmission
  - `set_initial_conditions()`: Configure starting state
  - `run(t_max, track_history)`: Execute simulation → SimulationResult

- [ ] **Task 4.3:** Implement standard models as subclasses
  - `SIRModel`: Susceptible → Infected → Recovered
  - `SISModel`: Susceptible → Infected → Susceptible
  - `SEIRModel`: Susceptible → Exposed → Infected → Recovered

- [ ] **Task 4.4:** Add performance features
  - Optional `n_cores=-1` for multi-core ensemble runs
  - Optional `use_gpu=True` for GPU acceleration
  - Optional `seed` parameter for reproducibility

#### Expected Output
✅ User-friendly OOP API matching documented examples; passing ODE comparison tests

---

### Phase 5: Dynamic Interventions Engine
**Duration:** 3-4 hours  
**Difficulty:** Medium

Enables mid-simulation modifications (lockdowns, vaccinations, parameter changes).

#### Key Concepts

- **Time-Triggered Interventions:** Execute callbacks at specific timestamps
- **State-Triggered Interventions:** Execute when simulation reaches thresholds
- **Time-Varying Parameters:** Support functions like `tau(t) = 0.5 * (1 + sin(t))`

#### Tasks

- [ ] **Task 5.1:** Implement `InterventionEngine` in `src/interventions/engine.py`
  - Time-triggered callback system
  - State-triggered condition checking
  - Time-varying parameter evaluation

- [ ] **Task 5.2:** Implement pause/resume mechanism
  - Safely interrupt Numba event loop at intervention points
  - Allow Python callbacks to modify node_states
  - Resume with modified state

- [ ] **Task 5.3:** Write comprehensive tests
  - Vaccination reduces susceptible population
  - Lockdown reduces transmission rates
  - Interventions execute at correct times

#### Expected Output
✅ Flexible intervention system supporting dynamic policy changes during simulation

---

### Phase 6: Outputs, Persistence & Web-UI Preparation
**Duration:** 4-5 hours  
**Difficulty:** Low

Formats results for analysis and future web dashboard integration.

#### Key Concepts

- **Multi-Format Export:** CSV, JSON, SQLite
- **API-Ready Structure:** Pydantic models compatible with FastAPI
- **Database Schema:** Extensible SQLite for experiment tracking

#### Tasks

- [ ] **Task 6.1:** Implement `SimulationResult` in `src/core/result.py`
  - Encapsulate time series arrays
  - Provide accessor methods: `t()`, `S()`, `I()`, `R()`

- [ ] **Task 6.2:** Define SQLite schema
  - `experiments` table: run metadata
  - `time_series` table: macroscopic dynamics
  - `interventions_log` table: callback history

- [ ] **Task 6.3:** Implement `result.to_sqlite(db_path, run_name)`
  - Map arrays to database tables
  - Handle parameter JSON serialization

- [ ] **Task 6.4:** Implement `result.to_csv()` using pandas
  
- [ ] **Task 6.5:** Implement `result.to_json()` with Pydantic serialization

#### Expected Output
✅ Multi-format data export with SQLite persistence for web backend

---

### Phase 7: Final Documentation & Type Verification
**Duration:** 2-3 hours  
**Difficulty:** Low

Ensures production-readiness and excellent developer experience.

#### Tasks

- [ ] **Task 7.1:** Write NumPy-style docstrings for all classes/methods
  - Include parameter descriptions, return types, examples
  - Add mathematical basis where relevant

- [ ] **Task 7.2:** Achieve 100% type coverage
  - Run `mypy --strict` across codebase
  - Eliminate all `Any` type annotations
  - Validate generic types: `list[int]`, `dict[str, float]`

- [ ] **Task 7.3:** Run final code formatting
  - `ruff check --fix` for style violations
  - `ruff format` for automatic formatting
  - Verify 88-character line length

- [ ] **Task 7.4:** Run memory stress test
  - 10,000+ iterations on small graph
  - Assert stable memory consumption
  - Profile Numba compilation times

#### Expected Output
✅ Production-ready codebase with flawless developer experience

---

## Contributing Guidelines

### Development Workflow

1. **Always write tests first (TDD)** before implementing features
2. **Run linting before committing:** `ruff check --fix && ruff format`
3. **Run type checking:** `mypy` should pass with zero errors
4. **Run the test suite:** `pytest -v` should pass all tests
5. **Keep the CSR graph logic in `core/`**, not scattered in models
6. **Document why, not just what** — focus on mathematical intent

### Code Style

- **Line length:** 88 characters (enforced by ruff)
- **Type hints:** Mandatory for all public APIs
- **Docstrings:** NumPy style for all classes/methods
- **No global state:** Forbidden to use `random` module or global variables

### Testing Standards

- **Unit tests:** Fast, deterministic, focused on single components
- **Integration tests:** Verify graph compilation → simulation → results
- **Property-based tests:** Use `hypothesis` for algorithmic invariants
- **Reproducibility:** Always include seed in test simulations

---

## Key Constraints

### 🚫 Absolute Prohibitions

1. **No Python `random` module** — Use deterministic hash-based seeding only
2. **No Python loops in execution** — All simulation loops must be Numba-compiled (`@numba.njit`)
3. **No allocations in hot loops** — Pre-allocate buffers (Memory Pool pattern)
4. **No modification of `EoN/` directory** — Reference only, do not alter

### ✅ Design Mandates

1. **Object-Oriented API:** Users inherit from `BaseEpidemicModel`
2. **Numba + CSR compilation:** All graphs → flat arrays before execution
3. **Type Safety:** Full type hints with `mypy --strict`
4. **Reproducibility:** Hash-based RNG ensures local perturbations don't affect distant nodes
5. **GPU-Ready:** Architecture supports CUDA + Metal acceleration

---

## FAQ & Clarifications

### Q: What if the legacy `EoN/` code has formulas I need?

**A:** Copy the mathematical formula into comments/docstrings. Never copy procedural code. The entire point of EpiNext is to replace the legacy procedural structure.

### Q: Should I optimize prematurely?

**A:** No. Write correct code first, then profile. Numba compilation will handle most performance gains automatically. Only micro-optimize after profiling identifies bottlenecks.

### Q: How do I handle custom compartment states beyond SEIR?

**A:** Subclass `BaseEpidemicModel` and override `define_transitions()`. The framework automatically maps your custom states to uint8 internally. Example:

```python
class CustomModel(BaseEpidemicModel):
    def define_transitions(self):
        self.add_compartments(['S', 'E1', 'E2', 'I', 'R'])
        self.add_spontaneous_transition('E1', 'E2', rate=1.0)
        # ... add more transitions
```

### Q: How do I add a time-varying parameter?

**A:** Pass a callable instead of a float. The engine evaluates it at each time step:

```python
model = SIRModel(
    graph=G,
    params={
        'tau': lambda t: 0.5 * (1 + np.sin(t)),  # Seasonality
        'gamma': 0.1  # Static parameter
    }
)
```

### Q: When should I use GPU acceleration?

**A:** When running massive ensemble simulations (100+ replicates) or very large graphs (100K+ nodes). GPU setup has overhead; benefits emerge at scale.

---

## Next Steps

**For New Contributors:**
1. Read the full [Architecture Specification](./agents/EpiNext_Architecture_Specification.md)
2. Start with Phase 1 verification (should take <1 hour)
3. Wait for lead architect's sign-off before Phase 2
4. Follow TDD discipline strictly

**For Lead Architect:**
- [ ] Review this plan for clarity and completeness
- [ ] Approve GPU strategy (CUDA-only vs. CUDA + Metal)
- [ ] Confirm phase sequencing
- [ ] Begin Phase 1 verification

---

**Document Status:** Ready for wiki publication  
**Last Reviewed:** June 5, 2026  
**Next Review:** Upon Phase 1 completion
