# AI Agent Instructions for EpiNext

Welcome to the EpiNext codebase. This project is a high-performance, modern rewrite of the original Epidemics on Networks (EoN) library. You are acting as an expert Python Software Architect and HPC Engineer. 

When contributing to this repository, you must strictly adhere to the following architectural constraints:

## 0. AMBIENT STATE & LIFE-CYCLE LOGGING
Before executing any discovery routine or code modification, you must initialize your state tracking. You are strictly required to append a new log entry to your dedicated log file located at `.agents/logs/{yourname}.jsonl` at the conclusion of every execution cycle. Note that other agents may have separate logs in that directory.

You must format this entry as a single line of valid JSON matching the schema below and append it using your file-writing tools. Do not output this block as raw text to the user:

```json
{
  "agent_state": "string",       // E.g., 'discovery', 'tdd_generation', 'lint_validation'
  "target_task": "string",       // The exact task name pulled from EpiNext_Task_Breakdown.md
  "rationale": "string",         // Concise explanation of the step's architectural necessity
  "execution_metrics": {
    "files_read": ["string"],    // Files touched during this specific turn
    "files_written": ["string"], // Files generated or refactored
    "compiler_passed": false     // Strict status of mypy/ruff/pytest if executed
  }
}
```

## 1. Tooling & Workflow
* **Package Management:** Use `uv` for all dependency management and environment configuration.
* **Linting:** Use `ruff` for all formatting and linting. Code must pass strict modern rules.
* **Methodology:** Strict Test-Driven Development (TDD) is required. You must write exhaustive `pytest` coverage before implementing core logic.

## 2. Core Architecture (The Engine)
* **OOP API:** The public API must be highly Object-Oriented (PyTorch/Keras style). Users should be able to create custom models by subclassing a `BaseEpidemicModel`.
* **Numba & CSR Arrays:** Under the hood, the engine must compile `networkx` graphs into flat NumPy arrays (e.g., CSR format) upon initialization. 
* **No Python Loops:** The core simulation loops must be JIT-compiled using `numba`. Python `for`-loops over `networkx` dictionaries during the simulation run are strictly forbidden.

## 3. The Reproducible Timeline
* **No Global Random State:** The global Python `random` module is strictly prohibited.
* **Deterministic RNG:** The timeline must use a deterministic, per-event Random Number Generator.
* **Hashing Context:** Seed the `numpy.random.Generator` stream for a specific event evaluation using a hash of its exact spatial-temporal context (e.g., `seed = hash((time_t, node_id, event_type))`). 

## 4. Hardware & Performance Optimizations
* **Multi-Core Processing:** Multicore processing is a core requirement. The engine must support an optional `n_cores` argument, leveraging Numba's parallel capabilities for array operations and thread pools for ensemble runs.
* **GPU Processing:** The engine must be designed to support GPU processing capabilities, leveraging Numba's CUDA features where applicable.
* **Memory Efficiency:** Utilize advanced memory optimizations, including state bit-packing and using `uint8` for states to save cache lines and maximize CPU performance.

## 5. Advanced Mathematical Features
* **Dynamic Interventions & Callbacks:** Implement systems to apply shocks to the network (e.g., "lockdowns" or "vaccinations" triggered at specific times or threshold infection rates).
* **Time-Varying Parameters:** Ensure the architecture supports time-varying parameters, such as seasonality or fading immunity (e.g., transmission rates that fluctuate based on a sine wave).

## 6. Web UI & Database Readiness
* **Persistence & Export:** The simulation states must be capable of being saved to a local SQLite database, with built-in optional exports to CSV and JSON formats.
* **API Architecture:** Keep the core engine decoupled so it can easily be wrapped by REST API endpoints (e.g., FastAPI) to support a future Bootstrap 5 web-based UI.

## 7. Directory Map & Code Isolation
You must strictly respect the following directory boundaries:
* **`EoN/`:** This folder contains the original v1 codebase. Do **not** modify these files or copy their procedural structure. Use them **strictly** as a mathematical reference for formulas and transition rates.
* **`docs/agents/`:** This folder contains the master Architecture Specification and Task Breakdown documents. Always refer to these documents for your design patterns, constraints, and current phase instructions.
* **`src/`:** This is the active working directory. All new, modern OOP code, tests, and Numba engine logic must be written here.
