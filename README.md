# EpiNext: High-Performance Epidemics on Networks

A blazing-fast, modern rewrite of the EoN (Epidemics on Networks) library. EpiNext transforms the classic codebase into an extensible, object-oriented framework designed for high-performance stochastic epidemic simulations.

## Key Features

* **Blazing-Fast Execution:** Leverages Numba to JIT-compile `networkx` graphs into flat CSR arrays for pure C-level speeds.
* **Modern OOP Architecture:** Features a PyTorch-style API where researchers can easily craft custom disease models by subclassing a base engine (`BaseEpidemicModel`).
* **Reproducible Timeline:** Completely abandons the global Python random state. Uses a deterministic, per-event Random Number Generator seeded by a hash of its exact spatial-temporal context (e.g., `hash((time_t, node_id, event_type))`) to guarantee perfect reproducibility.
* **HPC Ready:** Engineered for rapid iteration with built-in multi-core execution and GPU processing capabilities.

---

## Acknowledgments & History

This project is a high-performance, modern rewrite of the original Epidemics on Networks (EoN) library created by Joel C Miller. While the core engine has been completely rebuilt from the ground up for GPU/multi-core processing, the underlying mathematical models were heavily inspired by the original EoN implementation.

The legacy source code accompanied the book:
**[*Mathematics of Epidemics on Networks*](http://www.springer.com/book/9783319508047) by Kiss, Miller, and Simon (Springer, 2017).** For the original v1.x Python software and legacy documentation, please visit the [Epidemics on Networks ReadTheDocs](http://epidemicsonnetworks.readthedocs.io/en/latest/).

---

## Installation & Development

This project uses `uv` for dependency management and environment configuration.

```bash
# Clone the repository
git clone [git@github.com:marklancaster/EpiNext.git](git@github.com:marklancaster/EpiNext.git)
cd EpiNext

# Sync dependencies and setup the environment
uv sync

# Run the test suite
uv run pytest -v
```

## Issues & Contributions

* **Bugs & Feature Requests:** Please use the GitHub Issues tab to report bugs or request new features for the EoNv2 engine.
* **Legacy Book Errata:** For corrections or to report an error to the content in the published Springer book, please see the legacy `errata.md` file from the original repository.
* **Contributing:** We welcome new contributors! EpiNext follows strict Test-Driven Development (TDD). Please ensure all pull requests are accompanied by exhaustive `pytest` coverage and adhere to modern `ruff` formatting rules.