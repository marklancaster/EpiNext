"""EpiNext models package.

Exports the OOP API layer for defining and running epidemic models.

Public API
----------
BaseEpidemicModel
    Abstract base class for all custom epidemic models.
SimulationResult
    Encapsulates simulation output with time-series extraction methods.
SIRModel
    Standard Susceptible-Infectious-Removed model.
SISModel
    Standard Susceptible-Infectious-Susceptible model.
SEIRModel
    Standard Susceptible-Exposed-Infectious-Removed model.
"""

from __future__ import annotations

from EpiNext.models.base import BaseEpidemicModel, SimulationResult
from EpiNext.models.standard import SEIRModel, SIRModel, SISModel

__all__ = [
    "BaseEpidemicModel",
    "SimulationResult",
    "SIRModel",
    "SISModel",
    "SEIRModel",
]
