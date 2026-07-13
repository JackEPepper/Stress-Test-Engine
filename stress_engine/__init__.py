"""Deterministic JSON-driven credit stress testing engine."""

from .batch import expand_batch_scenarios, run_batch_scenarios
from .config import load_scenario
from .engine import StressEngine, run_scenario

__all__ = ["StressEngine", "expand_batch_scenarios", "load_scenario", "run_batch_scenarios", "run_scenario"]

__version__ = "0.2.0"
