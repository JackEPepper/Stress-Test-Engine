"""Deterministic JSON-driven credit stress testing engine."""

from .config import load_scenario
from .engine import StressEngine, run_scenario

__all__ = ["StressEngine", "load_scenario", "run_scenario"]

__version__ = "0.1.0"
