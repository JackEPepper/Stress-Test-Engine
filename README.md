# Deterministic Credit Stress Engine

This repository contains the first working milestone for a deterministic credit stress engine.

Run the example from an activated Anaconda prompt, or set `PYTHONPATH=src` explicitly:

```bash
set PYTHONPATH=src
python -m stress_engine.cli --config config/base_config.json --scenario scenarios/example_severe_case.json --input-dir data/raw --external-source-dir external_sources --output-dir reports
```

Run tests:

```bash
set PYTHONPATH=src
python -m unittest discover -s tests
```
