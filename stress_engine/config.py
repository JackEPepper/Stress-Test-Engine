"""Scenario loading and light validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import pandas as pd

from .utils import deep_merge, get_metric_cutoffs, hash_json, resolve_path, to_number


_INCLUDE_KEY = "$include"
_SUPPORTED_MODULES = ("CRE", "C&I", "Consumer")
_DEFAULT_MODULE_ORDER = list(_SUPPORTED_MODULES)


def load_scenario(paths: str | Path | Iterable[str | Path]) -> Tuple[Dict[str, Any], Path]:
    """Load one or more JSON files, including manifests, and deep-merge them.

    A file can declare ``"$include": ["relative/file.json", ...]``. Includes
    are merged in listed order, followed by the declaring file, so local values
    override included defaults. Explicitly supplied files are then merged in
    command-line order. Relative data input paths remain anchored to the first
    explicitly supplied scenario file.
    """
    if isinstance(paths, (str, Path)):
        path_list = [Path(paths)]
    else:
        path_list = [Path(path) for path in paths]
    if not path_list:
        raise ValueError("At least one scenario JSON path is required.")

    merged: Dict[str, Any] = {}
    resolved_paths: List[str] = []
    for path in path_list:
        payload, loaded_paths = _load_scenario_file(path.resolve(), stack=())
        merged = deep_merge(merged, payload)
        resolved_paths.extend(str(item) for item in loaded_paths)

    base_dir = path_list[0].resolve().parent
    merged.setdefault("_metadata", {})
    merged["_metadata"].update(
        {
            "scenario_files": resolved_paths,
            "scenario_hash": hash_json({k: v for k, v in merged.items() if k != "_metadata"}),
        }
    )
    validate_scenario(merged)
    return merged, base_dir


def _load_scenario_file(path: Path, stack: Tuple[Path, ...]) -> Tuple[Dict[str, Any], List[Path]]:
    """Load one scenario fragment and recursively expand its relative includes."""
    actual = path.resolve()
    if actual in stack:
        cycle = " -> ".join(str(item) for item in (*stack, actual))
        raise ValueError(f"Scenario include cycle detected: {cycle}")

    with actual.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Scenario file must contain a JSON object: {actual}")

    local_payload = dict(payload)
    raw_includes = local_payload.pop(_INCLUDE_KEY, [])
    if isinstance(raw_includes, (str, Path)):
        includes = [raw_includes]
    elif isinstance(raw_includes, list):
        includes = raw_includes
    else:
        raise ValueError(f"Scenario {_INCLUDE_KEY} must be a string or list: {actual}")

    merged: Dict[str, Any] = {}
    resolved_paths: List[Path] = []
    next_stack = (*stack, actual)
    for include in includes:
        if not isinstance(include, str) or not include.strip():
            raise ValueError(f"Scenario {_INCLUDE_KEY} entries must be nonblank strings: {actual}")
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = actual.parent / include_path
        included, included_paths = _load_scenario_file(include_path, next_stack)
        merged = deep_merge(merged, included)
        resolved_paths.extend(included_paths)

    merged = deep_merge(merged, local_payload)
    resolved_paths.append(actual)
    return merged, resolved_paths


def validate_scenario(scenario: Dict[str, Any]) -> None:
    """Perform lightweight required-section validation after JSON merge."""
    required = ["inputs", "borrower", "tags", "modules"]
    missing = [key for key in required if key not in scenario]
    if missing:
        raise ValueError(f"Scenario is missing required sections: {', '.join(missing)}")
    if "identity" not in scenario["inputs"]:
        raise ValueError("Scenario inputs must include an 'identity' source.")
    borrower = scenario.get("borrower", {})
    for field in ("borrower_id_field", "balance_field"):
        if field not in borrower:
            raise ValueError(f"Scenario borrower section must define '{field}'.")
    from .cecl import validate_cecl_config

    validate_cecl_config(scenario)
    # Import locally to keep scenario loading independent from tagging's input
    # table types while still failing malformed tag flags before any data load.
    from .tagging import normalize_tag_defs

    tag_defs = normalize_tag_defs(scenario.get("tags", {}))
    configured_modules = scenario.get("modules", {})
    for tag in tag_defs:
        if not tag.get("cecl_level", False):
            continue
        cecl_module = str(tag["cecl_module"])
        if cecl_module != "Overlay" and (
            not isinstance(configured_modules, Mapping)
            or cecl_module not in configured_modules
        ):
            raise ValueError(
                f"CECL-level tag '{tag['name']}' references cecl_module "
                f"'{cecl_module}', which is not a configured module or Overlay."
            )
    module_field = borrower.get("module_field", "model_module")
    portfolio_field = borrower.get("portfolio_field", "model_portfolio")
    cecl_portfolio_field = scenario.get("cecl", {}).get(
        "portfolio_field", "cecl_portfolio"
    )
    for tag in tag_defs:
        if tag.get("model_eligible", True):
            continue
        assignments = tag.get("assign", {})
        assigned_module = assignments.get(module_field)
        is_overlay_route = assigned_module == "Overlay"
        disallowed_fields = [
            field
            for field in (
                module_field,
                portfolio_field,
                cecl_portfolio_field,
            )
            if field in assignments and not is_overlay_route
        ]
        if disallowed_fields:
            raise ValueError(
                f"Non-model-eligible tag '{tag['name']}' cannot assign modeled "
                f"routing fields: {', '.join(dict.fromkeys(disallowed_fields))}."
            )
    levels = [str(level) for level in scenario.get("stress_levels", ["S1", "S2"])]
    if not levels or len(levels) != len(set(levels)):
        raise ValueError("Scenario stress_levels must contain unique, nonblank levels.")
    if any(not level.strip() for level in levels):
        raise ValueError("Scenario stress_levels cannot contain blank names.")
    modules = scenario.get("modules", {})
    _validate_module_order(scenario, modules)
    _validate_overlays(scenario.get("overlays", {}))
    cre = modules.get("CRE", {})
    ci = modules.get("C&I", {})
    nested_cutoff_paths = []
    cre_tests = cre.get("tests", {}) if isinstance(cre, dict) else {}
    if isinstance(cre_tests, dict):
        nested_cutoff_paths.extend(
            f"modules.CRE.tests.{test_name}.cutoffs"
            for test_name, test_config in cre_tests.items()
            if isinstance(test_config, dict) and "cutoffs" in test_config
        )
    if isinstance(ci, dict) and "cutoffs" in ci:
        nested_cutoff_paths.append("modules.C&I.cutoffs")
    if nested_cutoff_paths:
        raise ValueError(
            "Migration cutoffs must be defined only in the master scenario's "
            f"top-level 'cutoffs' object; remove: {', '.join(nested_cutoff_paths)}."
        )
    if cre and cre.get("enabled", True):
        cutoff = scenario.get("run", {}).get("cutoff_date")
        if cutoff is None or pd.isna(pd.to_datetime(cutoff, errors="coerce")):
            raise ValueError("An enabled CRE module requires a valid run.cutoff_date.")
        get_metric_cutoffs(scenario, "dscr")
        get_metric_cutoffs(scenario, "ltv")
    if ci and ci.get("enabled", True):
        get_metric_cutoffs(scenario, "fccr")
    if scenario.get("targeted_stress"):
        from .targeted import validate_targeted_config

        validate_targeted_config(scenario)


def _validate_module_order(
    scenario: Dict[str, Any],
    modules: Mapping[str, Any],
) -> None:
    """Require an executable order that cannot omit an enabled module."""
    if not isinstance(modules, Mapping):
        raise ValueError("Scenario modules must be a JSON object.")

    unsupported_configs = [module for module in modules if module not in _SUPPORTED_MODULES]
    if unsupported_configs:
        names = ", ".join(repr(module) for module in unsupported_configs)
        raise ValueError(f"Scenario modules contains unsupported module configurations: {names}.")

    malformed_configs = [
        module for module, config in modules.items() if not isinstance(config, Mapping)
    ]
    if malformed_configs:
        raise ValueError(
            "Every configured module must be a JSON object; invalid: "
            f"{', '.join(str(module) for module in malformed_configs)}."
        )

    empty_configs = [module for module, config in modules.items() if not config]
    if empty_configs:
        raise ValueError(
            "Every configured module must be a nonempty JSON object; invalid: "
            f"{', '.join(str(module) for module in empty_configs)}."
        )

    invalid_enabled = [
        module
        for module, config in modules.items()
        if "enabled" in config and not isinstance(config["enabled"], bool)
    ]
    if invalid_enabled:
        raise ValueError(
            "Configured module 'enabled' values must be JSON booleans; invalid: "
            f"{', '.join(str(module) for module in invalid_enabled)}."
        )

    module_order = scenario.get("module_order", _DEFAULT_MODULE_ORDER)
    if not isinstance(module_order, list) or not module_order:
        raise ValueError("Scenario module_order must be a nonempty JSON list.")

    unsupported = [
        module
        for module in module_order
        if not isinstance(module, str) or module not in _SUPPORTED_MODULES
    ]
    if unsupported:
        names = ", ".join(repr(module) for module in unsupported)
        supported = ", ".join(_SUPPORTED_MODULES)
        raise ValueError(
            f"Scenario module_order contains unsupported modules: {names}. "
            f"Supported modules: {supported}."
        )

    duplicates = list(
        dict.fromkeys(
            module
            for module in module_order
            if module_order.count(module) > 1
        )
    )
    if duplicates:
        raise ValueError(
            "Scenario module_order must contain unique module names; duplicates: "
            f"{', '.join(duplicates)}."
        )

    enabled_modules = [
        module
        for module in _SUPPORTED_MODULES
        if modules.get(module) and modules[module].get("enabled", True)
    ]
    missing = [module for module in enabled_modules if module not in module_order]
    if missing:
        raise ValueError(
            "Scenario module_order must include every enabled configured module "
            f"exactly once; missing: {', '.join(missing)}."
        )


def _validate_overlays(overlays: Any) -> None:
    """Require every enabled overlay to have executable source definitions."""
    if not overlays:
        return
    if not isinstance(overlays, Mapping):
        raise ValueError("Scenario overlays must be a JSON object keyed by portfolio name.")
    for portfolio, config in overlays.items():
        if not isinstance(config, Mapping) or not config:
            raise ValueError(
                f"Overlay '{portfolio}' must be a nonempty JSON object."
            )
        enabled = config.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"Overlay '{portfolio}' enabled must be a JSON boolean.")
        if not enabled:
            continue
        sources = config.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(
                f"Enabled overlay '{portfolio}' must define a nonempty sources list."
            )
        total_weight = 0.0
        for index, source in enumerate(sources):
            if not isinstance(source, Mapping) or not str(source.get("name", "")).strip():
                raise ValueError(
                    f"Overlay '{portfolio}' sources[{index}] must be an object "
                    "with a nonblank name."
                )
            raw_weight = source.get("weight", 1.0)
            weight = to_number(raw_weight)
            if (
                isinstance(raw_weight, bool)
                or not math.isfinite(weight)
                or weight < 0
            ):
                raise ValueError(
                    f"Overlay '{portfolio}' sources[{index}].weight must be "
                    "a nonnegative finite number."
                )
            total_weight += weight
        if total_weight <= 0:
            raise ValueError(
                f"Enabled overlay '{portfolio}' must have at least one "
                "positive source weight."
            )


def output_dir_for(scenario: Dict[str, Any], base_dir: Path, override: str | Path | None = None) -> Path:
    """Resolve the scenario or CLI output directory relative to scenario JSON."""
    if override:
        return resolve_path(override, base_dir)
    outputs = scenario.get("outputs", {})
    directory = outputs.get("directory", "outputs/latest")
    return resolve_path(directory, base_dir)
