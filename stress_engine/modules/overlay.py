"""Portfolio-level migration overlays for portfolios without financials."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from ..tagging import evaluate_conditions
from ..utils import as_list, get_levels, pct, stable_name, to_number
from ..exceptions import record_exception


BUCKETS = ["Pass", "Special Mention", "Substandard"]
OVERLAY_OUTPUT_BUCKETS = [*BUCKETS, "Unknown"]


def apply_overlays(
    bucket_summary: pd.DataFrame,
    borrowers: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply migration-pattern overlays for portfolios without financials.

    Called by `reporting.build_reports` after modeled bucket summaries are
    built. It replaces overlay portfolio rows (EF/BCC in the example) with
    balances derived from source portfolio migration growth.
    """
    exceptions = exceptions if exceptions is not None else []
    overlays = scenario.get("overlays", {})
    if not overlays:
        return bucket_summary, pd.DataFrame()
    if isinstance(overlays, list):
        overlay_items = {item["portfolio"]: item for item in overlays}
    else:
        overlay_items = overlays
    portfolio_field = scenario["borrower"].get("portfolio_field", "portfolio")
    balance_field = scenario["borrower"]["balance_field"]
    levels = get_levels(scenario)
    summary = bucket_summary.copy()
    overlay_rows: List[Dict[str, Any]] = []

    for portfolio, config in overlay_items.items():
        if not config or not config.get("enabled", True):
            continue
        source_specs = _source_specs(config)
        if not source_specs:
            continue
        base_rows = borrowers[borrowers[portfolio_field] == portfolio]
        total_balance = float(pd.to_numeric(base_rows[balance_field], errors="coerce").sum())
        if total_balance <= 0:
            continue
        base_ratios = _portfolio_base_ratios(base_rows, balance_field)
        source_base = _source_ratios(summary, borrowers, source_specs, scenario, "Base", config, exceptions, portfolio)
        replacement: List[Dict[str, Any]] = []
        for level in levels:
            source_level = _source_ratios(summary, borrowers, source_specs, scenario, level, config, exceptions, portfolio)
            # Overlay ratios scale the overlay portfolio's own base SM/Sub
            # ratios by observed growth in the source portfolio set.
            sm_ratio = _grown_ratio(
                base_ratios["Special Mention"],
                source_base["Special Mention"],
                source_level["Special Mention"],
                config,
                exceptions,
                portfolio,
                level,
                "Special Mention",
                source_base["source_description"],
            )
            sub_ratio = _grown_ratio(
                base_ratios["Substandard"],
                source_base["Substandard"],
                source_level["Substandard"],
                config,
                exceptions,
                portfolio,
                level,
                "Substandard",
                source_base["source_description"],
            )
            unknown_ratio = base_ratios["Unknown"]
            sm_ratio, sub_ratio = _cap_ratios(sm_ratio, sub_ratio, max_total=max(1.0 - unknown_ratio, 0.0))
            ratios = {
                "Pass": max(1 - unknown_ratio - sm_ratio - sub_ratio, 0.0),
                "Special Mention": sm_ratio,
                "Substandard": sub_ratio,
                "Unknown": unknown_ratio,
            }
            for bucket in OVERLAY_OUTPUT_BUCKETS:
                replacement.append(
                    {
                        "portfolio": portfolio,
                        "stress_level": level,
                        "bucket": bucket,
                        "balance": total_balance * ratios[bucket],
                        "borrower_count": np.nan,
                        "source": "overlay",
                    }
                )
            overlay_rows.append(
                {
                    "portfolio": portfolio,
                    "stress_level": level,
                    "source_portfolios": source_base["source_names"],
                    "source_weights": source_base["source_weights"],
                    "source_selection": source_base["source_selection"],
                    "base_special_mention_ratio": base_ratios["Special Mention"],
                    "base_substandard_ratio": base_ratios["Substandard"],
                    "base_unknown_ratio": base_ratios["Unknown"],
                    "weighted_source_base_special_mention_ratio": source_base["Special Mention"],
                    "weighted_source_base_substandard_ratio": source_base["Substandard"],
                    "weighted_source_stressed_special_mention_ratio": source_level["Special Mention"],
                    "weighted_source_stressed_substandard_ratio": source_level["Substandard"],
                    "stressed_special_mention_ratio": sm_ratio,
                    "stressed_substandard_ratio": sub_ratio,
                    "stressed_unknown_ratio": unknown_ratio,
                }
            )

        summary = summary[summary["portfolio"] != portfolio]
        base_replacement = []
        for bucket in OVERLAY_OUTPUT_BUCKETS:
            base_replacement.append(
                {
                    "portfolio": portfolio,
                    "stress_level": "Base",
                    "bucket": bucket,
                    "balance": total_balance * base_ratios[bucket],
                    "borrower_count": int((base_rows["base_bucket"] == bucket).sum()) if "base_bucket" in base_rows.columns else np.nan,
                    "source": "overlay_base",
                }
            )
        summary = pd.concat([summary, pd.DataFrame(base_replacement + replacement)], ignore_index=True)
    return summary, pd.DataFrame(overlay_rows)


def _portfolio_base_ratios(rows: pd.DataFrame, balance_field: str) -> Dict[str, float]:
    """Calculate base bucket balance ratios for one overlay portfolio."""
    total = float(pd.to_numeric(rows[balance_field], errors="coerce").sum())
    ratios = {bucket: 0.0 for bucket in OVERLAY_OUTPUT_BUCKETS}
    if total <= 0 or "base_bucket" not in rows.columns:
        ratios["Pass"] = 1.0
        return ratios
    for bucket in OVERLAY_OUTPUT_BUCKETS:
        ratios[bucket] = pct(pd.to_numeric(rows.loc[rows["base_bucket"] == bucket, balance_field], errors="coerce").sum(), total)
    return ratios


def _source_specs(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Normalize legacy and weighted/tagged source configuration.

    Legacy `source_portfolios: ["CRE", "C&I"]` remains balance-weighted from
    migration summary rows. New `sources` specs can define `tags` plus explicit
    `weight` values, which are calculated from borrower-level stressed results.
    """
    raw = config.get("sources", config.get("source_portfolios", config.get("basis_portfolios", [])))
    if isinstance(raw, Mapping):
        raw_items = []
        for name, spec in raw.items():
            if isinstance(spec, Mapping):
                item = dict(spec)
                item.setdefault("name", name)
            else:
                item = {"name": name, "portfolio": name, "weight": spec}
            raw_items.append(item)
    else:
        raw_items = as_list(raw)

    source_weights = config.get("source_weights", {})
    source_tags = config.get("source_tags", config.get("source_portfolio_tags", {}))
    specs: List[Dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, Mapping):
            spec = dict(item)
            name = spec.get("name", spec.get("label", spec.get("portfolio", spec.get("source"))))
            if name is None and spec.get("tags"):
                name = "+".join(map(str, as_list(spec.get("tags"))))
            spec["name"] = str(name) if name is not None else "source"
        else:
            spec = {"name": str(item), "portfolio": item, "_auto_portfolio": True}

        lookup_name = str(spec.get("name", spec.get("portfolio", "")))
        if "weight" not in spec and isinstance(source_weights, Mapping) and lookup_name in source_weights:
            spec["weight"] = source_weights[lookup_name]
        if not spec.get("tags") and isinstance(source_tags, Mapping) and lookup_name in source_tags:
            spec["tags"] = source_tags[lookup_name]
            if spec.get("_auto_portfolio"):
                spec.pop("portfolio", None)
        spec.pop("_auto_portfolio", None)
        specs.append(spec)
    return specs


def _source_ratios(
    summary: pd.DataFrame,
    borrowers: pd.DataFrame,
    source_specs: List[Mapping[str, Any]],
    scenario: Mapping[str, Any],
    level: str,
    config: Mapping[str, Any],
    exceptions: List[Dict[str, Any]],
    overlay_portfolio: Any,
) -> Dict[str, Any]:
    """Calculate source bucket ratios for one stress level.

    If no source spec uses tags, weights, or conditions, this preserves the
    original balance-weighted summary behavior. Otherwise each source spec is
    evaluated independently and combined using normalized configured weights.
    """
    if not _uses_weighted_source_specs(source_specs, config):
        portfolios = [spec.get("portfolio", spec.get("name")) for spec in source_specs]
        ratios = _summary_source_ratios(summary, portfolios, level)
        ratios.update(_source_audit_fields(source_specs, use_weights=False))
        return ratios
    ratios = _weighted_source_ratios(borrowers, source_specs, scenario, level, exceptions, overlay_portfolio)
    ratios.update(_source_audit_fields(source_specs, use_weights=True))
    return ratios


def _uses_weighted_source_specs(source_specs: List[Mapping[str, Any]], config: Mapping[str, Any]) -> bool:
    """Return whether overlay source ratios need row-level weighted logic."""
    if "sources" in config or "source_weights" in config or "source_tags" in config or "source_portfolio_tags" in config:
        return True
    for spec in source_specs:
        if any(key in spec for key in ("weight", "tags", "tag", "conditions", "include", "exclude")):
            return True
    return False


def _summary_source_ratios(summary: pd.DataFrame, portfolios: List[Any], level: str) -> Dict[str, float]:
    """Calculate legacy balance-weighted source portfolio ratios."""
    rows = summary[(summary["portfolio"].isin(portfolios)) & (summary["stress_level"] == level)]
    total = float(pd.to_numeric(rows["balance"], errors="coerce").sum())
    return {
        bucket: pct(pd.to_numeric(rows.loc[rows["bucket"] == bucket, "balance"], errors="coerce").sum(), total)
        for bucket in BUCKETS
    }


def _weighted_source_ratios(
    borrowers: pd.DataFrame,
    source_specs: List[Mapping[str, Any]],
    scenario: Mapping[str, Any],
    level: str,
    exceptions: List[Dict[str, Any]],
    overlay_portfolio: Any,
) -> Dict[str, float]:
    """Combine per-source migration ratios using configured weights."""
    balance_field = scenario["borrower"]["balance_field"]
    weighted = {bucket: 0.0 for bucket in BUCKETS}
    total_weight = 0.0
    for spec in source_specs:
        weight = to_number(spec.get("weight", 1.0), 1.0)
        if weight <= 0:
            continue
        rows = borrowers[_source_mask(borrowers, spec, scenario)]
        if rows.empty:
            record_exception(
                exceptions,
                "WARNING",
                "overlay",
                "OVERLAY_SOURCE_POPULATION_EMPTY",
                "Overlay source population matched no borrowers and its configured weight was excluded.",
                portfolio=overlay_portfolio,
                stress_level=level,
                source=str(spec.get("name", spec.get("portfolio", "source"))),
                details=_source_selection_text(spec),
            )
            continue
        ratios = _bucket_ratios_from_rows(rows, balance_field, level)
        if all(pd.isna(ratios[bucket]) for bucket in BUCKETS):
            record_exception(
                exceptions,
                "WARNING",
                "overlay",
                "OVERLAY_SOURCE_RATIO_UNAVAILABLE",
                "Overlay source ratios were unavailable and its configured weight was excluded.",
                portfolio=overlay_portfolio,
                stress_level=level,
                source=str(spec.get("name", spec.get("portfolio", "source"))),
                details=_source_selection_text(spec),
            )
            continue
        for bucket in BUCKETS:
            weighted[bucket] += weight * to_number(ratios[bucket], 0.0)
        total_weight += weight
    if total_weight <= 0:
        return {bucket: np.nan for bucket in BUCKETS}
    return {bucket: weighted[bucket] / total_weight for bucket in BUCKETS}


def _source_mask(df: pd.DataFrame, spec: Mapping[str, Any], scenario: Mapping[str, Any]) -> pd.Series:
    """Select one configured source population from borrower-level results."""
    mask = pd.Series(True, index=df.index)
    portfolio = spec.get("portfolio")
    portfolio_field = scenario["borrower"].get("portfolio_field", "portfolio")
    if portfolio is not None and portfolio_field in df.columns:
        mask &= df[portfolio_field] == portfolio

    tags = as_list(spec.get("tags", spec.get("tag")))
    if tags:
        tag_masks = []
        for tag in tags:
            column = f"tag_{stable_name(tag)}"
            tag_masks.append(df[column].fillna(False).astype(bool) if column in df.columns else pd.Series(False, index=df.index))
        if str(spec.get("tag_match", "all")).lower() == "any":
            tag_mask = pd.Series(False, index=df.index)
            for item in tag_masks:
                tag_mask |= item
        else:
            tag_mask = pd.Series(True, index=df.index)
            for item in tag_masks:
                tag_mask &= item
        mask &= tag_mask
        # Keep tag-based source populations aligned with module priority. This
        # excludes a CRE-priority overlap from a source labeled C&I.
        if spec.get("respect_module_priority", True) and "primary_module" in df.columns:
            source_module = _recognized_source_module(spec.get("module", spec.get("name")))
            if source_module:
                mask &= df["primary_module"].astype(str).map(_normalize_module_name) == source_module

    exclude_tags = as_list(spec.get("exclude_tags", spec.get("exclude_tag")))
    for tag in exclude_tags:
        column = f"tag_{stable_name(tag)}"
        if column in df.columns:
            mask &= ~df[column].fillna(False).astype(bool)

    include = spec.get("conditions", spec.get("include"))
    if include:
        mask &= evaluate_conditions(df, include)
    exclude = spec.get("exclude")
    if exclude:
        mask &= ~evaluate_conditions(df, exclude)
    return mask.fillna(False)


def _recognized_source_module(value: Any) -> str | None:
    """Map common overlay source labels to normalized model module names."""
    aliases = {
        "cre": "cre",
        "commercial_real_estate": "cre",
        "candi": "candi",
        "ci": "candi",
        "commercial_and_industrial": "candi",
        "consumer": "consumer",
    }
    return aliases.get(_normalize_module_name(value))


def _normalize_module_name(value: Any) -> str:
    """Normalize module labels consistently with model routing."""
    return str(value).lower().replace("&", "and").replace(" ", "_")


def _bucket_ratios_from_rows(rows: pd.DataFrame, balance_field: str, level: str) -> Dict[str, float]:
    """Calculate bucket ratios from borrower-level rows for one stress level."""
    bucket_col = "base_bucket" if level == "Base" else f"stressed_bucket_{level}"
    if bucket_col not in rows.columns:
        return {bucket: np.nan for bucket in BUCKETS}
    total = float(pd.to_numeric(rows[balance_field], errors="coerce").sum())
    return {
        bucket: pct(pd.to_numeric(rows.loc[rows[bucket_col] == bucket, balance_field], errors="coerce").sum(), total)
        for bucket in BUCKETS
    }


def _source_audit_fields(source_specs: List[Mapping[str, Any]], use_weights: bool) -> Dict[str, str]:
    """Build semicolon-delimited overlay source audit fields."""
    return {
        "source_names": ";".join(str(spec.get("name", spec.get("portfolio", "source"))) for spec in source_specs),
        "source_weights": ";".join(_source_weight_text(spec, use_weights) for spec in source_specs),
        "source_selection": ";".join(_source_selection_text(spec) for spec in source_specs),
        "source_description": ";".join(str(spec.get("name", spec.get("portfolio", "source"))) for spec in source_specs),
    }


def _source_weight_text(spec: Mapping[str, Any], use_weights: bool) -> str:
    """Format one source weight for overlay_summary."""
    name = str(spec.get("name", spec.get("portfolio", "source")))
    if not use_weights:
        return f"{name}=balance_weighted"
    return f"{name}={to_number(spec.get('weight', 1.0), 1.0):g}"


def _source_selection_text(spec: Mapping[str, Any]) -> str:
    """Describe how one source population was selected."""
    name = str(spec.get("name", spec.get("portfolio", "source")))
    pieces = []
    if spec.get("portfolio") is not None:
        pieces.append(f"portfolio={spec.get('portfolio')}")
    if spec.get("tags") or spec.get("tag"):
        pieces.append(f"tags={','.join(map(str, as_list(spec.get('tags', spec.get('tag')))))}")
        if spec.get("respect_module_priority", True):
            pieces.append("primary_module=required")
    if spec.get("conditions") or spec.get("include"):
        pieces.append("conditions=configured")
    if spec.get("exclude") or spec.get("exclude_tags") or spec.get("exclude_tag"):
        pieces.append("exclusions=configured")
    return f"{name}({';'.join(pieces) if pieces else 'summary_portfolio'})"


def _grown_ratio(
    base_ratio: float,
    source_base_ratio: float,
    source_level_ratio: float,
    config: Mapping[str, Any],
    exceptions: List[Dict[str, Any]],
    portfolio: Any,
    level: str,
    bucket: str,
    source_description: Any,
) -> float:
    """Scale an overlay ratio by source ratio growth.

    Called twice per overlay/level: once for Special Mention and once for
    Substandard. Zero source-base ratios cannot produce growth factors, so the
    configured zero-base behavior is logged.
    """
    if source_base_ratio and not pd.isna(source_base_ratio):
        return to_number(base_ratio, 0.0) * (source_level_ratio / source_base_ratio)
    behavior = config.get("zero_base_behavior", "absolute_delta")
    record_exception(
        exceptions,
        "WARNING",
        "overlay",
        "OVERLAY_ZERO_SOURCE_BASE_RATIO",
        "Overlay source base ratio was zero or unavailable; zero-base overlay behavior was used.",
        portfolio=portfolio,
        stress_level=level,
        bucket=bucket,
        source=str(source_description),
        details=f"zero_base_behavior={behavior}; source_level_ratio={source_level_ratio}",
    )
    if behavior == "absolute_delta":
        return to_number(base_ratio, 0.0) + max(to_number(source_level_ratio, 0.0) - to_number(source_base_ratio, 0.0), 0.0)
    return to_number(base_ratio, 0.0)


def _cap_ratios(sm_ratio: float, sub_ratio: float, max_total: float = 1.0) -> Tuple[float, float]:
    """Constrain SM/Sub ratios so combined stressed ratios do not exceed 100%."""
    sm_ratio = min(max(to_number(sm_ratio, 0.0), 0.0), 1.0)
    sub_ratio = min(max(to_number(sub_ratio, 0.0), 0.0), 1.0)
    max_total = min(max(to_number(max_total, 1.0), 0.0), 1.0)
    if sm_ratio + sub_ratio <= max_total:
        return sm_ratio, sub_ratio
    total = sm_ratio + sub_ratio
    if total <= 0:
        return 0.0, 0.0
    return max_total * sm_ratio / total, max_total * sub_ratio / total
