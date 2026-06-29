"""External source tag population tie-outs."""

from __future__ import annotations

import pandas as pd

from stress_engine.tagging.loan_tags import tags_as_list


def tag_population_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        for tag in tags_as_list(row.get("tags")):
            rows.append({"tag_name": tag, "loan_id": row.get("loan_id"), "balance": row.get("balance", 0.0)})
    if not rows:
        return pd.DataFrame(columns=["tag_name", "count", "balance"])
    return pd.DataFrame(rows).groupby("tag_name").agg(count=("loan_id", "count"), balance=("balance", "sum")).reset_index()


def tie_out_tag_populations(frame: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    populations = tag_population_summary(frame)
    rows = []
    for _, target in targets.iterrows():
        metric = target["population_metric"]
        tag_name = target["tag_name"]
        match = populations[populations["tag_name"] == tag_name]
        engine_value = 0.0 if match.empty else float(match.iloc[0].get(metric, 0.0))
        source_value = float(target["source_value"])
        tolerance = float(target["tolerance"])
        difference = engine_value - source_value
        rows.append(
            {
                "source_id": target["source_id"],
                "as_of_date": target["as_of_date"],
                "tag_name": tag_name,
                "population_metric": metric,
                "source_value": source_value,
                "engine_value": engine_value,
                "difference": difference,
                "pct_difference": difference / source_value if source_value else 0.0,
                "tolerance": tolerance,
                "tie_out_status": "pass" if abs(difference) <= tolerance else "fail",
                "source_owner": target["source_owner"],
            }
        )
    return pd.DataFrame(rows)
