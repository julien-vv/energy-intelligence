
import json
import math
import re
import unicodedata
from html import escape
from pathlib import Path
from uuid import uuid4

import plotly.graph_objects as go

OUTPUT_DIR = Path("data/outputs")

METRICS = {
    "consommation": "Electricity consumption",
    "nucleaire": "Nuclear production",
    "eolien": "Wind production",
    "solaire": "Solar production",
    "hydraulique": "Hydropower production",
    "gaz": "Gas production",
    "charbon": "Coal production",
    "fioul": "Oil production",
    "bioenergies": "Bioenergy production",
}

METRIC_ALIASES = {
    "consommation": ("consommation", "consumption"),
    "nucleaire": ("nucleaire", "nuclear"),
    "eolien": ("eolien", "eolienne", "wind"),
    "solaire": ("solaire", "solar"),
    "hydraulique": ("hydraulique", "hydroelectric", "hydropower", "hydro"),
    "gaz": ("gaz", "gas"),
    "charbon": ("charbon", "coal"),
    "fioul": ("fioul", "oil"),
    "bioenergies": ("bioenergies", "bioenergie", "bioenergy"),
}


def normalize(text: str) -> str:
    # Remove accents and normalize case
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(char)
    )


def create_chart(question: str, data_context: str) -> str | None:
    payload = json.loads(data_context)
    if payload.get("mode") == "generated_sql":
        return create_sql_chart(payload)

    # Identify the metric requested by the user
    normalized_question = normalize(question)

    metric = next(
        (
            name for name, aliases in METRIC_ALIASES.items()
            if any(
                re.search(rf"\b{re.escape(alias)}\b", normalized_question)
                for alias in aliases
            )
        ),
        None,
    )

    if metric is None:
        print("[Chart] No supported metric detected")
        return None

    # Never substitute supported years for an explicitly unsupported year.
    years = sorted({
        int(year)
        for year in re.findall(r"\b(?:19|20)\d{2}\b", question)
    }) or [2024, 2025]

    if any(year not in (2024, 2025) for year in years):
        print("[Chart] Requested years are outside the supported 2024–2025 period")
        return None

    # Reuse the SQL results already stored in LangGraph state
    granularity = payload["granularity"]
    rows = payload["data"]

    value_column = f"{metric}_twh"
    period_column = "month" if granularity == "month" else "year"

    selected_rows = [
        row for row in rows
        if int(str(row[period_column])[:4]) in years
        and row.get(value_column) is not None
    ]

    if not selected_rows:
        print("[Chart] No matching data found")
        return None

    periods = [str(row[period_column]) for row in selected_rows]
    values = [float(row[value_column]) for row in selected_rows]

    fig = go.Figure()

    # Select chart type based on time granularity
    if granularity == "month":
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=values,
                mode="lines+markers",
                name=METRICS[metric],
            )
        )
    else:
        fig.add_trace(
            go.Bar(
                x=periods,
                y=values,
                name=METRICS[metric],
            )
        )

    fig.update_layout(
        title=f"{METRICS[metric]} — {', '.join(map(str, years))}",
        xaxis_title="Period",
        yaxis_title="Energy (TWh)",
        hovermode="x unified",
        template="plotly_white",
    )

    # Save an interactive standalone HTML chart
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    year_label = "_".join(map(str, years))
    output_path = (
        OUTPUT_DIR / f"{metric}_{granularity}_{year_label}.html"
    )

    fig.write_html(
        str(output_path),
        include_plotlyjs=True,
    )

    return str(output_path.resolve())


def create_sql_chart(payload: dict) -> str | None:
    """Plot explicitly named query columns without guessing their meaning."""
    spec = payload.get("chart")
    rows = payload.get("data", [])
    if not spec or not rows or payload.get("truncated") or payload.get("status") != "ok":
        return None
    x_column, y_column = spec["x"], spec["y"]
    if x_column not in payload["columns"] or y_column not in payload["columns"]:
        return None
    points = []
    for row in rows:
        if row.get(x_column) is None or row.get(y_column) is None:
            continue
        try:
            value = float(row[y_column])
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(value):
            return None
        points.append((str(row[x_column]), value))
    if not points:
        return None
    x_values, y_values = zip(*points, strict=True)
    trace = (
        go.Scatter(x=x_values, y=y_values, mode="lines+markers")
        if spec["kind"] == "line"
        else go.Bar(x=x_values, y=y_values)
    )
    figure = go.Figure(trace)
    figure.update_layout(
        title=escape(spec["title"]),
        xaxis_title=escape(x_column),
        yaxis_title=escape(spec["unit"]),
        template="plotly_white",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"analysis_{uuid4().hex}.html"
    figure.write_html(str(path), include_plotlyjs=True)
    return str(path.resolve())
