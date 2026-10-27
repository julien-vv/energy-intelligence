
import json
import re
import unicodedata
from pathlib import Path

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


def normalize(text: str) -> str:
    # Remove accents and normalize case
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(char)
    )


def create_chart(question: str, data_context: str) -> str | None:
    # Identify the metric requested by the user
    normalized_question = normalize(question)

    metric = next(
        (
            name for name in METRICS
            if name in normalized_question
        ),
        None,
    )

    if metric is None:
        print("[Chart] No supported metric detected")
        return None

    # Identify requested years; use both if unspecified
    years = sorted({
        int(year)
        for year in re.findall(r"\b202[45]\b", question)
    }) or [2024, 2025]

    # Reuse the SQL results already stored in LangGraph state
    payload = json.loads(data_context)
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
