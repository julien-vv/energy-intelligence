
import psycopg
from psycopg.rows import dict_row


METRICS = [
    "consommation",
    "nucleaire",
    "eolien",
    "solaire",
    "hydraulique",
    "gaz",
    "charbon",
    "fioul",
    "bioenergies",
]

GROUPINGS = {
    "year": (
        "EXTRACT(YEAR FROM timestamp_utc AT TIME ZONE 'UTC')::int",
        "year",
    ),
    "month": (
        "TO_CHAR(timestamp_utc AT TIME ZONE 'UTC', 'YYYY-MM')",
        "month",
    ),
}


def query_energy_data(
    database_url: str,
    granularity: str = "year",
) -> list[dict]:
    # Restrict queries to predefined aggregations
    if granularity not in GROUPINGS:
        raise ValueError("Unsupported granularity")

    period_expression, period_alias = GROUPINGS[granularity]

    # Convert 30-minute power measurements (MW) into energy (TWh)
    metric_columns = ",\n".join(
        (
            f"ROUND((SUM({metric}_mw) * 0.5 / 1000000)::numeric, 3) "
            f"AS {metric}_twh"
        )
        for metric in METRICS
    )

    query = f"""
        SELECT
            {period_expression} AS {period_alias},
            COUNT(*) AS measurements,
            {metric_columns}
        FROM rte_eco2mix
        WHERE timestamp_utc >= '2024-01-01T00:00:00Z'
          AND timestamp_utc < '2026-01-01T00:00:00Z'
        GROUP BY 1
        ORDER BY 1
    """

    # Execute only the predefined read-only SQL query
    with psycopg.connect(
        database_url,
        row_factory=dict_row,
    ) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        rows = conn.execute(query).fetchall()

    return rows
