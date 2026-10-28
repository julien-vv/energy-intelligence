from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SQLChart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: str = Field(min_length=1, max_length=63)
    y: str = Field(min_length=1, max_length=63)
    kind: Literal["bar", "line"] = "bar"
    title: str = Field(min_length=1, max_length=160)
    unit: str = Field(min_length=1, max_length=40)


class SQLPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sql: str = Field(min_length=1, max_length=12000)
    chart: SQLChart | None = None


SQL_GENERATION_PROMPT = """
You are the French electricity data analyst. Write a PostgreSQL SELECT query
answering the question. Return JSON only: {"sql": "...", "chart": null}.
For useful charts set chart to {"x": "result_column", "y": "numeric_result_column",
"kind": "bar" or "line", "title": "...", "unit": "TWh" or "MW" or "%"}.
Use exactly those keys. Choose chart columns that are actually returned by SQL.

Available table: public.rte_eco2mix
Columns: timestamp_utc TIMESTAMPTZ (unique, measured every 30 minutes),
consommation_mw, nucleaire_mw, eolien_mw, solaire_mw, hydraulique_mw,
gaz_mw, charbon_mw, fioul_mw, bioenergies_mw (all DOUBLE PRECISION, nullable).
The available period is 2024-01-01T00:00:00Z inclusive through
2026-01-01T00:00:00Z exclusive. Do not substitute different years for a request
outside this period: an unsupported period has no measurements.

Only SELECT, read-only nonrecursive CTEs, subqueries, joins on this table,
grouping, ordinary numeric/date expressions and window functions are allowed.
Use standard aggregates SUM, AVG, MIN, MAX, COUNT; ROUND, ABS, COALESCE, NULLIF;
EXTRACT, DATE_TRUNC and TO_CHAR; LAG, LEAD, ROW_NUMBER, RANK, DENSE_RANK.
Never write data, create objects, access catalogs, call other functions, use
custom types or operators, locks, recursive CTEs or multiple statements.
Do not schema-qualify function names. Use built-in numeric/date/timestamp types.
Always use explicit UTC calendar boundaries (timestamp_utc AT TIME ZONE 'UTC').
Filter to the requested period, order time-series chronologically, and use clear
unique result aliases. For yearly/monthly energy charts prefer aliases year/month
and metric_twh. Other time granularities, peaks, shares, averages and comparisons
are allowed; choose the SQL that answers the actual question.

Power is MW. An observed energy total is SUM(metric_mw) * 0.5 / 1000000 TWh;
AVG(metric_mw) is mean observed power, not energy. Never replace missing values
with zero or extrapolate totals. Never mix measured and climate-adjusted demand;
only measured consumption is available here. Round only final display values.
For each metric/period used, include COUNT(metric_mw) as metric_measurements and
coverage against the expected number of half-hour intervals in the ENTIRE
requested period, not COUNT(*): 48 per UTC day; 17568 in 2024; 17520 in 2025.
For full months calculate calendar days * 48, including leap years. Name coverage
columns metric_coverage_pct. Explicitly account for wholly absent periods where
possible. For arbitrary filters where an expected interval count has no clear
meaning, report valid counts and explain that coverage is unavailable rather
than inventing a denominator. Top-N and aggregates must be calculated in SQL
before output truncation. The executor returns at most 200 rows; don't use an
unnecessary small LIMIT that would hide requested periods.
""".strip()
