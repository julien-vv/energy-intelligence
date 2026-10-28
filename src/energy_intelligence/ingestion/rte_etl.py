
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl
import psycopg
from pydantic_settings import BaseSettings, SettingsConfigDict

API_URL = (
    "https://odre.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/eco2mix-national-cons-def/records"
)

MEASURE_COLUMNS = [
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


class Settings(BaseSettings):
    database_url: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


def extract() -> list[dict]:
    # Fetch one day for testing
    params = {
        "where": "date = '2024-01-15'",
        "limit": 100,
        "order_by": "date_heure",
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()

    records = data["results"]

    if len(records) != data["total_count"]:
        raise ValueError("Incomplete API response")

    print(f"Extracted records: {len(records)}")
    return records


def transform(
    records: list[dict],
    quarantine_path: Path | None = None,
) -> pl.DataFrame:
    if not records:
        raise ValueError("No source measurements to transform")

    # Equivalent offset representations must compare as the same instant.
    normalized_records = []
    for record in records:
        timestamp = datetime.fromisoformat(record["date_heure"])
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("Measurement timestamps must include a timezone")
        timestamp = timestamp.astimezone(UTC)
        if timestamp.second or timestamp.microsecond or timestamp.minute not in (0, 30):
            raise ValueError("Measurements must fall on the UTC 30-minute grid")
        normalized_records.append({**record, "date_heure": timestamp.isoformat()})
    records = normalized_records
    # Select relevant electricity measurements
    df = pl.DataFrame(records, infer_schema_length=None)

    df = df.select(
        ["date_heure", *MEASURE_COLUMNS]
    )

    # Convert measurements to numeric values
    df = df.with_columns(
        pl.col(MEASURE_COLUMNS).cast(
            pl.Float64,
            strict=False,
        )
    )

    # NaN/infinity must be missing readings rather than enter energy totals.
    df = df.with_columns(
        pl.when(pl.col(MEASURE_COLUMNS).is_finite())
        .then(pl.col(MEASURE_COLUMNS))
        .otherwise(None)
    )

    # Remove rows without measured consumption
    raw_count = df.height

    df = df.drop_nulls(
        subset=["date_heure", "consommation"]
    )

    incomplete_count = raw_count - df.height

    # Remove strictly identical records
    before_dedup = df.height

    df = df.unique(maintain_order=True)

    identical_duplicates = before_dedup - df.height

    # Identify timestamps with conflicting measurements
    conflicts = (
        df.group_by("date_heure")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)
    )

    if not conflicts.is_empty():
        # Preserve all conflicting records for investigation
        conflicting_rows = df.join(
            conflicts.select("date_heure"),
            on="date_heure",
            how="inner",
        )

        if quarantine_path is None:
            raise ValueError(
                "Conflicting timestamps detected. "
                "Provide a quarantine_path."
            )

        quarantine_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        conflicting_rows.write_csv(quarantine_path)

        print(
            f"WARNING: {conflicts.height} conflicting "
            f"timestamps detected"
        )
        print(
            f"Conflicting records saved to: "
            f"{quarantine_path}"
        )

        # Exclude ambiguous timestamps from analysis
        df = df.join(
            conflicts.select("date_heure"),
            on="date_heure",
            how="anti",
        )

    # Sort chronologically
    df = df.sort("date_heure")

    print(f"Valid measurement records: {df.height}")
    print(f"Discarded incomplete records: {incomplete_count}")
    print(f"Removed identical duplicates: {identical_duplicates}")
    print(f"Conflicting timestamps excluded: {conflicts.height}")

    return df


def load(
    df: pl.DataFrame,
    database_url: str,
    *,
    period_start: datetime | None = None,
    period_end: datetime | None = None,
) -> None:
    """Upsert measurements, or atomically replace an explicitly validated period."""
    if df.is_empty():
        raise ValueError("Refusing to load an empty measurement set")
    if not {"date_heure", *MEASURE_COLUMNS}.issubset(df.columns):
        raise ValueError("Missing required measurement columns")
    if (period_start is None) != (period_end is None):
        raise ValueError("Both period_start and period_end are required for a refresh")
    if period_start is not None:
        if (
            period_start.tzinfo is None or period_start.utcoffset() is None
            or period_end.tzinfo is None or period_end.utcoffset() is None
        ):
            raise ValueError("Refresh bounds must include a timezone")
        period_start = period_start.astimezone(UTC)
        period_end = period_end.astimezone(UTC)
        if not timedelta(0) < period_end - period_start <= timedelta(days=366):
            raise ValueError("Refresh period must be positive and at most 366 days")

    # Create the PostgreSQL table
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS rte_eco2mix (
            timestamp_utc TIMESTAMPTZ PRIMARY KEY,
            consommation_mw DOUBLE PRECISION,
            nucleaire_mw DOUBLE PRECISION,
            eolien_mw DOUBLE PRECISION,
            solaire_mw DOUBLE PRECISION,
            hydraulique_mw DOUBLE PRECISION,
            gaz_mw DOUBLE PRECISION,
            charbon_mw DOUBLE PRECISION,
            fioul_mw DOUBLE PRECISION,
            bioenergies_mw DOUBLE PRECISION
        )
    """

    insert_sql = """
        INSERT INTO rte_eco2mix (
            timestamp_utc,
            consommation_mw,
            nucleaire_mw,
            eolien_mw,
            solaire_mw,
            hydraulique_mw,
            gaz_mw,
            charbon_mw,
            fioul_mw,
            bioenergies_mw
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        ON CONFLICT (timestamp_utc)
        DO UPDATE SET
            consommation_mw = EXCLUDED.consommation_mw,
            nucleaire_mw = EXCLUDED.nucleaire_mw,
            eolien_mw = EXCLUDED.eolien_mw,
            solaire_mw = EXCLUDED.solaire_mw,
            hydraulique_mw = EXCLUDED.hydraulique_mw,
            gaz_mw = EXCLUDED.gaz_mw,
            charbon_mw = EXCLUDED.charbon_mw,
            fioul_mw = EXCLUDED.fioul_mw,
            bioenergies_mw = EXCLUDED.bioenergies_mw
    """

    # Prepare PostgreSQL records
    rows = [
        (
            datetime.fromisoformat(
                record["date_heure"]
            ).astimezone(UTC),
            *[
                record[column]
                for column in MEASURE_COLUMNS
            ],
        )
        for record in df.iter_rows(named=True)
    ]

    timestamps = [row[0] for row in rows]
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("Duplicate measurement timestamps must be resolved before loading")
    for timestamp, consumption, *production in rows:
        if period_start is not None and not period_start <= timestamp < period_end:
            raise ValueError("Measurement falls outside the requested refresh period")
        if consumption is None:
            raise ValueError("Consumption is required for a valid measurement")
        if any(
            value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value))
            for value in (consumption, *production)
        ):
            raise ValueError("Measurements must be finite numbers or missing")

    # Insert records in batches
    batch_size = 1000

    with psycopg.connect(database_url) as conn:
        conn.execute(create_table_sql)

        if period_start is not None:
            # A later quarantine or missing source row must not leave a stale
            # measurement from a previous import. This deletion and all inserts
            # share the connection transaction and roll back together on failure.
            conn.execute(
                "DELETE FROM rte_eco2mix WHERE timestamp_utc >= %s AND timestamp_utc < %s",
                (period_start, period_end),
            )

        with conn.cursor() as cursor:
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]

                cursor.executemany(insert_sql, batch)

    print(f"Loaded records: {len(rows)}")


def main():
    settings = Settings()

    # Test the ETL on a single day
    records = extract()
    df = transform(records)
    load(df, settings.database_url)


if __name__ == "__main__":
    main()
