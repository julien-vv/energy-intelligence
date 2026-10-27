
from datetime import datetime
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


def load(df: pl.DataFrame, database_url: str) -> None:
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
            datetime.fromisoformat(record["date_heure"]),
            *[
                record[column]
                for column in MEASURE_COLUMNS
            ],
        )
        for record in df.iter_rows(named=True)
    ]

    # Insert records in batches
    batch_size = 1000

    with psycopg.connect(database_url) as conn:
        conn.execute(create_table_sql)

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
