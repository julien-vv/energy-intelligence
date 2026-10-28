
import argparse
import calendar
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl

from energy_intelligence.ingestion.rte_etl import (
    Settings,
    load,
    transform,
)

API_URL = (
    "https://odre.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/eco2mix-national-cons-def/exports/csv"
)

DATA_DIR = Path("data/raw/rte")
QUALITY_DIR = Path("data/quality")

YEARS = [2024, 2025]

# Use existing CSV files to avoid unnecessary downloads
FORCE_DOWNLOAD = False
MIN_YEAR_COVERAGE = 0.9


def year_bounds(year: int) -> tuple[datetime, datetime]:
    if isinstance(year, bool) or not isinstance(year, int) or not 2000 <= year < 9999:
        raise ValueError("Year must be an integer between 2000 and 9998")
    return datetime(year, 1, 1, tzinfo=UTC), datetime(year + 1, 1, 1, tzinfo=UTC)


def download_year(year: int, *, refresh: bool = False) -> Path:
    year_bounds(year)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = DATA_DIR / f"eco2mix_{year}.csv"

    # Reuse an existing CSV file
    if (
        output_path.exists()
        and output_path.stat().st_size > 0
        and not FORCE_DOWNLOAD
        and not refresh
    ):
        print(f"\nUsing existing file: {output_path}")
        return output_path

    # Filter by UTC timestamp, not the text-based date field
    params = {
        "where": (
            f"date_heure >= date'{year}-01-01T00:00:00Z' "
            f"AND date_heure < date'{year + 1}-01-01T00:00:00Z'"
        ),
        "timezone": "UTC",
        "use_labels": "false",
        "delimiter": ";",
    }

    print(f"\nDownloading RTE data for {year}...")

    # Download to a temporary file before replacing the CSV
    temporary_path = output_path.with_suffix(".tmp")

    try:
        with httpx.Client(
            timeout=180.0,
            follow_redirects=True,
        ) as client, client.stream(
            "GET",
            API_URL,
            params=params,
        ) as response:

            if response.status_code != 200:
                print(
                    response.read().decode(
                        "utf-8",
                        errors="replace",
                    )
                )
                response.raise_for_status()

            with temporary_path.open("wb") as file:
                for chunk in response.iter_bytes():
                    file.write(chunk)

        temporary_path.replace(output_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print(f"Downloaded: {output_path}")
    return output_path


def process_year(year: int, database_url: str, *, refresh: bool = False) -> None:
    period_start, period_end = year_bounds(year)
    # 1. Extract historical data
    csv_path = download_year(year, refresh=refresh)

    df = pl.read_csv(
        csv_path,
        separator=";",
        infer_schema_length=0,
    )

    print(f"Raw records: {df.height}")

    if df.is_empty():
        raise ValueError(f"No data found for {year}")

    # 2. Transform and handle data-quality issues
    quarantine_path = QUALITY_DIR / (
        f"rte_conflicts_{year}_{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.csv"
    )

    records = df.to_dicts()
    for record in records:
        timestamp = datetime.fromisoformat(
            record["date_heure"]
        ).astimezone(UTC)
        if not period_start <= timestamp < period_end:
            raise ValueError(f"Source measurement falls outside UTC year {year}")

    cleaned_df = transform(
        records,
        quarantine_path=quarantine_path,
    )

    if cleaned_df.is_empty():
        raise ValueError(
            f"No valid measurements for {year}"
        )

    expected = (366 if calendar.isleap(year) else 365) * 48
    coverage = cleaned_df.height / expected
    print(f"Year {year} valid coverage: {cleaned_df.height}/{expected} ({coverage:.2%})")
    if coverage < MIN_YEAR_COVERAGE:
        raise ValueError(
            f"Refusing to replace {year}: valid measurement coverage {coverage:.2%} "
            f"is below {MIN_YEAR_COVERAGE:.0%}; inspect or refresh the source CSV"
        )

    # 3. Load into Neon PostgreSQL
    load(
        cleaned_df, database_url,
        period_start=period_start, period_end=period_end,
    )

    print(f"Year {year} completed.")


def main():
    parser = argparse.ArgumentParser(description="Import validated historical RTE measurements")
    parser.add_argument("--refresh", action="store_true", help="Download fresh CSV files instead of reusing the cache")
    args = parser.parse_args()
    settings = Settings()

    for year in YEARS:
        process_year(year, settings.database_url, refresh=args.refresh)

    print("\nHistorical RTE ingestion completed.")


if __name__ == "__main__":
    main()
