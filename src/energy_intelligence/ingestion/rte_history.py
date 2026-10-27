
from pathlib import Path

import httpx
import polars as pl

from energy_intelligence.ingestion.rte_etl import (
    Settings,
    transform,
    load,
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


def download_year(year: int) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = DATA_DIR / f"eco2mix_{year}.csv"

    # Reuse an existing CSV file
    if (
        output_path.exists()
        and output_path.stat().st_size > 0
        and not FORCE_DOWNLOAD
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
        ) as client:
            with client.stream(
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


def process_year(year: int, database_url: str) -> None:
    # 1. Extract historical data
    csv_path = download_year(year)

    df = pl.read_csv(
        csv_path,
        separator=";",
        infer_schema_length=0,
    )

    print(f"Raw records: {df.height}")

    if df.is_empty():
        raise ValueError(f"No data found for {year}")

    # 2. Transform and handle data-quality issues
    quarantine_path = (
        QUALITY_DIR / f"rte_conflicts_{year}.csv"
    )

    # Remove an obsolete quarantine report
    if quarantine_path.exists():
        quarantine_path.unlink()

    cleaned_df = transform(
        df.to_dicts(),
        quarantine_path=quarantine_path,
    )

    if cleaned_df.is_empty():
        raise ValueError(
            f"No valid measurements for {year}"
        )

    # 3. Load into Neon PostgreSQL
    load(cleaned_df, database_url)

    print(f"Year {year} completed.")


def main():
    settings = Settings()

    for year in YEARS:
        process_year(year, settings.database_url)

    print("\nHistorical RTE ingestion completed.")


if __name__ == "__main__":
    main()
