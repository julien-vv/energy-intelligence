import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import polars as pl
import pytest

from energy_intelligence.ingestion import rte_etl, rte_history


def record(timestamp, consumption=100, **metrics):
    return {
        "date_heure": timestamp,
        **{name: 10 for name in rte_etl.MEASURE_COLUMNS},
        "consommation": consumption,
        **metrics,
    }


def test_equivalent_offsets_are_deduplicated_before_loading():
    result = rte_etl.transform([
        record("2024-01-01T00:00:00Z"),
        record("2024-01-01T01:00:00+01:00"),
    ])
    assert result.height == 1
    assert result["date_heure"][0] == "2024-01-01T00:00:00+00:00"


def test_conflicting_equivalent_instants_are_quarantined(tmp_path):
    quarantine = tmp_path / "conflicts.csv"
    result = rte_etl.transform([
        record("2024-01-01T00:00:00Z", 100),
        record("2024-01-01T01:00:00+01:00", 200),
        record("2024-01-01T00:30:00Z", 300),
    ], quarantine_path=quarantine)
    assert result.height == 1
    assert result["consommation"][0] == 300
    assert pl.read_csv(quarantine).height == 2


@pytest.mark.parametrize("timestamp", [
    "invalid", "2024-01-01T00:00:00", "2024-01-01T00:15:00Z",
])
def test_invalid_naive_or_off_grid_timestamps_are_rejected(timestamp):
    with pytest.raises(ValueError):
        rte_etl.transform([record(timestamp)])


def test_nonfinite_measurements_are_missing_not_energy():
    result = rte_etl.transform([
        record("2024-01-01T00:00:00Z", consommation=100, solaire=float("nan")),
        record("2024-01-01T00:30:00Z", float("inf")),
    ])
    assert result.height == 1
    assert result["solaire"][0] is None


def test_period_refresh_deletes_old_rows_and_inserts_within_one_transaction(monkeypatch):
    df = rte_etl.transform([record("2024-01-01T00:30:00Z")])
    conn = MagicMock()
    monkeypatch.setattr(rte_etl.psycopg, "connect", MagicMock(return_value=conn))
    start, end = datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)
    rte_etl.load(df, "test", period_start=start, period_end=end)
    active = conn.__enter__.return_value
    deletion = next(call for call in active.execute.call_args_list if "DELETE" in call.args[0])
    assert "timestamp_utc >= %s" in deletion.args[0]
    assert "timestamp_utc < %s" in deletion.args[0]
    assert deletion.args[1] == (start, end)
    active.cursor.return_value.__enter__.return_value.executemany.assert_called_once()
    assert conn.__exit__.call_args.args == (None, None, None)


def test_refresh_failure_is_propagated_to_transaction_for_rollback(monkeypatch):
    df = rte_etl.transform([record("2024-01-01T00:30:00Z")])
    conn = MagicMock()
    active = conn.__enter__.return_value
    active.cursor.return_value.__enter__.return_value.executemany.side_effect = RuntimeError("insert failed")
    monkeypatch.setattr(rte_etl.psycopg, "connect", MagicMock(return_value=conn))
    with pytest.raises(RuntimeError, match="insert failed"):
        rte_etl.load(
            df, "test", period_start=datetime(2024, 1, 1, tzinfo=UTC),
            period_end=datetime(2024, 1, 2, tzinfo=UTC),
        )
    assert conn.__exit__.call_args.args[0] is RuntimeError


@pytest.mark.parametrize("kind", ["empty", "outside", "naive", "reversed", "one_bound"])
def test_invalid_refresh_never_connects_to_database(monkeypatch, kind):
    df = rte_etl.transform([record("2024-01-01T00:00:00Z")])
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)
    if kind == "empty":
        df = df.clear()
    elif kind == "outside":
        start, end = datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 2, tzinfo=UTC)
    elif kind == "naive":
        start = start.replace(tzinfo=None)
    elif kind == "reversed":
        start, end = end, start
    else:
        end = None
    connect = MagicMock()
    monkeypatch.setattr(rte_etl.psycopg, "connect", connect)
    with pytest.raises(ValueError):
        rte_etl.load(df, "test", period_start=start, period_end=end)
    connect.assert_not_called()


def test_default_load_is_upsert_without_period_delete(monkeypatch):
    df = rte_etl.transform([record("2024-01-01T00:00:00Z")])
    conn = MagicMock()
    monkeypatch.setattr(rte_etl.psycopg, "connect", MagicMock(return_value=conn))
    rte_etl.load(df, "test")
    active = conn.__enter__.return_value
    assert not any("DELETE" in call.args[0] for call in active.execute.call_args_list)


def test_process_year_keeps_old_quarantine_if_validation_fails(tmp_path, monkeypatch):
    csv_path = tmp_path / "source.csv"
    pl.DataFrame([record("invalid")]).write_csv(csv_path, separator=";")
    quarantine = tmp_path / "rte_conflicts_2024.csv"
    quarantine.write_text("previous evidence", encoding="utf-8")
    monkeypatch.setattr(rte_history, "QUALITY_DIR", tmp_path)
    monkeypatch.setattr(rte_history, "download_year", lambda *args, **kwargs: csv_path)
    load = MagicMock()
    monkeypatch.setattr(rte_history, "load", load)
    with pytest.raises(ValueError):
        rte_history.process_year(2024, "test")
    assert quarantine.read_text(encoding="utf-8") == "previous evidence"
    load.assert_not_called()


def test_truncated_annual_csv_cannot_replace_existing_year(tmp_path, monkeypatch):
    csv_path = tmp_path / "source.csv"
    pl.DataFrame([record("2024-01-01T00:00:00Z")]).write_csv(csv_path, separator=";")
    monkeypatch.setattr(rte_history, "download_year", lambda *args, **kwargs: csv_path)
    load = MagicMock()
    monkeypatch.setattr(rte_history, "load", load)
    with pytest.raises(ValueError, match="coverage"):
        rte_history.process_year(2024, "test")
    load.assert_not_called()


def test_historical_refresh_passes_cleaned_complete_year_and_explicit_utc_bounds(tmp_path, monkeypatch):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    records = [record((start + timedelta(minutes=30 * i)).isoformat()) for i in range(17568)]
    records.append(record("2024-01-01T01:00:00+01:00", 200))
    csv_path = tmp_path / "source.csv"
    pl.DataFrame(records).write_csv(csv_path, separator=";")
    download = MagicMock(return_value=csv_path)
    monkeypatch.setattr(rte_history, "download_year", download)
    monkeypatch.setattr(rte_history, "QUALITY_DIR", tmp_path)
    load = MagicMock()
    monkeypatch.setattr(rte_history, "load", load)
    rte_history.process_year(2024, "test", refresh=True)
    download.assert_called_once_with(2024, refresh=True)
    cleaned = load.call_args.args[0]
    assert cleaned.height == 17567
    assert start.isoformat() not in cleaned["date_heure"].to_list()
    assert load.call_args.kwargs == {
        "period_start": start, "period_end": datetime(2025, 1, 1, tzinfo=UTC),
    }
    reports = list(tmp_path.glob("rte_conflicts_2024_*.csv"))
    assert len(reports) == 1
    assert pl.read_csv(reports[0]).height == 2


@pytest.mark.parametrize("year", [True, "2024", 1999, 9999])
def test_invalid_year_never_starts_download(year, monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(rte_history.httpx, "Client", client)
    with pytest.raises(ValueError, match="Year"):
        rte_history.download_year(year)
    client.assert_not_called()


def test_cached_csv_is_reused_unless_refresh_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(rte_history, "DATA_DIR", tmp_path)
    csv_path = tmp_path / "eco2mix_2024.csv"
    csv_path.write_bytes(b"cached")
    client = MagicMock()
    response = client.return_value.__enter__.return_value.stream.return_value.__enter__.return_value
    response.status_code = 200
    response.iter_bytes.return_value = iter([b"fresh CSV"])
    monkeypatch.setattr(rte_history.httpx, "Client", client)
    assert rte_history.download_year(2024) == csv_path
    client.assert_not_called()
    assert rte_history.download_year(2024, refresh=True) == csv_path
    assert csv_path.read_bytes() == b"fresh CSV"
    assert not csv_path.with_suffix(".tmp").exists()


@pytest.mark.parametrize("kind", ["duplicate", "nan", "outside_end"])
def test_invalid_measurements_are_refused_before_connecting(monkeypatch, kind):
    records = [record("2024-01-01T00:00:00Z")]
    if kind == "duplicate":
        records.append(record("2024-01-01T01:00:00+01:00"))
    elif kind == "nan":
        records[0]["solaire"] = float("nan")
    else:
        records[0]["date_heure"] = "2024-01-02T00:00:00Z"
    df = pl.DataFrame(records)
    connect = MagicMock()
    monkeypatch.setattr(rte_etl.psycopg, "connect", connect)
    with pytest.raises(ValueError):
        rte_etl.load(
            df, "test", period_start=datetime(2024, 1, 1, tzinfo=UTC),
            period_end=datetime(2024, 1, 2, tzinfo=UTC),
        )
    connect.assert_not_called()


class LocalConnection:
    """Exercise the loader's standard SQL/transaction flow on an isolated DB."""

    def __init__(self, database, fail_inserts=False):
        self.database = database
        self.fail_inserts = fail_inserts

    def __enter__(self):
        self.database.__enter__()
        return self

    def __exit__(self, *args):
        return self.database.__exit__(*args)

    def execute(self, sql, params=()):
        converted = tuple(value.isoformat() if isinstance(value, datetime) else value for value in params)
        return self.database.execute(sql.replace("%s", "?"), converted)

    def cursor(self):
        return LocalCursor(self.database, self.fail_inserts)


class LocalCursor:
    def __init__(self, database, fail_inserts):
        self.database = database
        self.fail_inserts = fail_inserts

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def executemany(self, sql, rows):
        if self.fail_inserts:
            raise RuntimeError("insert failed")
        converted = [
            tuple(value.isoformat() if isinstance(value, datetime) else value for value in row)
            for row in rows
        ]
        self.database.executemany(sql.replace("%s", "?"), converted)


def test_previously_loaded_quarantined_row_is_removed_on_refresh(tmp_path, monkeypatch):
    database = sqlite3.connect(":memory:")
    monkeypatch.setattr(rte_etl.psycopg, "connect", lambda _: LocalConnection(database))
    rte_etl.load(rte_etl.transform([
        record("2024-01-01T00:00:00Z", 100),
        record("2024-01-01T00:30:00Z", 200),
        record("2025-01-01T00:00:00Z", 300),
    ]), "test")
    updated = rte_etl.transform([
        record("2024-01-01T00:00:00Z", 100),
        record("2024-01-01T01:00:00+01:00", 900),
        record("2024-01-01T00:30:00Z", 600),
    ], quarantine_path=tmp_path / "conflicts.csv")
    rte_etl.load(
        updated, "test", period_start=datetime(2024, 1, 1, tzinfo=UTC),
        period_end=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert database.execute(
        "SELECT timestamp_utc, consommation_mw FROM rte_eco2mix ORDER BY timestamp_utc"
    ).fetchall() == [
        ("2024-01-01T00:30:00+00:00", 600),
        ("2025-01-01T00:00:00+00:00", 300),
    ]
    database.close()


def test_failed_refresh_restores_old_rows_in_real_transaction(monkeypatch):
    database = sqlite3.connect(":memory:")
    monkeypatch.setattr(rte_etl.psycopg, "connect", lambda _: LocalConnection(database))
    initial = rte_etl.transform([record("2024-01-01T00:00:00Z", 100)])
    rte_etl.load(initial, "test")
    monkeypatch.setattr(rte_etl.psycopg, "connect", lambda _: LocalConnection(database, fail_inserts=True))
    with pytest.raises(RuntimeError, match="insert failed"):
        rte_etl.load(
            rte_etl.transform([record("2024-01-01T00:30:00Z", 600)]), "test",
            period_start=datetime(2024, 1, 1, tzinfo=UTC),
            period_end=datetime(2025, 1, 1, tzinfo=UTC),
        )
    assert database.execute("SELECT consommation_mw FROM rte_eco2mix").fetchall() == [(100,)]
    database.close()
