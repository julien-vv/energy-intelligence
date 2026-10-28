from unittest.mock import MagicMock

import pytest

from energy_intelligence.agents import sql_tools


def database_rows(monkeypatch, rows):
    connection = MagicMock()
    connection.execute.return_value.fetchall.return_value = rows
    connect = MagicMock()
    connect.return_value.__enter__.return_value = connection
    monkeypatch.setattr(sql_tools.psycopg, "connect", connect)
    return connection


def test_annual_metric_coverage_uses_full_leap_year(monkeypatch):
    connection = database_rows(monkeypatch, [{
        "year": 2024, "measurements": 17566,
        "consommation_twh": 440, "consommation_measurements": 17566,
        "solaire_twh": 20, "solaire_measurements": 8784,
    }])
    rows = sql_tools.query_energy_data("test")
    assert rows[0]["expected_measurements"] == 17568
    assert rows[0]["consommation_coverage_pct"] == 99.99
    assert rows[0]["solaire_coverage_pct"] == 50.0
    assert rows[0]["solaire_twh"] == 20
    assert rows[1]["year"] == 2025
    assert rows[1]["expected_measurements"] == 17520
    assert rows[1]["solaire_measurements"] == 0
    assert rows[1]["solaire_coverage_pct"] == 0
    assert rows[1]["solaire_twh"] is None
    query = connection.execute.call_args_list[-1].args[0]
    assert "COUNT(solaire_mw) AS solaire_measurements" in query


def test_monthly_coverage_keeps_missing_periods_and_null_values(monkeypatch):
    database_rows(monkeypatch, [{
        "month": "2024-02", "measurements": 1392,
        "consommation_twh": 30, "consommation_measurements": 1392,
        "solaire_twh": None, "solaire_measurements": 0,
    }])
    rows = sql_tools.query_energy_data("test", granularity="month")
    assert len(rows) == 24
    february = rows[1]
    assert february["month"] == "2024-02"
    assert february["expected_measurements"] == 1392
    assert february["consommation_coverage_pct"] == 100
    assert february["solaire_coverage_pct"] == 0
    assert february["solaire_twh"] is None
    assert rows[13]["expected_measurements"] == 1344
    assert rows[0]["measurements"] == 0
    assert rows[0]["consommation_twh"] is None


def test_empty_database_reports_zero_coverage_without_inventing_energy(monkeypatch):
    database_rows(monkeypatch, [])
    rows = sql_tools.query_energy_data("test")
    assert len(rows) == 2
    for row in rows:
        for metric in sql_tools.METRICS:
            assert row[f"{metric}_measurements"] == 0
            assert row[f"{metric}_coverage_pct"] == 0
            assert row[f"{metric}_twh"] is None


def test_invalid_granularity_is_rejected():
    with pytest.raises(ValueError, match="Unsupported granularity"):
        sql_tools.query_energy_data("test", granularity="day")
