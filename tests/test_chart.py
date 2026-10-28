import json

import pytest

from energy_intelligence.agents import chart


@pytest.fixture
def chart_context(tmp_path, monkeypatch):
    monkeypatch.setattr(chart, "OUTPUT_DIR", tmp_path)
    return json.dumps({
        "granularity": "year",
        "data": [
            {"year": year, **{f"{metric}_twh": 1 for metric in chart.METRICS}}
            for year in (2024, 2025)
        ],
    })


@pytest.mark.parametrize("metric", [
    "consumption", "nuclear", "wind", "solar", "hydropower",
    "gas", "coal", "oil", "bioenergy", "nucléaire", "éolien",
])
def test_english_and_french_metric_charts(metric, chart_context):
    assert chart.create_chart(f"Compare {metric} in 2024 and 2025", chart_context)


@pytest.mark.parametrize("years", ["2023", "2026", "2023 and 2024"])
def test_unsupported_years_do_not_show_different_years(years, chart_context):
    assert chart.create_chart(f"Production nucléaire en {years}", chart_context) is None


def test_metric_matching_uses_words(chart_context):
    assert chart.create_chart("Compare gasoline in 2024", chart_context) is None


def test_monthly_chart_only_includes_requested_year(tmp_path, monkeypatch):
    monkeypatch.setattr(chart, "OUTPUT_DIR", tmp_path)
    figures = []
    monkeypatch.setattr(chart.go.Figure, "write_html", lambda self, *a, **kw: figures.append(self))
    payload = json.dumps({
        "granularity": "month",
        "data": [
            {"month": "2024-01", "solaire_twh": 1},
            {"month": "2025-01", "solaire_twh": 2},
        ],
    })
    assert chart.create_chart("Solar production in 2025", payload)
    assert list(figures[0].data[0].x) == ["2025-01"]
    assert list(figures[0].data[0].y) == [2]


def test_generated_sql_chart_uses_returned_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(chart, "OUTPUT_DIR", tmp_path)
    figures = []
    monkeypatch.setattr(chart.go.Figure, "write_html", lambda self, *a, **kw: figures.append(self))
    payload = {
        "mode": "generated_sql", "status": "ok", "truncated": False,
        "columns": ["day", "peak_mw"],
        "data": [{"day": "2025-01-01", "peak_mw": 90000}, {"day": "2025-01-02", "peak_mw": None}],
        "chart": {"x": "day", "y": "peak_mw", "kind": "line", "title": "Daily peak", "unit": "MW"},
    }
    assert chart.create_chart("And daily?", json.dumps(payload))
    assert list(figures[0].data[0].x) == ["2025-01-01"]
    assert list(figures[0].data[0].y) == [90000]
    payload["truncated"] = True
    assert chart.create_chart("And daily?", json.dumps(payload)) is None
    payload["truncated"] = False
    payload["chart"]["y"] = "unknown"
    assert chart.create_chart("And daily?", json.dumps(payload)) is None
