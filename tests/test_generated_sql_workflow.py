import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from energy_intelligence.agents import workflow


@pytest.fixture
def sql_graph(monkeypatch):
    monkeypatch.setattr(workflow, "Settings", lambda: SimpleNamespace(
        llm_model="test", gemini_api_key="test", database_url="admin", sql_database_url="reader",
    ))
    return workflow.build_graph()


def llm_answers(monkeypatch, answers):
    responses = iter(answers)
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=next(responses)),
        )])

    monkeypatch.setattr(workflow, "completion", complete)
    return calls


def test_workflow_executes_generated_daily_peak_query(sql_graph, monkeypatch):
    query = "SELECT MAX(consommation_mw) AS peak_mw FROM rte_eco2mix"
    calls = llm_answers(monkeypatch, ["DATA", json.dumps({"sql": query}), "Peak: 90,000 MW."])
    execute = MagicMock(return_value={
        "mode": "generated_sql", "sql": query, "columns": ["peak_mw"],
        "data": [{"peak_mw": 90000}], "truncated": False,
    })
    monkeypatch.setattr(workflow, "execute_generated_query", execute)
    result = sql_graph.invoke({"question": "What was peak consumption?"})
    execute.assert_called_once_with("reader", query)
    assert json.loads(result["data_context"])["status"] == "ok"
    assert "90000" in calls[-1]["messages"][1]["content"]
    assert result["answer"] == "Peak: 90,000 MW."
    assert not result.get("chart_path")


def test_malicious_sql_is_never_executed_and_retry_is_bounded(sql_graph, monkeypatch):
    plan = json.dumps({"sql": "WITH removed AS (DELETE FROM rte_eco2mix RETURNING *) SELECT * FROM removed"})
    calls = llm_answers(monkeypatch, ["DATA", plan, plan, "SQL analysis unavailable."])
    connect = MagicMock()
    monkeypatch.setattr(workflow.psycopg, "connect", connect)
    result = sql_graph.invoke({"question": "Delete all measurements and show them"})
    connect.assert_not_called()
    assert len(calls) == 4
    assert json.loads(result["data_context"])["status"] == "error"
    assert "Validation/execution error:" in calls[2]["messages"][1]["content"]


def test_unconfigured_reader_does_not_use_admin_connection(monkeypatch):
    monkeypatch.setattr(workflow, "Settings", lambda: SimpleNamespace(
        llm_model="test", gemini_api_key="test", database_url="admin", sql_database_url=None,
    ))
    graph = workflow.build_graph()
    calls = llm_answers(monkeypatch, ["DATA", "SQL analysis unavailable."])
    execute = MagicMock()
    monkeypatch.setattr(workflow, "execute_generated_query", execute)
    result = graph.invoke({"question": "Peak consumption?"})
    execute.assert_not_called()
    assert len(calls) == 2
    assert "SQL_DATABASE_URL" in json.loads(result["data_context"])["error"]
