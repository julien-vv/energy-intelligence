from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from energy_intelligence.agents import workflow
from energy_intelligence.conversation import limit_history


@pytest.fixture
def conversation_graph(monkeypatch):
    monkeypatch.setattr(workflow, "Settings", lambda: SimpleNamespace(
        llm_model="test", gemini_api_key="test", database_url="test", sql_database_url="reader",
    ))
    calls = []
    resolved = "Pourquoi la production nucléaire française a-t-elle augmenté en 2025 ?"

    def completion(**kwargs):
        calls.append(kwargs["messages"])
        system = kwargs["messages"][0]["content"]
        if "Rewrite" in system:
            answer = resolved
        elif "Route electricity" in system:
            answer = "RESEARCH"
        else:
            answer = "Current documentary evidence [1]."
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=answer),
        )])

    retrieve = MagicMock(return_value=[{
        "id": 42, "source": "RTE.pdf", "page": 12, "content": "Current evidence.",
    }])
    monkeypatch.setattr(workflow, "completion", completion)
    monkeypatch.setattr(workflow, "GoogleGenAIEmbedding", MagicMock())
    monkeypatch.setattr(workflow.psycopg, "connect", MagicMock())
    monkeypatch.setattr(workflow, "register_vector", MagicMock())
    monkeypatch.setattr(workflow, "HybridRetriever", lambda: SimpleNamespace(retrieve=retrieve))
    return workflow.build_graph(), calls, retrieve, resolved


def test_follow_up_uses_questions_and_answers_to_resolve_references(conversation_graph):
    graph, calls, retrieve, resolved = conversation_graph
    original = "Pourquoi cette hausse ?"
    prior_answer = "An earlier assistant answer that must not become evidence."
    result = graph.invoke({"question": original, "history": [
        {"role": "user", "content": "Compare le nucléaire français en 2024 et 2025."},
        {"role": "assistant", "content": prior_answer},
    ]})
    assert retrieve.call_args.kwargs["question"] == resolved
    assert result["question"] == original
    assert result["standalone_question"] == resolved
    assert prior_answer in calls[0][1]["content"]
    assert "Compare le nucléaire" in calls[0][1]["content"]
    assert calls[1][1]["content"] == resolved
    assert original in calls[-1][1]["content"]
    assert resolved in calls[-1][1]["content"]
    assert prior_answer not in calls[-1][1]["content"]


def test_request_without_history_does_not_reuse_previous_conversation(conversation_graph):
    graph, calls, retrieve, _ = conversation_graph
    graph.invoke({"question": "Et ensuite ?", "history": [
        {"role": "user", "content": "Private prior question"},
    ]})
    calls.clear()
    graph.invoke({"question": "What does RTE report about solar?"})
    assert len(calls) == 2
    assert retrieve.call_args.kwargs["question"] == "What does RTE report about solar?"
    assert all("Private prior question" not in str(call) for call in calls)


def test_history_is_bounded_and_empty_answers_are_omitted():
    history = [{"role": "user", "content": f"message {i}"} for i in range(8)]
    history += [{"role": "assistant", "content": "x" * 5000}, {"role": "assistant", "content": ""}]
    limited = limit_history(history)
    assert limited[0]["content"] == "message 4"
    assert len(limited) == 5
    assert len(limited[-1]["content"]) == 4000
    assert len(history[-2]["content"]) == 5000


def test_empty_rewrite_falls_back_to_original_question(conversation_graph, monkeypatch):
    graph, _, retrieve, _ = conversation_graph
    answers = iter(["", "RESEARCH", "Fresh evidence [1]."])
    monkeypatch.setattr(workflow, "completion", lambda **kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=next(answers)))],
    ))
    result = graph.invoke({"question": "And in 2025?", "history": [
        {"role": "user", "content": "Production in 2024?"},
    ]})
    assert result["standalone_question"] == "And in 2025?"
    assert retrieve.call_args.kwargs["question"] == "And in 2025?"


def test_data_follow_up_uses_resolved_question_for_granularity_and_chart(conversation_graph, monkeypatch):
    graph, _, _, _ = conversation_graph
    prompts = []
    resolved = "Show monthly nuclear production in France in 2025."
    answers = iter([resolved, "DATA", '{"sql":"SELECT MAX(nucleaire_mw) AS peak_mw FROM rte_eco2mix"}', "SQL evidence."])

    def completion(**kwargs):
        prompts.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=next(answers)),
        )])

    monkeypatch.setattr(workflow, "completion", completion)
    query = MagicMock(return_value={"mode": "generated_sql", "data": [{"peak_mw": 60000}], "truncated": False})
    chart = MagicMock(return_value=None)
    monkeypatch.setattr(workflow, "execute_generated_query", query)
    monkeypatch.setattr(workflow, "create_chart", chart)
    result = graph.invoke({"question": "Monthly, please", "history": [
        {"role": "user", "content": "Nuclear production in France in 2025?"},
    ]})
    assert result["route"] == "data"
    assert prompts[2][1]["content"] == resolved
    assert "MAX(nucleaire_mw)" in query.call_args.args[1]
    assert query.call_args.args[0] == "reader"
    assert chart.call_args.kwargs["question"] == resolved
