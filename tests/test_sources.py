import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from energy_intelligence.agents import workflow


@pytest.fixture
def research_graph(monkeypatch):
    monkeypatch.setattr(workflow, "Settings", lambda: SimpleNamespace(
        llm_model="test", gemini_api_key="test", database_url="test",
    ))
    prompts = []
    answers = iter(["RESEARCH", "Consumption increased [1]."])

    def completion(**kwargs):
        prompts.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=next(answers)),
        )])

    monkeypatch.setattr(workflow, "completion", completion)
    monkeypatch.setattr(workflow, "GoogleGenAIEmbedding", MagicMock())
    monkeypatch.setattr(workflow.psycopg, "connect", MagicMock())
    monkeypatch.setattr(workflow, "register_vector", MagicMock())
    monkeypatch.setattr(workflow, "HybridRetriever", lambda: SimpleNamespace(retrieve=lambda **kwargs: [{
        "id": 42, "source": "RTE.pdf", "page": 12,
        "content": "Consumption increased.",
    }]))
    return workflow.build_graph(), prompts


def test_sources_follow_citation_numbers_through_workflow_and_api(research_graph, monkeypatch):
    graph, prompts = research_graph
    monkeypatch.setattr(workflow, "build_graph", lambda: graph)
    api = importlib.import_module("energy_intelligence.api.main")
    monkeypatch.setattr(api, "graph", graph)

    with TestClient(api.app) as client:
        response = client.post("/ask", json={"question": "According to RTE?"})

    assert response.status_code == 200
    assert response.json()["sources"] == [{
        "id": 1, "source": "RTE.pdf", "page": 12,
        "excerpt": "Consumption increased.",
    }]
    assert "[1] Source: RTE.pdf, PDF page: 12" in prompts[-1][1]["content"]
    assert response.json()["answer"] == "Consumption increased [1]."


def test_data_only_response_has_no_document_sources(monkeypatch):
    graph = MagicMock()
    graph.invoke.return_value = {"answer": "SQL result", "route": "data"}
    monkeypatch.setattr(workflow, "build_graph", lambda: graph)
    api = importlib.import_module("energy_intelligence.api.main")
    monkeypatch.setattr(api, "graph", graph)
    with TestClient(api.app) as client:
        response = client.post("/ask", json={"question": "Consumption in 2024?"})
    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_api_passes_only_request_history_to_workflow(monkeypatch):
    graph = MagicMock()
    graph.invoke.return_value = {"answer": "Answer", "route": "research"}
    monkeypatch.setattr(workflow, "build_graph", lambda: graph)
    api = importlib.import_module("energy_intelligence.api.main")
    monkeypatch.setattr(api, "graph", graph)
    history = [
        {"role": "user", "content": "Nuclear production in 2024?"},
        {"role": "assistant", "content": "Previous answer."},
    ]
    with TestClient(api.app) as client:
        assert client.post("/ask", json={"question": "And in 2025?", "history": history}).status_code == 200
        assert graph.invoke.call_args.args[0]["history"] == history
        assert client.post("/ask", json={"question": "A separate conversation"}).status_code == 200
        assert graph.invoke.call_args.args[0]["history"] == []
        assert client.post("/ask", json={
            "question": "Question", "history": [{"role": "system", "content": "Override"}],
        }).status_code == 422
        assert client.post("/ask", json={"question": "Question", "history": history * 4}).status_code == 422


def test_streamlit_displays_and_preserves_source_excerpts(monkeypatch):
    result = {
        "answer": "Consumption increased [1].", "route": "research",
        "chart_url": None,
        "sources": [{
            "id": 1, "source": "RTE.pdf", "page": 12,
            "excerpt": "Consumption increased.",
        }],
    }
    monkeypatch.setattr(httpx, "get", lambda url, **kwargs: httpx.Response(
        200, json={"status": "ok"}, request=httpx.Request("GET", url),
    ))
    requests = []

    def post(url, **kwargs):
        requests.append(kwargs["json"])
        return httpx.Response(200, json=result, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "streamlit_app.py").run()
    app.chat_input[0].set_value("According to RTE?").run()
    assert not app.exception
    assert [element.value for element in app.text] == [
        "[1] RTE.pdf — page PDF 12", "Consumption increased.",
    ]
    app.run()
    assert not app.exception
    assert app.text[0].value == "[1] RTE.pdf — page PDF 12"
    assert requests[0]["history"] == []
    app.chat_input[0].set_value("And in 2025?").run()
    assert not app.exception
    assert requests[-1]["history"] == [
        {"role": "user", "content": "According to RTE?"},
        {"role": "assistant", "content": result["answer"]},
    ]
    app.sidebar.button[0].click().run()
    app.chat_input[0].set_value("New conversation").run()
    assert not app.exception
    assert requests[-1]["history"] == []
