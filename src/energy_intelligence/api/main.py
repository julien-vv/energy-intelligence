from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from energy_intelligence.agents.workflow import build_graph, run_workflow
from energy_intelligence.conversation import (
    MAX_HISTORY_MESSAGE_CHARS,
    MAX_HISTORY_MESSAGES,
)

# Directory containing generated Plotly charts
CHART_DIR = Path("data/outputs").resolve()
CHART_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Energy Intelligence API",
    version="0.1.0",
    description="AI assistant for electricity data analysis.",
)

# Serve generated Plotly HTML charts
app.mount(
    "/charts",
    StaticFiles(directory=CHART_DIR),
    name="charts",
)

# Initialize the LangGraph workflow
graph = build_graph()


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_HISTORY_MESSAGE_CHARS)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ConversationMessage] = Field(
        default_factory=list, max_length=MAX_HISTORY_MESSAGES
    )


class DocumentSource(BaseModel):
    id: int
    source: str
    page: int
    excerpt: str


class QuestionResponse(BaseModel):
    answer: str
    route: str
    chart_url: str | None = None
    sources: list[DocumentSource] = Field(default_factory=list)


@app.get("/health")
def health():
    # Check whether the API is running
    return {"status": "ok"}


@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    # Execute the existing LangGraph workflow
    result = run_workflow(
        graph,
        request.question,
        [message.model_dump() for message in request.history],
    )

    chart_url = None
    chart_path = result.get("chart_path")

    if chart_path:
        path = Path(chart_path).resolve()

        # Only expose generated HTML files from the chart directory
        if (
            path.is_file()
            and path.suffix.lower() == ".html"
            and path.parent == CHART_DIR
        ):
            chart_url = f"/charts/{path.name}"

    return QuestionResponse(
        answer=result["answer"],
        route=result["route"],
        chart_url=chart_url,
        sources=result.get("sources", []),
    )
