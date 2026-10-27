
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from energy_intelligence.agents.workflow import build_graph


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


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class QuestionResponse(BaseModel):
    answer: str
    route: str
    chart_url: str | None = None


@app.get("/health")
def health():
    # Check whether the API is running
    return {"status": "ok"}


@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    # Execute the existing LangGraph workflow
    result = graph.invoke({
        "question": request.question,
    })

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
    )
