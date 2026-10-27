
# Energy Intelligence — MVP Specifications

## 1. Project Overview

Energy Intelligence is a Generative AI assistant for
analyzing the French and European electricity system.

It combines structured energy datasets, RAG, and
agentic workflows to deliver data-driven answers,
visualizations, and documented reports.

## 2. Data Sources

- RTE: electricity consumption, generation and CO2 data
- ENTSO-E: electricity market prices
- Météo-France: historical weather observations
- RTE and ENTSO-E reports: RAG document corpus

Initial analysis period: 2024–2025.

## 3. Functional Requirements

### UC01 — Electricity Consumption
Compare French electricity consumption in 2024 and 2025.
Expected: calculated indicators and time-series chart.

### UC02 — Energy Production
Compare nuclear, solar and wind generation.
Expected: values, percentages and charts.

### UC03 — Electricity Market Prices
Analyze French day-ahead electricity prices.
Expected: price statistics and visualizations.

### UC04 — Weather and Consumption
Compare temperature and electricity consumption.
Expected: charts and statistical analysis.

### UC05 — Document-Based Questions
Answer questions using official RTE and ENTSO-E reports.
Expected: grounded answers with document and page citations.

### UC06 — Multi-Agent Analysis
Combine numerical analysis and documentary research.
Expected: coordinated agent workflow and sourced synthesis.

### UC07 — Report Generation
Generate a structured energy analysis report.
Expected: indicators, charts, explanations and references.

## 4. Technical Requirements

- Python backend with FastAPI
- Two specialized agents orchestrated with LangGraph
- RAG using LlamaIndex and pgvector
- PostgreSQL for structured data
- Hosted LLMs accessed through LiteLLM
- Streamlit frontend with Plotly charts
- Langfuse tracing and observability
- Automated evaluation with pytest and Ragas
- Docker, Terraform, GitHub Actions and GCP deployment

## 5. Non-Functional Requirements

- Target operating cost: 0 EUR
- Provider-agnostic LLM architecture
- Reproducible numerical results
- Source citations for documentary answers
- Validated, read-only database access
- LLM and tool execution tracing
- Evaluation of answer quality and agent behavior
- Automated tests and CI/CD
- Secure handling of API keys

## 6. MVP Success Criteria

The project is complete when:

- All seven use cases are demonstrated
- RAG answers include verifiable citations
- Numerical calculations pass reference tests
- Both agents work together on hybrid questions
- LLM traces are available in Langfuse
- An evaluation report is generated
- The application is deployed and accessible
- The source code and documentation are published
