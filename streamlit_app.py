
import os

import httpx
import streamlit as st
from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from google.oauth2 import id_token

from energy_intelligence.conversation import limit_history


# FastAPI backend URL
API_URL = os.getenv(
    "ENERGY_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


def api_headers() -> dict[str, str]:
    """Authenticate with FastAPI when running on Cloud Run."""
    if not os.getenv("K_SERVICE"):
        return {}

    token = id_token.fetch_id_token(Request(), API_URL)
    return {"Authorization": f"Bearer {token}"}


st.set_page_config(
    page_title="Energy Intelligence",
    page_icon="⚡",
    layout="wide",
)


# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


def check_backend() -> bool:
    """Check FastAPI availability."""
    try:
        response = httpx.get(
            f"{API_URL}/health",
            headers=api_headers(),
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json().get("status") == "ok"

    except (httpx.HTTPError, GoogleAuthError, ValueError):
        return False


def ask_backend(question: str) -> dict:
    """Send the question and conversation history to FastAPI."""
    history = limit_history([
        {
            "role": message["role"],
            "content": (
                message["answer"]
                if message["role"] == "assistant"
                else message["content"]
            ),
        }
        for message in st.session_state.messages[:-1]
    ])

    response = httpx.post(
        f"{API_URL}/ask",
        json={
            "question": question,
            "history": history,
        },
        headers=api_headers(),
        timeout=180.0,
    )

    response.raise_for_status()
    return response.json()


def fetch_chart(chart_url: str) -> str:
    """Fetch a generated chart through the authenticated backend."""
    response = httpx.get(
        f"{API_URL}{chart_url}",
        headers=api_headers(),
        timeout=30.0,
    )
    response.raise_for_status()
    return response.text


def display_assistant_message(message: dict) -> None:
    """Display an assistant response and its associated content."""

    # Display generated answer
    st.markdown(message["answer"])

    # Display documentary sources
    sources = message.get("sources", [])

    if sources:
        with st.expander("Sources documentaires"):
            for source in sources:
                st.text(
                    f"[{source['id']}] "
                    f"{source['source']} — "
                    f"page PDF {source['page']}"
                )
                st.text(source["excerpt"])

    # Display workflow route
    route_labels = {
        "data": "Data Analyst · PostgreSQL",
        "research": "Research Analyst · RAG",
        "both": "Data Analyst + Research Analyst",
    }

    route = message.get("route", "")

    if route:
        st.caption(
            f"Workflow : {route_labels.get(route, route)}"
        )

    # Display generated Plotly chart
    chart_url = message.get("chart_url")

    if (
        isinstance(chart_url, str)
        and chart_url.startswith("/charts/")
        and chart_url.endswith(".html")
        and "/" not in chart_url[len("/charts/"):]
    ):
        try:
            chart_html = fetch_chart(chart_url)

            # HTML must come from our trusted chart generator
            st.iframe(
                chart_html,
                height=520,
                width="stretch",
            )

            st.download_button(
                "Télécharger le graphique",
                data=chart_html,
                file_name=chart_url.rsplit("/", 1)[-1],
                mime="text/html",
                key=f"download_{chart_url}",
            )

        except (httpx.HTTPError, GoogleAuthError):
            st.warning(
                "Impossible de charger le graphique."
            )


# Sidebar
with st.sidebar:
    st.title("⚡ Energy Intelligence")

    st.markdown(
        "Assistant d'analyse du système électrique "
        "français."
    )

    st.divider()

    backend_online = check_backend()

    if backend_online:
        st.success("API connectée")
    else:
        st.error("API déconnectée")
        st.caption(
            "Impossible de joindre FastAPI. "
            "Vérifiez l'URL du backend et ses autorisations."
        )

    st.divider()

    st.subheader("Sources")

    st.markdown(
        "**Données structurées**\n\n"
        "RTE éCO2mix — 2024 et 2025\n\n"
        "**Documents**\n\n"
        "Rapports RTE et ENTSO-E"
    )

    st.divider()

    st.caption(
        "Les calculs SQL reposent sur les mesures "
        "disponibles. Certains intervalles sont manquants."
    )

    if st.button(
        "Effacer la conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# Main page
st.title("⚡ Energy Intelligence")

st.markdown(
    "Interrogez les données électriques françaises "
    "et les rapports RTE / ENTSO-E."
)

# Show example questions
if not st.session_state.messages:
    st.info(
        "**Exemples de questions**\n\n"
        "• Compare la production nucléaire en 2024 et 2025.\n\n"
        "• Montre l'évolution de la production solaire "
        "mois par mois en 2025.\n\n"
        "• Selon RTE, pourquoi la consommation électrique "
        "a-t-elle évolué en 2024 ?"
    )


# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            display_assistant_message(message)
        else:
            st.markdown(message["content"])


# Chat input
question = st.chat_input(
    "Posez une question sur l'énergie...",
    disabled=not backend_online,
)


if question:
    # Save and display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Execute the LangGraph workflow
    with st.chat_message("assistant"), st.spinner("Analyse en cours..."):
        try:
            result = ask_backend(question)

        except httpx.HTTPStatusError as exc:
            st.error(
                f"Erreur API : HTTP {exc.response.status_code}"
            )

        except (httpx.RequestError, GoogleAuthError):
            st.error(
                "Impossible de contacter FastAPI. "
                "Vérifiez le backend et son authentification."
            )

        else:
            assistant_message = {
                "role": "assistant",
                "answer": result.get(
                    "answer",
                    "Aucune réponse générée.",
                ),
                "route": result.get("route", ""),
                "chart_url": result.get("chart_url"),
                "sources": result.get("sources", []),
            }

            st.session_state.messages.append(
                assistant_message
            )

            display_assistant_message(
                assistant_message
            )
