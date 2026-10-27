
import os

import httpx
import streamlit as st


# FastAPI backend configuration
API_URL = os.getenv(
    "ENERGY_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


st.set_page_config(
    page_title="Energy Intelligence",
    page_icon="⚡",
    layout="wide",
)


# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


def check_backend() -> bool:
    # Verify FastAPI availability
    try:
        response = httpx.get(
            f"{API_URL}/health",
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json().get("status") == "ok"
    except (httpx.RequestError, httpx.HTTPStatusError):
        return False


def ask_backend(question: str) -> dict:
    # Send the question to FastAPI and LangGraph
    response = httpx.post(
        f"{API_URL}/ask",
        json={"question": question},
        timeout=180.0,
    )

    response.raise_for_status()
    return response.json()


def display_assistant_message(message: dict) -> None:
    # Display generated answer
    st.markdown(message["answer"])

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

    # Embed interactive Plotly chart when available
    chart_url = message.get("chart_url")

    if (
        isinstance(chart_url, str)
        and chart_url.startswith("/charts/")
        and chart_url.endswith(".html")
    ):
        full_chart_url = f"{API_URL}{chart_url}"

        st.iframe(
            full_chart_url,
            height=520,
            width="stretch",
        )

        st.link_button(
            "Ouvrir le graphique",
            full_chart_url,
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
            "Démarre FastAPI sur http://127.0.0.1:8000"
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

# Show example questions before the first interaction
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
    with st.chat_message("assistant"):
        with st.spinner("Analyse en cours..."):
            try:
                result = ask_backend(question)

            except httpx.HTTPStatusError as exc:
                st.error(
                    f"Erreur API : "
                    f"HTTP {exc.response.status_code}"
                )

            except httpx.RequestError:
                st.error(
                    "Impossible de contacter FastAPI. "
                    "Vérifiez que le backend est démarré."
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
                }

                st.session_state.messages.append(
                    assistant_message
                )

                display_assistant_message(
                    assistant_message
                )
