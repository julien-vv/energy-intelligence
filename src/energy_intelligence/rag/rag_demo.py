
import psycopg
from litellm import completion
from pgvector.psycopg import register_vector
from pydantic_settings import BaseSettings, SettingsConfigDict
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import litellm
from dotenv import load_dotenv
from langfuse import get_client, observe
from energy_intelligence.rag.hybrid_search import hybrid_retrieve


class Settings(BaseSettings):
    gemini_api_key: str
    database_url: str
    llm_model: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

@observe(name="energy-intelligence-rag")
def run_rag():
    main()

def main():
    settings = Settings()

    question = (
        "Comment la consommation électrique française "
        "a-t-elle évolué en 2024 ?"
    )

    # 1. Initialize the embedding model
    model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",
        api_key=settings.gemini_api_key,
        embedding_config={"output_dimensionality": 768},
    )

    # 2. Retrieve chunks using hybrid search
    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)

        results = hybrid_retrieve(
            question=question,
            conn=conn,
            model=model,
            top_k=5,
        )

    if not results:
        print("No relevant documents found.")
        return

    # 3. Build context with source references
    context = "\n\n".join(
        f"[{i}] Source: {result['source']}, "
        f"PDF page: {result['page']}\n"
        f"{result['content']}"
        for i, result in enumerate(results, start=1)
    )

    # 4. Generate an answer using Gemini
    response = completion(
        model=settings.llm_model,
        api_key=settings.gemini_api_key,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an electricity market analyst. "
                    "Answer the user's question using only "
                    "the provided document excerpts. "
                    "Do not invent facts, figures, or dates. "
                    "Preserve numerical values, years, and units "
                    "exactly as stated in the sources. "
                    "Cite supporting excerpts using [1], [2], etc. "
                    "If the excerpts do not contain enough "
                    "information, explicitly say so. "
                    "Answer in the same language as the question."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Document excerpts:\n{context}"
                ),
            },
        ],
        max_tokens=1200,
    )

    # 5. Display the answer
    print(f"\nQuestion: {question}")
    print("\n--- RAG Answer ---")
    print(response.choices[0].message.content)

    # 6. Display retrieved sources
    print("\n--- Retrieved Sources ---")

    for i, result in enumerate(results, start=1):
        print(
            f"[{i}] {result['source']} "
            f"- Page {result['page']} "
            f"- RRF: {result['score']:.5f}"
        )


if __name__ == "__main__":
    # Load Langfuse credentials from .env
    load_dotenv()

    # Enable LiteLLM tracing
    litellm.callbacks = ["langfuse_otel"]

    # Run the RAG pipeline
    run_rag()

    # Send pending traces before exiting
    get_client().flush()
