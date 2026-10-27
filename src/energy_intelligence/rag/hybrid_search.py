
import re
import unicodedata

import psycopg
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from pgvector import Vector
from pgvector.psycopg import register_vector
from pydantic_settings import BaseSettings, SettingsConfigDict
from rank_bm25 import BM25Okapi
from langfuse import observe

class Settings(BaseSettings):
    gemini_api_key: str
    database_url: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "comment", "dans",
    "de", "des", "du", "en", "est", "et",
    "la", "le", "les", "par", "pour", "que", "qui",
    "sur", "the", "and", "of", "in", "to", "what",
}


def tokenize(text: str) -> list[str]:
    # Normalize accents and convert text to lowercase
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    # Extract words and numbers
    tokens = re.findall(r"[a-z0-9]+", text)

    return [
        token for token in tokens
        if token not in STOPWORDS
    ]


def reciprocal_rank_fusion(rankings, k=60):
    # Combine multiple rankings using reciprocal rank fusion
    scores = {}

    for ranking in rankings:
        for position, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1.0 / (k + position)
            )

    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            chunk_id,
        ),
    )

    return ranked_ids, scores


@observe(
    name="hybrid-retrieval",
    capture_input=False,
    capture_output=False,
)
def hybrid_retrieve(
    question: str,
    conn,
    model,
    top_k: int = 5,
) -> list[dict]:
    # 1. Load all indexed chunks
    rows = conn.execute(
        """
        SELECT id, source, page, content
        FROM rag_chunks
        ORDER BY id
        """
    ).fetchall()

    if not rows:
        return []

    # 2. Vector search with pgvector
    query_embedding = Vector(
        model.get_query_embedding(question)
    )

    vector_rows = conn.execute(
        """
        SELECT id
        FROM rag_chunks
        ORDER BY embedding <=> %s
        LIMIT 20
        """,
        (query_embedding,),
    ).fetchall()

    vector_ranking = [
        row[0] for row in vector_rows
    ]

    # 3. BM25 keyword search
    tokenized_corpus = [
        tokenize(content)
        for _, _, _, content in rows
    ]

    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(
        tokenize(question)
    )

    ranked_indices = sorted(
        range(len(rows)),
        key=lambda i: (
            -bm25_scores[i],
            rows[i][0],
        ),
    )

    bm25_ranking = [
        rows[i][0]
        for i in ranked_indices
        if bm25_scores[i] > 0
    ][:20]

    # 4. Fuse vector and BM25 rankings
    fused_ids, rrf_scores = reciprocal_rank_fusion(
        [vector_ranking, bm25_ranking]
    )

    # 5. Map chunk IDs to their content
    chunks_by_id = {
        row[0]: row
        for row in rows
    }

    # 6. Return the best chunks
    results = []

    for chunk_id in fused_ids[:top_k]:
        _, source, page, content = chunks_by_id[chunk_id]

        results.append(
            {
                "id": chunk_id,
                "source": source,
                "page": page,
                "content": content,
                "score": rrf_scores[chunk_id],
            }
        )

    return results


def main():
    settings = Settings()

    question = (
        "Comment la consommation électrique française "
        "a-t-elle évolué en 2024 ?"
    )

    # Initialize Gemini embedding model
    model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",
        api_key=settings.gemini_api_key,
        embedding_config={"output_dimensionality": 768},
    )

    # Connect to PostgreSQL
    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)

        results = hybrid_retrieve(
            question=question,
            conn=conn,
            model=model,
            top_k=5,
        )

    # Display retrieved chunks
    print(f"\nQuestion: {question}")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"RRF score: {result['score']:.5f}")
        print(f"Content: {result['content'][:400]}")


if __name__ == "__main__":
    main()
