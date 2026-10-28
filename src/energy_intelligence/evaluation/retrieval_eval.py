
import json
from pathlib import Path

import psycopg
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from pgvector import Vector
from pgvector.psycopg import register_vector
from pydantic_settings import BaseSettings, SettingsConfigDict
from rank_bm25 import BM25Okapi

from energy_intelligence.rag.hybrid_search import (
    reciprocal_rank_fusion,
    tokenize,
)

EVAL_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "evaluation"
    / "retrieval_questions.json"
)


class Settings(BaseSettings):
    gemini_api_key: str
    database_url: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


def hit_at_k(ranking, chunks_by_id, source, pages, k=5):
    # Check whether at least one relevant page appears in the top k
    for chunk_id in ranking[:k]:
        chunk_source, chunk_page, _ = chunks_by_id[chunk_id]

        if chunk_source == source and chunk_page in pages:
            return 1

    return 0


def reciprocal_rank_at_k(
    ranking, chunks_by_id, source, pages, k=5
):
    # Find the rank of the first relevant page
    for rank, chunk_id in enumerate(ranking[:k], start=1):
        chunk_source, chunk_page, _ = chunks_by_id[chunk_id]

        if chunk_source == source and chunk_page in pages:
            return 1.0 / rank

    return 0.0


def main():
    settings = Settings()

    # 1. Load evaluation questions
    eval_cases = json.loads(
        EVAL_PATH.read_text(encoding="utf-8")
    )

    if not eval_cases:
        raise ValueError("Evaluation dataset is empty")

    # 2. Initialize the embedding model
    model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",
        api_key=settings.gemini_api_key,
        embedding_config={"output_dimensionality": 768},
    )

    # Initialize metric accumulators
    vector_hits = 0
    hybrid_hits = 0
    vector_rr_total = 0.0
    hybrid_rr_total = 0.0

    # 3. Connect to PostgreSQL
    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)

        # Load indexed chunks
        rows = conn.execute(
            """
            SELECT id, source, page, content
            FROM rag_chunks
            ORDER BY id
            """
        ).fetchall()

        if not rows:
            raise ValueError("No chunks found in PostgreSQL")

        chunks_by_id = {
            chunk_id: (source, page, content)
            for chunk_id, source, page, content in rows
        }

        # 4. Build the BM25 index once
        tokenized_corpus = [
            tokenize(content)
            for _, _, _, content in rows
        ]

        bm25 = BM25Okapi(tokenized_corpus)

        # 5. Evaluate each question
        for case in eval_cases:
            question = case["question"]
            source = case["source"]
            pages = case["pages"]

            # Generate the question embedding
            query_embedding = Vector(
                model.get_query_embedding(question)
            )

            # Vector retrieval
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

            # BM25 retrieval
            bm25_scores = bm25.get_scores(
                tokenize(question)
            )

            ranked_indices = sorted(
                range(len(rows)),
                key=lambda i: (-bm25_scores[i], rows[i][0]),
            )

            bm25_ranking = [
                rows[i][0]
                for i in ranked_indices
                if bm25_scores[i] > 0
            ][:20]

            # Hybrid retrieval using RRF
            hybrid_ranking, _ = reciprocal_rank_fusion(
                [vector_ranking, bm25_ranking]
            )

            # 6. Calculate Hit@5
            vector_hit = hit_at_k(
                vector_ranking, chunks_by_id, source, pages
            )

            hybrid_hit = hit_at_k(
                hybrid_ranking, chunks_by_id, source, pages
            )

            # 7. Calculate reciprocal rank at 5
            vector_rr = reciprocal_rank_at_k(
                vector_ranking, chunks_by_id, source, pages
            )

            hybrid_rr = reciprocal_rank_at_k(
                hybrid_ranking, chunks_by_id, source, pages
            )

            # Update metric accumulators
            vector_hits += vector_hit
            hybrid_hits += hybrid_hit
            vector_rr_total += vector_rr
            hybrid_rr_total += hybrid_rr

            # Display individual results
            print(f"\nQuestion: {question}")
            print(
                f"Vector: Hit@5={vector_hit}, "
                f"RR@5={vector_rr:.3f}"
            )
            print(
                f"Hybrid: Hit@5={hybrid_hit}, "
                f"RR@5={hybrid_rr:.3f}"
            )

    # 8. Calculate average metrics
    total = len(eval_cases)

    print("\n--- Evaluation Results ---")
    print(f"Questions evaluated: {total}")

    print("\nVector Search:")
    print(f"  Hit@5: {vector_hits / total:.1%}")
    print(f"  MRR@5: {vector_rr_total / total:.1%}")

    print("\nHybrid Search:")
    print(f"  Hit@5: {hybrid_hits / total:.1%}")
    print(f"  MRR@5: {hybrid_rr_total / total:.1%}")


if __name__ == "__main__":
    main()
