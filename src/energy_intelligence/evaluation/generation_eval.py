
import json
import re
import unicodedata
from pathlib import Path

import psycopg
from litellm import completion
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from pgvector.psycopg import register_vector

from energy_intelligence.rag.hybrid_search import hybrid_retrieve
from energy_intelligence.rag.rag_demo import Settings

# Ground-truth questions verified against the PDF reports
EVAL_CASES = [
    {
        "question": (
            "De combien la consommation électrique française "
            "corrigée du climat a-t-elle augmenté en 2024 ?"
        ),
        "expected": "0,7 %",
        "source": "BE2024 - Rapport Complet.pdf",
        "page": 6,
    },
    {
        "question": (
            "Quelle était la consommation brute "
            "d'électricité en France en 2024 ?"
        ),
        "expected": "442,2 TWh",
        "source": "BE2024 - Rapport Complet.pdf",
        "page": 12,
    },
    {
        "question": (
            "Quelle était la production totale "
            "d'électricité en France en 2024 ?"
        ),
        "expected": "539 TWh",
        "source": "BE2024 - Rapport Complet.pdf",
        "page": 16,
    },
    {
        "question": (
            "Quelle était la consommation brute "
            "d'électricité française en 2025 ?"
        ),
        "expected": "446,1 TWh",
        "source": "BE2025 - Rapport Complet.pdf",
        "page": 10,
    },
    {
        "question": (
            "By what percentage did the European Power Benchmark "
            "fall in Q3 2023 compared with Q3 2022?"
        ),
        "expected": "74 %",
        "source": "entso-e_market_report_2024.pdf",
        "page": 7,
    },
]


def normalize_text(text: str) -> str:
    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()

    # Normalize percentage formatting: "74 %" and "74%" match
    text = re.sub(r"(\d)\s+%", r"\1%", text)

    # Normalize repeated whitespace
    return " ".join(text.split())


def main():
    settings = Settings()

    # 1. Initialize the embedding model
    model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",
        api_key=settings.gemini_api_key,
        embedding_config={"output_dimensionality": 768},
    )

    evaluation_results = []

    # 2. Connect to PostgreSQL
    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)

        for case in EVAL_CASES:
            question = case["question"]

            # 3. Retrieve relevant chunks using hybrid search
            chunks = hybrid_retrieve(
                question=question,
                conn=conn,
                model=model,
                top_k=5,
            )

            if not chunks:
                print(f"No chunks found for: {question}")
                continue

            # 4. Build the context with source references
            context = "\n\n".join(
                f"[{i}] Source: {chunk['source']}, "
                f"PDF page: {chunk['page']}\n"
                f"{chunk['content']}"
                for i, chunk in enumerate(chunks, start=1)
            )

            # 5. Generate an answer using Gemini
            response = completion(
                model=settings.llm_model,
                api_key=settings.gemini_api_key,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an electricity market analyst. "
                            "Answer using only the provided excerpts. "
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

            answer = response.choices[0].message.content or ""

            # 6. Check whether the expected value appears in the answer
            expected_found = (
                normalize_text(case["expected"])
                in normalize_text(answer)
            )

            # 7. Check whether the reference page was retrieved
            reference_retrieved = any(
                chunk["source"] == case["source"]
                and chunk["page"] == case["page"]
                for chunk in chunks
            )

            # 8. Save individual evaluation result
            result = {
                **case,
                "answer": answer,
                "expected_value_found": expected_found,
                "reference_page_retrieved": reference_retrieved,
                "retrieved_sources": [
                    {
                        "source": chunk["source"],
                        "page": chunk["page"],
                    }
                    for chunk in chunks
                ],
                "context": context,
            }

            evaluation_results.append(result)

            # Display individual results
            print(f"\nQuestion: {question}")
            print(f"Expected: {case['expected']}")
            print(f"Reference retrieved: {reference_retrieved}")
            print(f"Expected value found: {expected_found}")
            print(f"Answer:\n{answer}")
            print("-" * 60)

    # 9. Save evaluation results to JSON
    output_path = Path("data/evaluation/generation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(
            evaluation_results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 10. Calculate evaluation metrics
    total = len(evaluation_results)

    if total == 0:
        raise ValueError("No questions were evaluated")

    expected_matches = sum(
        result["expected_value_found"]
        for result in evaluation_results
    )

    retrieved_references = sum(
        result["reference_page_retrieved"]
        for result in evaluation_results
    )

    print("\n--- Generation Evaluation Results ---")
    print(f"Questions evaluated: {total}")

    print(
        f"Expected values found: "
        f"{expected_matches}/{total} "
        f"({expected_matches / total:.1%})"
    )

    print(
        f"Reference pages retrieved: "
        f"{retrieved_references}/{total} "
        f"({retrieved_references / total:.1%})"
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
