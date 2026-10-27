
from pathlib import Path

import pymupdf
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


REPORTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "raw"
    / "reports"
)


def load_reports(directory: Path) -> list[Document]:
    documents = []

    for pdf_path in sorted(directory.glob("*.pdf")):
        with pymupdf.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text(sort=True).strip()

                if not text:
                    continue

                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_number,
                        },
                    )
                )

    return documents


def main():
    documents = load_reports(REPORTS_DIR)

    if not documents:
        raise ValueError("No extractable PDF text found.")

    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    print(f"Extracted pages: {len(documents)}")
    print(f"Generated chunks: {len(nodes)}")

    for node in nodes[:3]:
        print(f"\nSource: {node.metadata['source']}")
        print(f"PDF page: {node.metadata['page']}")
        print(node.text[:300])


if __name__ == "__main__":
    main()
