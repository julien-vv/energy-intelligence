# Lessons

- Distinguish a pipeline missing from the repository from data missing in an existing cloud database. State which was verified.
- Describe empty retrieval according to the implementation: a populated vector index still returns nearest matches for unrelated questions; empty retrieval is a separate edge case.
- Separate automatic PostgreSQL index maintenance from text extraction, embedding generation, and row insertion; do not imply that existing vector indexes need rebuilding for each inserted chunk.
- Keep cache refresh proportional to current needs: when automatic invalidation is deferred, use process-local reuse and document restart-based refresh rather than introducing corpus versions or triggers.
