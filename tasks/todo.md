# Review follow-up

- [x] Return document citation metadata through the workflow and API and display it in Streamlit.
- [x] Fix French/English chart metric matching and explicit unsupported years.
- [x] Run Ruff fixes and verify regression cases and the full local suite.
- [x] Review the RAG and ingestion folders and explain remaining concerns.

## Review

Charts: reproduced 13 failing regression cases before the fix. Final suite: 20 passed in 56.51s, covering chart selection, unsupported years, workflow/API source metadata, and Streamlit source display after reruns. Ruff: All checks passed! git diff --check: exit 0. Live Neon/Gemini calls were not performed; external dependencies were mocked for the workflow tests. Existing workflow edits were preserved. Remaining RAG/ingestion findings were reviewed without changing their behavior; Ruff applied lint corrections throughout the project.

## Measurement coverage follow-up

- [x] Count valid readings for each metric and report coverage against full UTC periods.
- [x] Explain incomplete totals in synthesis and verify missing months, null metrics, and leap years.
- [x] Explain automatic vector indexing, BM25 reuse, ingestion consistency, query rewriting, and safe SQL generation.

Coverage review: three regression cases failed before implementation. Full suite: 24 passed in 22.80s. Coverage uses valid per-metric counts divided by complete UTC calendar periods, including wholly missing periods and leap years. Missing totals remain null rather than zero or extrapolated estimates. Synthesis is instructed to disclose incomplete coverage. Database reads are mocked in tests; live Neon contents and privileges were not inspected or changed. BM25 caching, query rewriting, generated SQL, and ingestion reconciliation remain design recommendations.

## Conversation and BM25 reuse

- [x] Reuse a thread-safe BM25 instance in Python RAM for the running workflow; refresh by restarting FastAPI.
- [x] Pass bounded recent user/assistant history from Streamlit through the API.
- [x] Resolve follow-ups before routing, retrieval, and chart selection while retaining the original question.
- [x] Verify caching, history isolation, source grounding, and UI clear-history behavior; update usage documentation.

Conversation/cache review: a follow-up regression failed before implementation. Final suite: 37 passed in 22.11s; Ruff: All checks passed! git diff --check: exit 0. History is limited to six messages, 4,000 characters each, and remains request-local. No-history requests avoid an extra rewrite call; prior answers are used only to resolve references. Reusable BM25 is initialized lazily under a lock; vector queries still fetch current matches and handle new IDs. No refresh trigger or database changes were introduced. Tests mock Gemini/Neon; live rewrite quality and retrieval benchmark improvements have not been measured.

## Generated SQL and ingestion reliability

- [ ] Validate generated PostgreSQL queries with a fail-closed AST policy.
- [ ] Execute bounded queries using a verified dedicated reader and read-only transactions.
- [ ] Integrate generation, result metadata, charting, and source/coverage instructions in the workflow.
- [ ] Provide and test dedicated reader provisioning, then verify the live reader if authorized.
- [ ] Normalize UTC timestamps before conflict checks and safely replace validated historical periods.
- [ ] Test query safety, ingestion regressions, and the full user flow; update documentation.
