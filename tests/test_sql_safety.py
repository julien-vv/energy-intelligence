import pytest

from energy_intelligence.agents.sql_safety import SQLValidationError, validate_sql


@pytest.mark.parametrize("query", [
    "SELECT timestamp_utc, consommation_mw FROM rte_eco2mix ORDER BY consommation_mw DESC LIMIT 10",
    ("SELECT EXTRACT(YEAR FROM timestamp_utc AT TIME ZONE 'UTC')::int AS year, "
    "ROUND((SUM(nucleaire_mw) * 0.5 / 1000000)::numeric, 3) AS nucleaire_twh, "
    "COUNT(nucleaire_mw) AS measurements FROM rte_eco2mix GROUP BY 1"),
    ("SELECT TO_CHAR(timestamp_utc AT TIME ZONE 'UTC', 'YYYY-MM') AS month, "
    "AVG(consommation_mw) FROM public.rte_eco2mix GROUP BY 1"),
    ("SELECT DATE_TRUNC('day', timestamp_utc AT TIME ZONE 'UTC') AS day, "
    "MAX(consommation_mw) AS peak FROM rte_eco2mix GROUP BY 1 ORDER BY peak DESC LIMIT 1"),
    ("WITH monthly AS (SELECT DATE_TRUNC('month', timestamp_utc) AS month, "
    "SUM(solaire_mw) AS output FROM rte_eco2mix GROUP BY 1) "
    "SELECT month, output, LAG(output) OVER (ORDER BY month) AS previous FROM monthly"),
    ("SELECT SUM(CASE WHEN solaire_mw IS NOT NULL THEN solaire_mw ELSE 0 END), "
    "COUNT(*) FILTER (WHERE solaire_mw IS NOT NULL) FROM rte_eco2mix"),
    ("SELECT COUNT(*) FROM rte_eco2mix WHERE timestamp_utc >= '2025-01-01T00:00:00Z'::timestamptz "
    "AND timestamp_utc < '2026-01-01T00:00:00Z'::timestamptz"),
    "SELECT SUM(gaz_mw) / NULLIF(SUM(consommation_mw), 0) FROM rte_eco2mix",
    "SELECT MAX(consommation_mw) FROM rte_eco2mix UNION ALL SELECT MIN(consommation_mw) FROM rte_eco2mix",
    ("SELECT AVG(consommation_mw) OVER (ORDER BY timestamp_utc ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) "
    "FROM rte_eco2mix LIMIT 10"),
    "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY consommation_mw) FROM rte_eco2mix",
])
def test_allows_analytical_read_queries(query):
    normalized = validate_sql(query)
    assert "public.rte_eco2mix" in normalized
    assert validate_sql(normalized) == normalized


@pytest.mark.parametrize("query", [
    "DELETE FROM rte_eco2mix",
    "UPDATE rte_eco2mix SET solaire_mw = 0",
    "INSERT INTO rte_eco2mix SELECT * FROM rte_eco2mix",
    "TRUNCATE rte_eco2mix",
    "DROP TABLE rte_eco2mix",
    "CREATE TABLE attack AS SELECT * FROM rte_eco2mix",
    "SELECT * INTO attack FROM rte_eco2mix",
    "SELECT * FROM rte_eco2mix FOR UPDATE",
    "SELECT * FROM rte_eco2mix; DELETE FROM rte_eco2mix",
    "SELECT * FROM rte_eco2mix; SET TRANSACTION READ WRITE",
    "WITH deleted AS (DELETE FROM rte_eco2mix RETURNING *) SELECT * FROM deleted",
    "WITH updated AS (UPDATE rte_eco2mix SET solaire_mw = 0 RETURNING *) SELECT * FROM updated",
    "WITH RECURSIVE x AS (SELECT * FROM rte_eco2mix) SELECT * FROM x",
    "SELECT pg_sleep(10) FROM rte_eco2mix",
    "SELECT set_config('transaction_read_only', 'off', true) FROM rte_eco2mix",
    "SELECT nextval('sequence') FROM rte_eco2mix",
    "SELECT pg_read_file('/etc/passwd') FROM rte_eco2mix",
    "SELECT dblink('connection', 'DELETE FROM rte_eco2mix') FROM rte_eco2mix",
    "SELECT public.sum(solaire_mw) FROM rte_eco2mix",
    "SELECT pg_catalog.sum(solaire_mw) FROM rte_eco2mix",
    "SELECT 1 OPERATOR(public.+) 2 FROM rte_eco2mix",
    "SELECT consommation_mw::public.custom_type FROM rte_eco2mix",
    "SELECT consommation_mw::regclass FROM rte_eco2mix",
    "SELECT * FROM pg_catalog.pg_roles",
    "SELECT * FROM rag_chunks",
    "SELECT * FROM other.rte_eco2mix",
    "SELECT * FROM generate_series(1, 1000000)",
    "SELECT * FROM rte_eco2mix LIMIT (SELECT 1)",
    "SELECT * FROM rte_eco2mix LIMIT ALL",
    "SELECT * FROM rte_eco2mix OFFSET -1",
    "SELECT 1",
    "COPY rte_eco2mix TO PROGRAM 'command'",
    "DO $$ BEGIN DELETE FROM rte_eco2mix; END $$",
    "EXPLAIN ANALYZE SELECT * FROM rte_eco2mix",
    "SELECT 'unterminated FROM rte_eco2mix",
    "SELECT * FROM rte_eco2mix CROSS JOIN LATERAL public.hidden_function()",
    "SELECT * FROM rte_eco2mix r JOIN pg_catalog.pg_roles p ON true",
])
def test_rejects_queries_outside_safe_subset(query):
    with pytest.raises(SQLValidationError):
        validate_sql(query)


def test_cte_name_cannot_hide_system_schema():
    query = "WITH pg_roles AS (SELECT * FROM rte_eco2mix) SELECT * FROM pg_catalog.pg_roles"
    with pytest.raises(SQLValidationError):
        validate_sql(query)


def test_scopes_allow_named_ctes_but_not_unrelated_physical_tables():
    query = "WITH x AS (SELECT * FROM rte_eco2mix) SELECT * FROM (SELECT * FROM x) nested"
    assert "public.rte_eco2mix" in validate_sql(query)
    with pytest.raises(SQLValidationError):
        validate_sql("WITH x AS (SELECT * FROM rte_eco2mix) SELECT * FROM hidden")


def test_comment_text_is_not_executed_or_preserved():
    result = validate_sql("SELECT /* DROP TABLE rte_eco2mix */ COUNT(*) FROM rte_eco2mix -- DELETE")
    assert "DROP" not in result
    assert "DELETE" not in result


def test_query_limits_are_enforced():
    with pytest.raises(SQLValidationError):
        validate_sql(" " * 12001)
    query = "SELECT " + ",".join(f"solaire_mw AS x{i}" for i in range(300)) + " FROM rte_eco2mix"
    with pytest.raises(SQLValidationError, match="complex"):
        validate_sql(query)
