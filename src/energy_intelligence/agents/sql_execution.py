"""Execute model-written queries through a dedicated PostgreSQL reader."""

import psycopg
from psycopg.rows import dict_row

from energy_intelligence.agents.sql_safety import validate_sql

MAX_SQL_ROWS = 200


class SQLExecutionError(ValueError):
    """A safe, user-displayable failure without connection credentials."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


READER_CHECK_SQL = """
    SELECT (
        NOT (r.rolsuper OR r.rolcreatedb OR r.rolcreaterole
             OR r.rolreplication OR r.rolbypassrls)
        AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_auth_members WHERE member = r.oid)
        AND NOT pg_catalog.pg_has_role(r.oid, c.relowner, 'MEMBER')
        AND NOT pg_catalog.has_database_privilege(r.oid, current_database(), 'CREATE')
        AND NOT pg_catalog.has_schema_privilege(r.oid, 'public', 'CREATE')
        AND pg_catalog.has_table_privilege(r.oid, c.oid, 'SELECT')
        AND NOT pg_catalog.has_table_privilege(r.oid, c.oid, 'INSERT')
        AND NOT pg_catalog.has_table_privilege(r.oid, c.oid, 'UPDATE')
        AND NOT pg_catalog.has_table_privilege(r.oid, c.oid, 'DELETE')
        AND NOT pg_catalog.has_table_privilege(r.oid, c.oid, 'TRUNCATE')
        AND NOT pg_catalog.has_table_privilege(r.oid, c.oid, 'TRIGGER')
        AND NOT pg_catalog.has_table_privilege(r.oid, c.oid, 'REFERENCES')
    ) AS safe
    FROM pg_catalog.pg_roles r
    JOIN pg_catalog.pg_class c ON c.oid = 'public.rte_eco2mix'::pg_catalog.regclass
    WHERE r.rolname = COALESCE(%s::name, current_user)
"""


def verify_reader(conn, role: str | None = None) -> None:
    row = conn.execute(READER_CHECK_SQL, (role,)).fetchone()
    if not row or row["safe"] is not True:
        raise SQLExecutionError(
            "DATABASE_URL must use a dedicated non-owner role with SELECT-only "
            "access to public.rte_eco2mix and no role memberships or creation privileges."
        )


def execute_generated_query(database_url: str | None, query: str) -> dict:
    normalized = validate_sql(query)
    if not database_url:
        raise SQLExecutionError(
            "Configure DATABASE_URL with the dedicated reader using "
            "python -m energy_intelligence.setup_sql_reader."
        )
    try:
        with psycopg.connect(database_url, row_factory=dict_row, connect_timeout=10) as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute("SET LOCAL search_path = pg_catalog")
            conn.execute("SET LOCAL TIME ZONE 'UTC'")
            conn.execute("SET LOCAL statement_timeout = '5s'")
            conn.execute("SET LOCAL lock_timeout = '1s'")
            verify_reader(conn)
            with conn.cursor(name="energy_analysis") as cursor:
                cursor.execute(normalized)
                columns = [column.name for column in cursor.description]
                if len(columns) > 32 or len(columns) != len(set(columns)):
                    raise SQLExecutionError("Return at most 32 uniquely named result columns.", retryable=True)
                rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
    except psycopg.Error as exc:
        # Column/syntax failures may be corrected by the model; credentials never enter prompts.
        if exc.sqlstate and exc.sqlstate.startswith(("42", "22")):
            message = exc.diag.message_primary or "SQL expression is invalid."
            raise SQLExecutionError(message[:300], retryable=True) from None
        raise SQLExecutionError("SQL execution failed or exceeded the allowed time.") from None
    return {
        "mode": "generated_sql",
        "sql": normalized,
        "columns": columns,
        "data": rows[:MAX_SQL_ROWS],
        "truncated": len(rows) > MAX_SQL_ROWS,
        "row_limit": MAX_SQL_ROWS,
    }
