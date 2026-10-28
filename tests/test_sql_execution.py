from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from energy_intelligence.agents import sql_execution
from energy_intelligence.agents.sql_execution import (
    SQLExecutionError,
    execute_generated_query,
)
from energy_intelligence.agents.sql_safety import SQLValidationError

QUERY = "SELECT MAX(consommation_mw) AS peak_mw FROM rte_eco2mix"


@pytest.fixture
def reader(monkeypatch):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {"safe": True}
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.description = [SimpleNamespace(name="peak_mw")]
    cursor.fetchmany.return_value = [{"peak_mw": 90000}]
    connect = MagicMock()
    connect.return_value.__enter__.return_value = conn
    monkeypatch.setattr(sql_execution.psycopg, "connect", connect)
    return conn, cursor, connect


def test_query_is_normalized_and_executed_with_resource_guards(reader):
    conn, cursor, _ = reader
    result = execute_generated_query("reader", QUERY)
    statements = [call.args[0] for call in conn.execute.call_args_list]
    assert statements[0] == "SET TRANSACTION READ ONLY"
    assert "SET LOCAL search_path = pg_catalog" in statements
    assert "SET LOCAL statement_timeout = '5s'" in statements
    assert "public.rte_eco2mix" in cursor.execute.call_args.args[0]
    assert cursor.fetchmany.call_args.args == (201,)
    assert result["data"] == [{"peak_mw": 90000}]
    assert result["columns"] == ["peak_mw"]
    assert result["truncated"] is False


def test_unsafe_query_never_connects(reader):
    _, _, connect = reader
    with pytest.raises(SQLValidationError):
        execute_generated_query("reader", "DROP TABLE rte_eco2mix")
    connect.assert_not_called()


def test_missing_reader_has_no_admin_fallback(reader):
    _, _, connect = reader
    with pytest.raises(SQLExecutionError, match="SQL_DATABASE_URL"):
        execute_generated_query(None, QUERY)
    connect.assert_not_called()


def test_privileged_role_is_rejected_before_executing_model_sql(reader):
    conn, cursor, _ = reader
    conn.execute.return_value.fetchone.return_value = {"safe": False}
    with pytest.raises(SQLExecutionError, match="dedicated non-owner"):
        execute_generated_query("owner", QUERY)
    cursor.execute.assert_not_called()


def test_result_cap_is_explicit(reader):
    _, cursor, _ = reader
    cursor.fetchmany.return_value = [{"peak_mw": i} for i in range(201)]
    result = execute_generated_query("reader", QUERY)
    assert len(result["data"]) == 200
    assert result["truncated"] is True


def test_ambiguous_result_columns_are_rejected(reader):
    _, cursor, _ = reader
    cursor.description = [SimpleNamespace(name="value"), SimpleNamespace(name="value")]
    with pytest.raises(SQLExecutionError, match="uniquely named"):
        execute_generated_query("reader", QUERY)


def test_connection_failures_do_not_expose_credentials(reader):
    _, _, connect = reader
    connect.side_effect = sql_execution.psycopg.OperationalError("secret-password")
    with pytest.raises(SQLExecutionError) as failure:
        execute_generated_query("reader", QUERY)
    assert "secret-password" not in str(failure.value)
