"""A fail-closed PostgreSQL SELECT subset for the electricity analyst.

This validates structure, not whether a calculation answers the question.
Execution must additionally use a restricted role, pg_catalog search_path,
a read-only transaction, and resource limits.
"""

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import Scope, traverse_scope

MAX_SQL_CHARS = 12000
MAX_AST_NODES = 1000


class SQLValidationError(ValueError):
    """The generated statement is outside the allowed read-only SQL subset."""


# Exact classes: subclasses do not inherit permission to execute.
_ALLOWED_NODE_NAMES = {
    "Select", "Subquery", "Union", "Intersect", "Except", "With", "CTE",
    "From", "Table", "TableAlias", "Identifier", "Column", "Star", "Literal",
    "Null", "Boolean", "Alias", "Distinct", "Group", "Having", "Where",
    "Order", "Ordered", "Limit", "Offset", "Join", "Window", "WindowSpec",
    "Filter", "WithinGroup", "Case", "If", "EQ", "NEQ", "GT", "GTE",
    "LT", "LTE", "Is", "In", "Between", "Like", "ILike", "And", "Or",
    "Not", "Paren", "Add", "Sub", "Mul", "Div", "Mod", "Neg", "Tuple",
    "Exists", "Cast", "DataType", "DataTypeParam", "Interval", "Var",
    "AtTimeZone", "Extract", "TimestampTrunc", "DateTrunc", "TimeToStr",
    "Sum", "Avg", "Min", "Max", "Count", "Round", "Abs", "Ceil", "Floor",
    "Sqrt", "Pow", "Power", "Coalesce", "Nullif", "Greatest", "Least",
    "Lag", "Lead", "Rank", "DenseRank", "RowNumber", "FirstValue",
    "LastValue", "NthValue", "Stddev", "StddevPop", "StddevSamp",
    "Variance", "VariancePop", "PercentileCont", "PercentileDisc",
}
_ALLOWED_NODES = {
    getattr(exp, name) for name in _ALLOWED_NODE_NAMES if hasattr(exp, name)
}
_ALLOWED_TYPES = {
    "TINYINT", "SMALLINT", "INT", "BIGINT", "DECIMAL", "NUMERIC", "FLOAT",
    "DOUBLE", "REAL", "BOOLEAN", "DATE", "TIME", "TIMESTAMP", "TIMESTAMPTZ",
    "INTERVAL", "TEXT", "VARCHAR", "CHAR",
}
_QUERY_NODES = (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)


def _validate_nodes(statement: exp.Expression) -> None:
    nodes = list(statement.walk())
    if len(nodes) > MAX_AST_NODES:
        raise SQLValidationError("SQL query is too complex")
    for node in nodes:
        if type(node) not in _ALLOWED_NODES:
            raise SQLValidationError(f"SQL construct is not allowed: {type(node).__name__}")
        if isinstance(node, exp.Select) and (node.args.get("into") or node.args.get("locks")):
            raise SQLValidationError("SELECT INTO and row locks are not allowed")
        if isinstance(node, exp.With) and node.args.get("recursive"):
            raise SQLValidationError("Recursive CTEs are not allowed")
        if isinstance(node, exp.DataType) and node.this.name not in _ALLOWED_TYPES:
            raise SQLValidationError("Only built-in analytical data types are allowed")
        if isinstance(node, (exp.Limit, exp.Offset)):
            value = node.expression
            if not isinstance(value, exp.Literal) or value.is_string or not str(value.this).isdigit():
                raise SQLValidationError("LIMIT and OFFSET must be non-negative integer literals")
        if isinstance(node, exp.Column) and (node.args.get("db") or node.args.get("catalog")):
            raise SQLValidationError("Use table aliases rather than database-qualified columns")
        node.comments = None


def _qualify_physical_tables(statement: exp.Expression) -> None:
    physical_count = 0
    try:
        for scope in traverse_scope(statement):
            for table in scope.tables:
                # Schema-qualified names are physical even if a CTE has the same name.
                if table.catalog or (table.db and table.db != "public"):
                    raise SQLValidationError("Only public.rte_eco2mix may be queried")
                source = scope.sources.get(table.alias_or_name)
                if not table.db and isinstance(source, Scope):
                    continue
                if not isinstance(table.this, exp.Identifier) or table.name != "rte_eco2mix":
                    raise SQLValidationError("Only public.rte_eco2mix may be queried")
                table.set("db", exp.to_identifier("public"))
                physical_count += 1
    except SQLValidationError:
        raise
    except Exception as exc:
        raise SQLValidationError("Unable to resolve SQL table scopes") from exc
    if not physical_count:
        raise SQLValidationError("The query must read public.rte_eco2mix")


def validate_sql(query: str) -> str:
    """Return normalized SQL for a single allowed SELECT, or reject it.

    Deliberately unsupported: custom functions/operators/types, table functions,
    recursive CTEs, system tables, writes, locks, and procedural statements.
    The returned SQL, rather than the original model output, must be executed.
    """
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_SQL_CHARS:
        raise SQLValidationError("SQL must be non-empty and at most 12000 characters")
    try:
        statements = sqlglot.parse(query, read="postgres")
    except (SqlglotError, RecursionError) as exc:
        raise SQLValidationError("Invalid PostgreSQL query") from exc
    if len(statements) != 1 or not isinstance(statements[0], _QUERY_NODES):
        raise SQLValidationError("Exactly one SELECT query is required")
    statement = statements[0]
    _validate_nodes(statement)
    _qualify_physical_tables(statement)
    try:
        return statement.sql(dialect="postgres", unsupported_level=sqlglot.ErrorLevel.RAISE)
    except (SqlglotError, RecursionError) as exc:
        raise SQLValidationError("Unsupported PostgreSQL query") from exc
