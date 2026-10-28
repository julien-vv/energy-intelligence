"""Provision the separate PostgreSQL login used for generated SQL.

Run explicitly with ``python -m energy_intelligence.setup_sql_reader``.
The command never prints either connection string or the generated password.
"""

import argparse
import os
import secrets
from pathlib import Path

import psycopg
from dotenv import dotenv_values, set_key
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row

from energy_intelligence.agents.sql_execution import (
    execute_generated_query,
    verify_reader,
)

READER_ROLE = "energy_analyst"


class ReaderSetupError(ValueError):
    """A setup failure whose message is safe to display."""


def _create_reader(admin_url: str, password: str) -> str:
    with psycopg.connect(admin_url, row_factory=dict_row, connect_timeout=10) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = %s", (READER_ROLE,),
        ).fetchone()
        if exists:
            raise ReaderSetupError(
                "The energy_analyst role already exists. Configure its existing "
                "connection as DATABASE_URL; this command will not rotate its password."
            )
        database = conn.execute("SELECT current_database() AS name").fetchone()["name"]
        role = sql.Identifier(READER_ROLE)
        db = sql.Identifier(database)
        conn.execute(sql.SQL(
            "CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE "
            "NOINHERIT NOREPLICATION NOBYPASSRLS"
        ).format(role, sql.Literal(password)))
        conn.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(db, role))
        conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role))
        conn.execute(sql.SQL("GRANT SELECT ON TABLE public.rte_eco2mix TO {}").format(role))
        for setting, value in (
            ("default_transaction_read_only", "on"),
            ("statement_timeout", "5s"),
            ("lock_timeout", "1s"),
            ("search_path", "pg_catalog"),
        ):
            conn.execute(sql.SQL("ALTER ROLE {} IN DATABASE {} SET {} = {}").format(
                role, db, sql.Identifier(setting), sql.Literal(value),
            ))
        # Fail within the transaction if PUBLIC or inherited grants make this
        # role more powerful than intended, so role creation is rolled back.
        verify_reader(conn, role=READER_ROLE)

    parameters = conninfo_to_dict(admin_url)
    parameters.update(user=READER_ROLE, password=password)
    return make_conninfo(**parameters)


def _remove_unconfigured_reader(admin_url: str) -> None:
    """Undo only the new role created by this invocation if setup cannot finish."""
    with psycopg.connect(admin_url, row_factory=dict_row, connect_timeout=10) as conn:
        database = conn.execute("SELECT current_database() AS name").fetchone()["name"]
        role = sql.Identifier(READER_ROLE)
        conn.execute(sql.SQL("REVOKE SELECT ON TABLE public.rte_eco2mix FROM {}").format(role))
        conn.execute(sql.SQL("REVOKE USAGE ON SCHEMA public FROM {}").format(role))
        conn.execute(sql.SQL("REVOKE CONNECT ON DATABASE {} FROM {}").format(
            sql.Identifier(database), role,
        ))
        conn.execute(sql.SQL("DROP ROLE {}").format(role))


def setup_reader(env_file: Path = Path(".env")) -> bool:
    """Return True for a new reader, False when an existing reader was verified."""
    env_file = env_file.resolve()
    values = dotenv_values(env_file)
    reader_url = os.environ.get("DATABASE_URL") or values.get("DATABASE_URL")
    if reader_url:
        execute_generated_query(reader_url, "SELECT timestamp_utc FROM public.rte_eco2mix LIMIT 1")
        return False

    admin_url = os.environ.get("DATABASE_URL") or values.get("DATABASE_URL")
    if not admin_url:
        raise ReaderSetupError("DATABASE_URL is required to provision the dedicated reader.")
    if not env_file.is_file():
        raise ReaderSetupError("Create the requested environment file before provisioning the reader.")

    # Parse the DSN before making any database changes.
    try:
        conninfo_to_dict(admin_url)
    except psycopg.Error:
        raise ReaderSetupError("DATABASE_URL is not a valid PostgreSQL connection string.") from None

    created = False
    try:
        reader_url = _create_reader(admin_url, secrets.token_urlsafe(32))
        created = True
        execute_generated_query(reader_url, "SELECT timestamp_utc FROM public.rte_eco2mix LIMIT 1")
        success, _, _ = set_key(env_file, "DATABASE_URL", reader_url, quote_mode="always")
        if not success:
            raise ReaderSetupError("Could not save DATABASE_URL to the environment file.")
    except (psycopg.Error, OSError, ValueError) as exc:
        if created:
            try:
                _remove_unconfigured_reader(admin_url)
            except (psycopg.Error, OSError, ValueError):
                raise ReaderSetupError(
                    "Reader setup failed and cleanup could not finish. An administrator "
                    "must inspect the new energy_analyst role before retrying."
                ) from None
        if isinstance(exc, ReaderSetupError):
            raise
        raise ReaderSetupError(
            "Reader setup failed. Check database permissions, connectivity, and file access. "
            "No connection credentials were printed."
        ) from None
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    arguments = parser.parse_args()
    try:
        created = setup_reader(arguments.env_file)
    except ReaderSetupError as exc:
        print(str(exc))
        return 1
    except (psycopg.Error, OSError, ValueError):
        # Do not display third-party exception text: connection errors can
        # contain DSNs, and this CLI must never expose them to terminal logs.
        print("Reader setup failed. Verify configuration and administrator permissions.")
        return 1
    print(
        "Dedicated SQL reader created and verified; DATABASE_URL saved."
        if created else "Existing SQL reader verified; configuration unchanged."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
