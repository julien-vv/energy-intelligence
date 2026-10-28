from unittest.mock import MagicMock

import pytest
from dotenv import dotenv_values
from psycopg.conninfo import conninfo_to_dict

from energy_intelligence import setup_sql_reader as setup


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("SQL_DATABASE_URL", raising=False)
    path = tmp_path / ".env"
    path.write_text("DATABASE_URL='postgresql://admin:adminsecret@example.test/energy?sslmode=require&options=endpoint%3Dtest'\nOTHER=preserved\n")
    return path


def test_new_reader_preserves_admin_dsn_and_saves_only_verified_login(env_file, monkeypatch):
    admin = MagicMock()
    admin.execute.side_effect = [
        MagicMock(fetchone=lambda: None),
        MagicMock(fetchone=lambda: {"name": "energy"}),
    ] + [MagicMock() for _ in range(20)]
    connect = MagicMock()
    connect.return_value.__enter__.return_value = admin
    monkeypatch.setattr(setup.psycopg, "connect", connect)
    verify = MagicMock()
    execute = MagicMock()
    monkeypatch.setattr(setup, "verify_reader", verify)
    monkeypatch.setattr(setup, "execute_generated_query", execute)
    monkeypatch.setattr(setup.secrets, "token_urlsafe", lambda _: "reader-secret")

    assert setup.setup_reader(env_file) is True
    values = dotenv_values(env_file)
    reader = conninfo_to_dict(values["SQL_DATABASE_URL"])
    assert reader["user"] == "energy_analyst"
    assert reader["password"] == "reader-secret"
    assert reader["dbname"] == "energy"
    assert reader["host"] == "example.test"
    assert reader["sslmode"] == "require"
    assert reader["options"] == "endpoint=test"
    assert values["OTHER"] == "preserved"
    assert "admin:adminsecret" in values["DATABASE_URL"]
    verify.assert_called_once_with(admin, role="energy_analyst")
    execute.assert_called_once_with(values["SQL_DATABASE_URL"], "SELECT timestamp_utc FROM public.rte_eco2mix LIMIT 1")
    commands = "\n".join(str(call.args[0]) for call in admin.execute.call_args_list)
    assert "CREATE ROLE" in commands
    assert "NOSUPERUSER NOCREATEDB NOCREATEROLE" in commands
    assert "NOINHERIT NOREPLICATION NOBYPASSRLS" in commands
    assert "GRANT SELECT ON TABLE public.rte_eco2mix" in commands
    assert "SET LOCAL ROLE" not in commands
    assert "INSERT" not in commands
    assert "DROP ROLE" not in commands


def test_existing_reader_is_verified_without_admin_changes(env_file, monkeypatch):
    with env_file.open("a") as stream:
        stream.write("SQL_DATABASE_URL='postgresql://reader:secret@example.test/energy'\n")
    before = env_file.read_text()
    execute = MagicMock()
    connect = MagicMock()
    monkeypatch.setattr(setup, "execute_generated_query", execute)
    monkeypatch.setattr(setup.psycopg, "connect", connect)
    assert setup.setup_reader(env_file) is False
    assert env_file.read_text() == before
    connect.assert_not_called()
    execute.assert_called_once()


def test_preexisting_role_is_not_rotated_or_regranted(env_file, monkeypatch):
    admin = MagicMock()
    admin.execute.return_value.fetchone.return_value = {"exists": 1}
    connect = MagicMock()
    connect.return_value.__enter__.return_value = admin
    monkeypatch.setattr(setup.psycopg, "connect", connect)
    with pytest.raises(setup.ReaderSetupError, match="already exists"):
        setup.setup_reader(env_file)
    assert admin.execute.call_count == 1
    assert "SQL_DATABASE_URL" not in dotenv_values(env_file)


def test_permission_failure_rolls_back_before_password_is_saved(env_file, monkeypatch):
    admin = MagicMock()
    admin.execute.side_effect = [
        MagicMock(fetchone=lambda: None), MagicMock(fetchone=lambda: {"name": "energy"}),
    ] + [MagicMock() for _ in range(20)]
    connect = MagicMock()
    connect.return_value.__enter__.return_value = admin
    monkeypatch.setattr(setup.psycopg, "connect", connect)
    monkeypatch.setattr(setup, "verify_reader", MagicMock(side_effect=ValueError("sensitive DSN")))
    with pytest.raises(setup.ReaderSetupError, match="setup failed") as error:
        setup.setup_reader(env_file)
    assert "sensitive DSN" not in str(error.value)
    assert "SQL_DATABASE_URL" not in dotenv_values(env_file)
    assert connect.return_value.__exit__.call_args.args[0] is ValueError


def test_failed_login_cleans_up_only_new_role_and_does_not_save_password(env_file, monkeypatch):
    monkeypatch.setattr(setup, "_create_reader", lambda *args: "reader DSN")
    monkeypatch.setattr(setup, "execute_generated_query", MagicMock(side_effect=ValueError("private DSN")))
    cleanup = MagicMock()
    monkeypatch.setattr(setup, "_remove_unconfigured_reader", cleanup)
    with pytest.raises(setup.ReaderSetupError, match="setup failed") as error:
        setup.setup_reader(env_file)
    assert "private DSN" not in str(error.value)
    cleanup.assert_called_once()
    assert "SQL_DATABASE_URL" not in dotenv_values(env_file)


def test_existing_reader_failure_is_redacted_by_cli(env_file, monkeypatch, capsys):
    monkeypatch.setattr(setup, "setup_reader", MagicMock(side_effect=ValueError("password=private")))
    monkeypatch.setattr("sys.argv", ["setup_sql_reader", "--env-file", str(env_file)])
    assert setup.main() == 1
    output = capsys.readouterr().out
    assert "private" not in output
    assert "failed" in output


def test_file_save_failure_removes_only_new_reader(env_file, monkeypatch):
    before = env_file.read_text()
    monkeypatch.setattr(setup, "_create_reader", lambda *args: "reader DSN")
    monkeypatch.setattr(setup, "execute_generated_query", MagicMock())
    monkeypatch.setattr(setup, "set_key", lambda *args, **kwargs: (False, None, None))
    cleanup = MagicMock()
    monkeypatch.setattr(setup, "_remove_unconfigured_reader", cleanup)
    with pytest.raises(setup.ReaderSetupError, match="Could not save"):
        setup.setup_reader(env_file)
    cleanup.assert_called_once()
    assert env_file.read_text() == before
