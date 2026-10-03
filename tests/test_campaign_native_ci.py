"""Provider/DB-free admission and ordering tests for the dedicated CI helper."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from scripts import campaign_native_ci as ci

NAME = ci.PREFIX + "6c587160f5a84f32abcc1c0adf603ef8"
URL = f"postgresql://kairos:synthetic-fixture@localhost:5432/{NAME}"


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "postgresql://kairos:fixture@localhost:5432/kairos",
        URL.replace("localhost", "example.com"),
        URL.replace("localhost", "timescaledb"),
        URL.replace(":5432", ""),
        URL.replace(":5432", ":5433"),
        URL + "?database=kairos",
        URL + "#fragment",
        URL.replace("6c587160f5a84f32abcc1c0adf603ef8", "6c587160f5a81f32abcc1c0adf603ef8"),
        URL.replace("6c587160f5a84f32abcc1c0adf603ef8", "6c587160f5a84f321bcc1c0adf603ef8"),
        URL.replace("6c587160", "6C587160"),
        URL.replace("kairos:", "other_role:"),
        URL.replace(ci.PREFIX, ci.PREFIX + "unexpected_"),
    ],
)
def test_reject_nonexact_disposable_target_before_database_construction(monkeypatch, url):
    monkeypatch.setattr(ci, "Database", lambda *args, **kwargs: pytest.fail("unexpected DB construction"))
    with pytest.raises(ci.CampaignCIError):
        asyncio.run(ci.prepare(url))


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
def test_exact_uuid4_loopback_target_accepted(host):
    assert ci.require_target(URL.replace("localhost", host)) == NAME


@pytest.fixture
def database_stub(monkeypatch):
    calls = []
    facts = {"database_name": NAME, "role_name": "kairos", "server_version": 160010, "public_objects": 0}

    class Stub:
        def __init__(self, settings, *, migration_profile):
            assert settings.database_url == URL
            assert settings.pool_min_size == settings.pool_max_size == 1
            assert settings.command_timeout_s == 5.0
            assert migration_profile is ci.MigrationProfile.RESEARCH_CAMPAIGN
            self.pool = self
            calls.append("constructed")

        async def fetchrow(self, query):
            assert "current_database()" in query and "pg_catalog.pg_class" in query
            calls.append("facts")
            return facts

        async def migrate(self):
            calls.append("migrate")

        async def verify_schema(self):
            calls.append("schema")

        async def close(self):
            calls.append("close")

    async def verified(database, expected_name, *, local_only):
        assert expected_name == NAME and local_only is True
        calls.append("verified")

    monkeypatch.setattr(ci, "Database", Stub)
    monkeypatch.setattr(ci, "connect_verified_database", verified)
    return SimpleNamespace(calls=calls, facts=facts)


def test_verify_exact_fresh_target_before_migrations_and_always_close(database_stub, monkeypatch):
    monkeypatch.setenv("KAIROS_PERSISTENCE_DATABASE_URL", "postgresql://unused.invalid/primary")
    asyncio.run(ci.prepare(URL))
    assert database_stub.calls == ["constructed", "verified", "facts", "migrate", "schema", "close"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("database_name", "kairos"),
        ("role_name", "other_role"),
        ("server_version", 150010),
        ("server_version", 170000),
        ("server_version", True),
        ("public_objects", 1),
        ("public_objects", False),
    ],
)
def test_wrong_live_identity_or_nonempty_database_never_migrates(database_stub, key, value):
    database_stub.facts[key] = value
    with pytest.raises(ci.CampaignCIError):
        asyncio.run(ci.prepare(URL))
    assert database_stub.calls == ["constructed", "verified", "facts", "close"]


def test_verification_failure_closes_without_ddl(database_stub, monkeypatch):
    async def rejected(*args, **kwargs):
        raise RuntimeError("synthetic private DSN must not escape")

    monkeypatch.setattr(ci, "connect_verified_database", rejected)
    with pytest.raises(RuntimeError):
        asyncio.run(ci.prepare(URL))
    assert database_stub.calls == ["constructed", "close"]


def test_setup_timeout_closes_without_ddl(database_stub, monkeypatch):
    async def blocked(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(ci, "connect_verified_database", blocked)
    monkeypatch.setattr(ci, "PREPARE_TIMEOUT_S", 0.005)
    with pytest.raises(TimeoutError):
        asyncio.run(ci.prepare(URL))
    assert database_stub.calls == ["constructed", "close"]


@pytest.mark.parametrize("child", ["", "<skipped/>", "<failure/>", "<error/>"])
def test_native_report_requires_one_pass_not_skip(tmp_path, monkeypatch, child):
    monkeypatch.chdir(tmp_path)
    ci.REPORT.write_text(
        f'<testsuites><testsuite><testcase name="{ci.NATIVE_TARGET}">'
        f"{child}</testcase></testsuite></testsuites>",
        encoding="utf-8",
    )
    if child:
        with pytest.raises(ci.CampaignCIError):
            ci.check_result()
    else:
        ci.check_result()


@pytest.mark.parametrize(
    "xml",
    [
        "<testsuites/>",
        '<testsuite><testcase name="other"/></testsuite>',
        f'<testsuite><testcase name="{ci.NATIVE_TARGET}"/><testcase name="other"/></testsuite>',
        f'<testsuites><testcase name="{ci.NATIVE_TARGET}"/><error/></testsuites>',
        "not xml",
    ],
)
def test_zero_wrong_duplicate_or_invalid_report_denied(tmp_path, monkeypatch, xml):
    monkeypatch.chdir(tmp_path)
    ci.REPORT.write_text(xml, encoding="utf-8")
    with pytest.raises(ci.CampaignCIError):
        ci.check_result()


def test_oversized_or_missing_report_denied(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ci.CampaignCIError):
        ci.check_result()
    ci.REPORT.write_bytes(b"x" * (1024 * 1024 + 1))
    with pytest.raises(ci.CampaignCIError):
        ci.check_result()


def test_cli_never_prints_raw_driver_errors(database_stub, monkeypatch, capsys):
    async def rejected(*args, **kwargs):
        raise RuntimeError("synthetic secret DSN")

    monkeypatch.setattr(ci, "connect_verified_database", rejected)
    monkeypatch.setenv(ci.ENV_NAME, URL)
    assert ci.main(["prepare"]) == 1
    assert capsys.readouterr().out == "CAMPAIGN_NATIVE_CI_FAILED OPERATION_FAILED\n"


def test_cli_missing_env_and_unknown_command_fail_closed(monkeypatch, capsys):
    monkeypatch.delenv(ci.ENV_NAME, raising=False)
    assert ci.main(["prepare"]) == 1
    assert capsys.readouterr().out == "CAMPAIGN_NATIVE_CI_FAILED EXPLICIT_TARGET_REQUIRED\n"
    assert ci.main(["unknown-synthetic-input"]) == 1
    assert capsys.readouterr().out == "CAMPAIGN_NATIVE_CI_FAILED INVALID_COMMAND\n"


def test_cli_unknown_custom_error_category_never_leaks(monkeypatch, capsys):
    def rejected():
        raise ci.CampaignCIError("synthetic secret DSN")

    monkeypatch.setattr(ci, "check_result", rejected)
    assert ci.main(["check-result"]) == 1
    assert capsys.readouterr().out == "CAMPAIGN_NATIVE_CI_FAILED OPERATION_FAILED\n"
