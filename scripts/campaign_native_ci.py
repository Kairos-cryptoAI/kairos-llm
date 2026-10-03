"""CI-only preparation of one fresh local synthetic campaign database.

No default database, dotenv, provider or trading configuration is consulted.
This is not an operator migration command or economic qualification.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

from kairos_persistence import Database, PersistenceSettings
from kairos_persistence.database import MigrationProfile
from kairos_persistence.database_target import connect_verified_database, require_database_target_url

ENV_NAME = "KAIROS_RESEARCH_CAMPAIGN_TEST_DATABASE_URL"
PREFIX = "kairos_sim_test_campaign_"
NATIVE_TARGET = "test_native_three_arm_scheduler_and_restart_on_explicit_disposable_campaign_db"
PREPARE_TIMEOUT_S = 60.0
CLOSE_TIMEOUT_S = 5.0
REPORT = Path("campaign-native.xml")
CAUSAL_REPORT = Path("causal-campaign-native.xml")
CAUSAL_NATIVE_TARGET = (
    "test_native_causal_producer_strategy_router_pair_and_restart_on_explicit_disposable_campaign_db"
)
FAILURE_CATEGORIES = frozenset(
    {
        "EXPLICIT_TARGET_REQUIRED",
        "INVALID_TEST_TARGET",
        "FRESH_PG16_TARGET_REQUIRED",
        "NATIVE_RESULT_REQUIRED",
        "INVALID_NATIVE_RESULT",
        "EXACT_NATIVE_PASS_REQUIRED",
    }
)


class CampaignCIError(ValueError):
    """Fixed, sanitized failure category; never contains credentials or SQL."""


def require_target(database_url: str | None) -> str:
    if not isinstance(database_url, str) or not database_url:
        raise CampaignCIError("EXPLICIT_TARGET_REQUIRED")
    try:
        parsed = urlsplit(database_url)
        name = parsed.path.removeprefix("/")
        suffix = name.removeprefix(PREFIX)
        if not name.startswith(PREFIX) or re.fullmatch(r"[0-9a-f]{32}", suffix) is None:
            raise CampaignCIError("UUID4_TARGET_REQUIRED")
        namespace = UUID(hex=suffix)
        if namespace.version != 4 or namespace.hex != suffix:
            raise CampaignCIError("UUID4_TARGET_REQUIRED")
        require_database_target_url(database_url, name, local_only=True)
        if (
            parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.port != 5432
            or parsed.username != "kairos"
        ):
            raise CampaignCIError("CI_LOOPBACK_TARGET_REQUIRED")
        return name
    except (ValueError, TypeError):
        raise CampaignCIError("INVALID_TEST_TARGET") from None


async def prepare(database_url: str | None) -> None:
    name = require_target(database_url)
    # Explicit values override any ambient persistence settings; no envfile is loaded.
    database = Database(
        PersistenceSettings(
            _env_file=None,
            database_url=database_url,
            pool_min_size=1,
            pool_max_size=1,
            command_timeout_s=5.0,
        ),
        migration_profile=MigrationProfile.RESEARCH_CAMPAIGN,
    )
    try:
        async with asyncio.timeout(PREPARE_TIMEOUT_S):
            await connect_verified_database(database, name, local_only=True)
            facts = await database.pool.fetchrow(
                """SELECT current_database() AS database_name, current_user AS role_name,
                          current_setting('server_version_num')::integer AS server_version,
                          (SELECT count(*) FROM pg_catalog.pg_class c
                           JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
                           WHERE n.nspname='public' AND c.relkind IN ('r','p','v','m','S'))
                          AS public_objects"""
            )
            if (
                facts is None
                or facts["database_name"] != name
                or facts["role_name"] != "kairos"
                or type(facts["server_version"]) is not int
                or not 160000 <= facts["server_version"] < 170000
                or type(facts["public_objects"]) is not int
                or facts["public_objects"] != 0
            ):
                raise CampaignCIError("FRESH_PG16_TARGET_REQUIRED")
            await database.migrate()
            await database.verify_schema()
    finally:
        await asyncio.wait_for(database.close(), timeout=CLOSE_TIMEOUT_S)


def check_result(*, causal: bool = False) -> None:
    report, target = (CAUSAL_REPORT, CAUSAL_NATIVE_TARGET) if causal else (REPORT, NATIVE_TARGET)
    if report.is_symlink() or not report.is_file() or report.stat().st_size > 1024 * 1024:
        raise CampaignCIError("NATIVE_RESULT_REQUIRED")
    try:
        root = ET.fromstring(report.read_bytes())
    except (ET.ParseError, OSError):
        raise CampaignCIError("INVALID_NATIVE_RESULT") from None
    cases = list(root.iter("testcase"))
    if (
        len(cases) != 1
        or cases[0].get("name") != target
        or any(cases[0].find(tag) is not None for tag in ("skipped", "failure", "error"))
        or any(True for _ in root.iter("failure"))
        or any(True for _ in root.iter("error"))
    ):
        raise CampaignCIError("EXACT_NATIVE_PASS_REQUIRED")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args not in (["prepare"], ["check-result"], ["check-causal-result"]):
        print("CAMPAIGN_NATIVE_CI_FAILED INVALID_COMMAND")
        return 1
    try:
        if args == ["prepare"]:
            asyncio.run(prepare(os.environ.get(ENV_NAME)))
        else:
            check_result(causal=args == ["check-causal-result"])
    except CampaignCIError as exc:
        category = str(exc) if str(exc) in FAILURE_CATEGORIES else "OPERATION_FAILED"
        print(f"CAMPAIGN_NATIVE_CI_FAILED {category}")
        return 1
    except TimeoutError:
        print("CAMPAIGN_NATIVE_CI_FAILED TIMEOUT")
        return 1
    except BaseException:
        print("CAMPAIGN_NATIVE_CI_FAILED OPERATION_FAILED")
        return 1
    print("CAMPAIGN_NATIVE_CI_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
