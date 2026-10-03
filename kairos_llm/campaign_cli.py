"""Explicit verify/reconcile or offline-fixture tick; no keys/provider factory.

The existing database must already have the exact RESEARCH_CAMPAIGN schema.
This command never applies DDL or creates a database. Its in-memory test budget
is permitted only for an independently frozen OFFLINE_ENGINEERING_FIXTURE plan;
it is not an alternative to the shared durable budget for real observations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from kairos_core import canonical_sha256
from kairos_persistence import Database, MigrationProfile, PersistenceSettings
from kairos_persistence.research_campaign import ResearchCampaignRepository, ResearchReviewOutputV1

from .budget import BudgetedLLMGateway
from .campaign import AdaptiveCampaignScheduler
from .config import LLMSettings
from .models import ModelRouter
from .pricing import PriceTable
from .proposals import LLMProposalOutputV1
from .research import ResearchPromptArtifactV1
from .schemas import LLMResult, TokenUsage

_MAX_ARTIFACT_BYTES = 262_144
OFFLINE_NO_INTENT_EVALUATOR_SHA256 = canonical_sha256(
    {"version": "offline-no-intent-evaluator.v1", "output": None, "economic_evidence": False}
)


class _OfflineBudget:
    """Only a fixture ledger. Real paid observations cannot select this class."""

    def __init__(self):
        self.reserved, self.committed = {}, {}

    async def reserve(self, **kwargs):
        key = kwargs["reservation_id"]
        value = (kwargs["provider"], kwargs["reserved_microusd"])
        if key in self.reserved and self.reserved[key] != value:
            raise ValueError("fixture reservation identity conflict")
        self.reserved[key] = value

    async def commit(self, **kwargs):
        key, value = kwargs["reservation_id"], kwargs["actual_microusd"]
        if key not in self.reserved or value > self.reserved[key][1]:
            raise ValueError("fixture commit lacks its exact reservation")
        self.committed[key] = value


class _NoIntentFixture:
    def __init__(self, schedule):
        if schedule.evaluator_sha256 != OFFLINE_NO_INTENT_EVALUATOR_SHA256:
            raise ValueError("offline fixture cannot impersonate another frozen evaluator")
        self.artifact_sha256 = OFFLINE_NO_INTENT_EVALUATOR_SHA256
        self.strategy_id, self.strategy_revision = schedule.strategy_id, schedule.strategy_revision

    async def evaluate(self, **_kwargs):
        return None


class _OfflineGateway:
    def __init__(self):
        # Explicit empty credentials and no .env. No API client is constructed.
        self.settings = LLMSettings(_env_file=None, openai_api_key="", deepseek_api_key="", max_retries=0)
        self.router = ModelRouter()

    async def complete(self, **kwargs):
        route = self.router.resolve(workload=kwargs["workload"])
        payload = json.loads(kwargs["user"].split("\n", 1)[1])
        ids = (payload["sources"][0]["evidence_id"],)
        if kwargs["schema"] is ResearchReviewOutputV1:
            parsed = ResearchReviewOutputV1(
                action="DEFER", rationale="Offline engineering fixture only.", evidence_ids=ids
            )
        else:
            parsed = LLMProposalOutputV1(
                contract_version="kairos-llm-proposal-output.v1",
                action="NO_PROPOSAL",
                rationale="Offline engineering fixture only.",
                evidence_ids=ids,
            )
        content = parsed.model_dump_json()
        usage = TokenUsage(input_tokens=150, output_tokens=30)
        return LLMResult(
            content=content,
            parsed=parsed,
            provider=route.choice.provider.value,
            model=route.choice.model,
            resolved_model=route.choice.model,
            effort=route.effort.value,
            workload=kwargs["workload"].value,
            request_id="fixture-not-a-provider-request",
            usage=usage,
            cost_usd=PriceTable().cost(route.choice.model, usage),
            latency_s=0.0,
        )


def _artifact(path: str) -> dict:
    file = Path(path)
    if not file.is_file() or file.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("campaign prompt artifact must be a bounded existing JSON file")
    payload = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"review_prompt", "proposal_prompt"}:
        raise ValueError("campaign artifact contains only exact frozen prompt objects")
    return payload


async def _run(args):
    url = os.environ.get("KAIROS_RESEARCH_CAMPAIGN_DATABASE_URL")
    if not url:
        raise ValueError("explicit isolated campaign database configuration is missing")
    # Profile and literal name checks happen before the first connection.
    database = Database(
        PersistenceSettings(
            _env_file=None, database_url=url, pool_min_size=1, pool_max_size=1, command_timeout_s=5
        ),
        migration_profile=MigrationProfile.RESEARCH_CAMPAIGN,
    )
    try:
        await database.connect()
        await database.verify_schema()
        repository = ResearchCampaignRepository(database)
        plan, schedule, _ = await repository.load_campaign(args.campaign_id)
        if args.command == "verify":
            return {
                "result": "SOURCE_IDENTITY_ONLY",
                "plan_receipt_sha256": plan.receipt_sha256,
                "recording_mode": plan.recording_mode,
            }
        if args.command == "seal-denominator":
            receipt = await repository.seal_denominator(args.campaign_id)
            return receipt.model_dump(mode="json")
        if not args.prompt_artifact:
            raise ValueError("explicit frozen prompt artifact is required")
        if plan.recording_mode != "OFFLINE_ENGINEERING_FIXTURE":
            raise ValueError("CLI tick/reconcile accepts only preregistered offline engineering fixtures")
        artifact = _artifact(args.prompt_artifact)
        scheduler = AdaptiveCampaignScheduler(
            repository=repository,
            gateway=BudgetedLLMGateway(_OfflineGateway(), _OfflineBudget()),
            evaluator=_NoIntentFixture(schedule),
            review_prompt=ResearchPromptArtifactV1.model_validate(artifact["review_prompt"]),
            proposal_prompt=ResearchPromptArtifactV1.model_validate(artifact["proposal_prompt"]),
        )
        outcomes = await (
            scheduler.run_once(args.campaign_id)
            if args.command == "tick-offline"
            else scheduler.reconcile(args.campaign_id)
        )
        return {
            "result": "OFFLINE_ENGINEERING_ONLY",
            "outcomes": [x.model_dump(mode="json") for x in outcomes],
        }
    finally:
        await database.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("verify", "tick-offline", "reconcile-offline", "seal-denominator")
    )
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--prompt-artifact")
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(_run(args))
        result.update(
            {
                "economic_qualification": False,
                "live_orders_allowed": False,
                "provider_or_venue_calls": 0,
                "strategy_policy": "REJECT_ALL",
            }
        )
    except Exception:
        # No DSN, source payload, provider response or raw exception is emitted.
        print(
            json.dumps(
                {
                    "result": "BLOCKED",
                    "category": "CAMPAIGN_CONFIGURATION_OR_EVIDENCE_UNAVAILABLE",
                    "live_orders_allowed": False,
                    "provider_or_venue_calls": 0,
                }
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
