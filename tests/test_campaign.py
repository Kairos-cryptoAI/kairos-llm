"""Offline SQL-shaped tests exercise the real opt-in repository and scheduler.

The connection double does not prove PostgreSQL DDL/trigger behavior. Native
schema/permissions/concurrency acceptance must run in an owned disposable DB.
No provider, runtime DB, order or economic result is produced by these fixtures.
"""

import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import pytest
from kairos_core import (
    AdaptiveCandidateProtocolV1,
    ExitPlanV1,
    LLMProposalAdaptiveCandidateArmV1,
    ResearchObservationScheduleV1,
    ResearchObservationWindowV1,
    Side,
    StrategyIntentV1,
    StrategyOnlyAdaptiveCandidateArmV1,
    StrategyProvenanceV1,
    StrategyReviewAdaptiveCandidateArmV1,
    canonical_sha256,
)
from kairos_persistence import (
    Database,
    MessageIdentityConflict,
    MigrationProfile,
    PersistenceSettings,
    ResearchAdaptiveCandidateProtocolRepository,
    ResearchObservationScheduleRepository,
)
from kairos_persistence.database_target import connect_verified_database, require_database_target_url
from kairos_persistence.research_campaign import (
    ResearchArmOutcomeV1,
    ResearchCampaignPlanV1,
    ResearchCampaignRepository,
    ResearchCaptureRequirementV1,
    ResearchReviewOutputV1,
)

from kairos_llm.budget import BudgetedLLMGateway
from kairos_llm.campaign import CAMPAIGN_SCHEDULER_SHA256, AdaptiveCampaignScheduler
from kairos_llm.campaign_cli import OFFLINE_NO_INTENT_EVALUATOR_SHA256, _NoIntentFixture, main
from kairos_llm.errors import LLMBudgetError, LLMTimeout
from kairos_llm.models import LLMWorkload, ModelRouter
from kairos_llm.pricing import PriceTable
from kairos_llm.proposals import LLMProposalOutputV1
from kairos_llm.research import RESEARCH_INPUT_FEATURE_SHA256, ResearchPromptArtifactV1
from kairos_llm.schemas import LLMResult, TokenUsage

T0 = 1_790_064_000_000


class _Connection:
    def __init__(self, schedule):
        self.now, self.schedule = T0 - 10_000, schedule
        self.rows, self.sql = {}, []
        self.lock = asyncio.Lock()
        self.fail_terminal = False

    async def execute(self, sql, *args):
        self.sql.append(sql)
        if sql.startswith("SELECT pg_advisory"):
            return
        match = re.match(r"INSERT INTO (\w+)\(([^)]+)\)", sql)
        assert match, sql
        table, columns = match.groups()
        row = dict(zip(columns.split(","), args, strict=True))
        if self.fail_terminal and row.get("kind") == "terminal":
            raise RuntimeError("injected receipt storage outage")
        self.rows.setdefault(table, []).append(row)

    async def fetchval(self, sql, *args):
        assert "clock_timestamp" in sql, sql
        return self.now

    def select(self, sql, args):
        table = re.search(r"FROM (\w+)", sql).group(1)
        rows = list(self.rows.get(table, []))
        for field, pos in re.findall(r"(?<![\w.])(\w+)=\$(\d+)", sql):
            rows = [x for x in rows if x.get(field) == args[int(pos) - 1]]
        for field, literal in re.findall(r"(?<![\w.])(\w+)='([^']+)'", sql):
            rows = [x for x in rows if x.get(field) == literal]
        if "ORDER BY" in sql:
            order = sql.split("ORDER BY", 1)[1].split(" LIMIT", 1)[0].split(",")
            rows.sort(key=lambda x: tuple(x.get(k.strip(), "") for k in order))
        return rows

    async def fetchrow(self, sql, *args):
        if "LEFT JOIN sim_adaptive_window_claims" in sql:
            claimed = {x["sample_id"] for x in self.rows.get("sim_adaptive_window_claims", [])}
            due = [x for x in self.schedule.windows if x.market_as_of_ts_ms <= args[1]]
            return next(({"sample_id": x.sample_id} for x in due if x.sample_id not in claimed), None)
        rows = self.select(sql, args)
        return rows[0] if rows else None

    async def fetch(self, sql, *args):
        if "SELECT c.*" in sql:
            completed = {}
            for row in self.rows.get("sim_adaptive_campaign_receipts", []):
                if row["kind"] == "outcome":
                    completed[row["sample_id"]] = completed.get(row["sample_id"], 0) + 1
            return [
                x
                for x in self.rows.get("sim_adaptive_window_claims", [])
                if completed.get(x["sample_id"], 0) < 3
            ][: args[1]]
        return self.select(sql, args)


class _Pool:
    def __init__(self, connection):
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


class _Database(Database):
    def __init__(self, connection):
        super().__init__(
            PersistenceSettings(
                _env_file=None, database_url="postgresql://fixture@127.0.0.1/kairos_sim_campaign_fixture"
            ),
            migration_profile=MigrationProfile.RESEARCH_CAMPAIGN,
        )
        self.connection = connection

    @property
    def pool(self):
        return _Pool(self.connection)

    @asynccontextmanager
    async def transaction(self):
        async with self.connection.lock:
            yield self.connection


class _Repository(ResearchCampaignRepository):
    # Only frozen identity resolution is replaced; all campaign SQL algorithms,
    # canonical decoding, causal recording, claims and replay stay real.
    def __init__(self, connection, protocol):
        super().__init__(_Database(connection))
        self.connection, self.protocol = connection, protocol

    async def _identity(self, connection, campaign_id):
        assert campaign_id == connection.schedule.campaign_id
        return connection.schedule, self.protocol


class _Budget:
    def __init__(self):
        self.reserves, self.commits = [], []
        self.deny = False
        self.unknown_reserve = False

    async def reserve(self, **kwargs):
        if self.deny:
            raise LLMBudgetError("fixture budget refused")
        if self.unknown_reserve:
            raise RuntimeError("fixture unknown budget transport failure")
        self.reserves.append(kwargs)

    async def commit(self, **kwargs):
        self.commits.append(kwargs)


class _Gateway:
    def __init__(self, connection):
        self.connection, self.calls = connection, []
        self.settings = SimpleNamespace(max_retries=0, max_output_tokens=2_048)
        self.router = ModelRouter()
        self.review_action, self.proposal_action = "ALLOW", "SHORT_BIAS"
        self.error, self.delay_ms = None, 0

    async def complete(self, **kwargs):
        # Assert durable START really preceded every fixture provider invocation.
        arm = "strategy-review" if kwargs["schema"] is ResearchReviewOutputV1 else "llm-proposal-research"
        assert any(
            x.get("kind") == "start" and x["arm_id"] == arm
            for x in self.connection.rows["sim_adaptive_campaign_receipts"]
        )
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        self.connection.now += self.delay_ms
        route = self.router.resolve(workload=kwargs["workload"])
        payload = json.loads(kwargs["user"].split("\n", 1)[1])
        schema = kwargs["schema"]
        values = dict(
            action=self.review_action if arm == "strategy-review" else self.proposal_action,
            rationale="Offline causal conflict observation, never an order.",
            evidence_ids=(payload["sources"][0]["evidence_id"],),
        )
        if schema is LLMProposalOutputV1:
            values["contract_version"] = "kairos-llm-proposal-output.v1"
        parsed = schema(**values)
        usage = TokenUsage(input_tokens=150, output_tokens=30)
        return LLMResult(
            content=parsed.model_dump_json(),
            parsed=parsed,
            model=route.choice.model,
            resolved_model=route.choice.model,
            provider=route.choice.provider.value,
            request_id="offline-fixture-request",
            effort=route.effort.value,
            workload=kwargs["workload"].value,
            usage=usage,
            cost_usd=PriceTable().cost(route.choice.model, usage),
            latency_s=0.01,
        )


class _Evaluator:
    artifact_sha256 = "b" * 64
    strategy_id, strategy_revision = "baseline", "v1"

    def __init__(self):
        self.no_intent, self.calls, self.error = False, 0, False

    async def evaluate(self, *, window, sources):
        self.calls += 1
        if self.error:
            raise RuntimeError("bounded fixture evaluator failed")
        if self.no_intent:
            return None
        market = next(x for x in sources if x.source_kind == "MARKET_SNAPSHOT")
        return StrategyIntentV1(
            source="offline-fixture",
            strategy_id=self.strategy_id,
            strategy_revision=self.strategy_revision,
            symbol=window.symbol,
            side=Side.LONG,
            decision_ts_ms=window.market_as_of_ts_ms,
            entry_eligible_ts_ms=(window.market_as_of_ts_ms // 60_000 + 1) * 60_000,
            entry_expires_ts_ms=(window.market_as_of_ts_ms // 60_000 + 2) * 60_000,
            reference_price=100.0,
            signal_strength=0.5,
            gross_reward_bps=200.0,
            exit_plan=ExitPlanV1(stop_price=99.0, target_price=102.0, max_holding_ms=1_200_000),
            provenance=StrategyProvenanceV1(
                strategy_code_sha256="1" * 64,
                config_sha256="2" * 64,
                input_window_sha256="3" * 64,
                features_sha256="4" * 64,
                input_bar_sha256s=(market.content_sha256,),
            ),
        )


@pytest.fixture
async def setup():
    schedule = ResearchObservationScheduleV1(
        campaign_id="causal-engineering-fixture",
        strategy_id="baseline",
        strategy_revision="v1",
        source_set_sha256="a" * 64,
        evaluator_sha256="b" * 64,
        windows=(
            ResearchObservationWindowV1(
                sample_id="sample-1",
                symbol="BTCUSDT",
                timeframe="1m",
                market_as_of_ts_ms=T0 + 1_000,
                paired_at_ts_ms=T0 + 5_000,
                sample_deadline_ts_ms=T0 + 10_000,
            ),
        ),
    )
    prompt = ResearchPromptArtifactV1(
        system="Use only independently saved causal evidence.",
        user_prefix="Data:",
        workload=LLMWorkload.AGGREGATOR_NORMAL,
        reasoning_effort="medium",
        provider_effort="medium",
        max_output_tokens=2_048,
    )
    common = dict(
        candidate_revision="v1",
        artifact_sha256="c" * 64,
        input_feature_sha256=RESEARCH_INPUT_FEATURE_SHA256,
        decision_mapping_sha256="d" * 64,
        hypothetical_exit_sha256="e" * 64,
        cost_model_sha256="f" * 64,
    )
    llm = dict(provider="openai", model="gpt-6-luna", prompt_sha256=prompt.prompt_sha256)
    protocol = AdaptiveCandidateProtocolV1(
        campaign_id=schedule.campaign_id,
        schedule_digest=schedule.schedule_digest,
        arms=(
            StrategyOnlyAdaptiveCandidateArmV1(candidate_id="baseline", **common),
            StrategyReviewAdaptiveCandidateArmV1(
                candidate_id="review",
                **common,
                **llm,
                schema_sha256=canonical_sha256(ResearchReviewOutputV1.model_json_schema()),
            ),
            LLMProposalAdaptiveCandidateArmV1(
                candidate_id="proposal",
                **common,
                **llm,
                schema_sha256=canonical_sha256(LLMProposalOutputV1.model_json_schema()),
            ),
        ),
    )
    connection = _Connection(schedule)
    repository = _Repository(connection, protocol)
    plan = ResearchCampaignPlanV1(
        campaign_id=schedule.campaign_id,
        schedule_digest=schedule.schedule_digest,
        candidate_protocol_digest=protocol.protocol_digest,
        recording_mode="OFFLINE_ENGINEERING_FIXTURE",
        scheduler_sha256=CAMPAIGN_SCHEDULER_SHA256,
        required_sources=tuple(
            ResearchCaptureRequirementV1(source_kind=kind, source_name=name, maximum_age_ms=2_000)
            for kind, name in (("MACRO", "macro"), ("MARKET_SNAPSHOT", "bars"), ("NEWS", "news"))
        ),
        maximum_windows_per_tick=1,
        maximum_clock_skew_ms=0,
        maximum_call_seconds=1,
    )
    assert await repository.register(plan)
    connection.now = T0 + 100
    for requirement in plan.required_sources:
        await repository.capture_source(
            campaign_id=plan.campaign_id,
            sample_id="sample-1",
            source_kind=requirement.source_kind,
            source_name=requirement.source_name,
            reference=requirement.source_name,
            source_as_of_ts_ms=T0,
            content={"symbol": "BTCUSDT", "close": 100, "kind": requirement.source_kind},
        )
    connection.now = T0 + 1_100
    underlying, budget, evaluator = _Gateway(connection), _Budget(), _Evaluator()
    scheduler = AdaptiveCampaignScheduler(
        repository=repository,
        gateway=BudgetedLLMGateway(underlying, budget),
        evaluator=evaluator,
        review_prompt=prompt,
        proposal_prompt=prompt,
        clock=lambda: connection.now,
    )
    return SimpleNamespace(
        repository=repository,
        connection=connection,
        scheduler=scheduler,
        underlying=underlying,
        budget=budget,
        evaluator=evaluator,
        plan=plan,
        schedule=schedule,
        protocol=protocol,
    )


@pytest.mark.parametrize("action", ["ALLOW", "VETO", "DEFER"])
async def test_three_matched_arms_independent_costs_and_opposite_proposal(setup, action):
    setup.underlying.review_action = action
    outcomes = await setup.scheduler.run_once(setup.plan.campaign_id)
    assert [x.status for x in outcomes] == ["BASELINE", action, "PROPOSAL"]
    assert len({x.bundle_receipt_sha256 for x in outcomes}) == 1
    assert len({x.evaluation_receipt_sha256 for x in outcomes}) == 1
    assert len(setup.underlying.calls) == len(setup.budget.reserves) == len(setup.budget.commits) == 2
    state = await setup.repository.window_state(setup.plan.campaign_id, "sample-1")
    sample = state[("sample", "llm-proposal-research", "one")].sample
    assert sample.arm_protocol_digest == setup.protocol.arm_digest("llm-proposal-research")
    assert not hasattr(sample, "risk_trade_decision")
    denominator = await setup.repository.seal_denominator(setup.plan.campaign_id)
    assert denominator.expected_outcomes == 3 and denominator.committed_cost_microusd > 0
    assert not denominator.economic_qualification and not denominator.live_orders_allowed
    assert await setup.scheduler.run_once(setup.plan.campaign_id) == ()
    assert len(setup.underlying.calls) == 2  # Duplicate/restart has no provider or reservation.


async def test_no_intent_keeps_proposal_and_no_call_review_in_denominator(setup):
    setup.evaluator.no_intent = True
    outcomes = await setup.scheduler.run_once(setup.plan.campaign_id)
    assert [x.status for x in outcomes] == ["NO_INTENT", "NO_INTENT", "PROPOSAL"]
    assert len(setup.underlying.calls) == 1 and outcomes[1].attempt_id is None


@pytest.mark.parametrize("kind", ["missing", "stale", "late"])
async def test_missing_stale_late_sources_zero_dispatch(setup, kind):
    rows = setup.connection.rows["sim_adaptive_campaign_receipts"]
    news = next(x for x in rows if x["slot_key"] == "NEWS:news")
    rows.remove(news)
    if kind != "missing":
        setup.connection.now = T0 + 1_001 if kind == "late" else T0 + 100
        await setup.repository.capture_source(
            campaign_id=setup.plan.campaign_id,
            sample_id="sample-1",
            source_kind="NEWS",
            source_name="news",
            reference="news",
            source_as_of_ts_ms=T0 if kind == "late" else T0 - 5_000,
            content={"news": kind},
        )
    setup.connection.now = T0 + 1_100
    outcomes = await setup.scheduler.run_once(setup.plan.campaign_id)
    assert {x.status for x in outcomes} == {"SOURCE_MISSING"}
    assert setup.evaluator.calls == len(setup.underlying.calls) == len(setup.budget.reserves) == 0


async def test_future_source_cannot_be_backdated(setup):
    rows = setup.connection.rows["sim_adaptive_campaign_receipts"]
    rows.remove(next(x for x in rows if x["slot_key"] == "NEWS:news"))
    with pytest.raises(ValueError):
        await setup.repository.capture_source(
            campaign_id=setup.plan.campaign_id,
            sample_id="sample-1",
            source_kind="NEWS",
            source_name="news",
            reference="different",
            source_as_of_ts_ms=T0 + 99_000,
            content={"future": True},
        )


async def test_expired_unclaimed_window_is_three_missed_zero_calls(setup):
    setup.connection.now = T0 + 5_100
    outcomes = await setup.scheduler.run_once(setup.plan.campaign_id)
    assert {x.status for x in outcomes} == {"MISSED"}
    assert setup.evaluator.calls == len(setup.underlying.calls) == 0


async def test_claim_crash_is_never_reclaimed_or_reevaluated(setup):
    assert await setup.repository.claim_next(setup.plan.campaign_id)
    assert await setup.scheduler.run_once(setup.plan.campaign_id) == ()
    setup.connection.now = T0 + 5_100
    outcomes = await setup.scheduler.reconcile(setup.plan.campaign_id)
    assert {x.status for x in outcomes} == {"MISSED"}
    assert setup.evaluator.calls == len(setup.underlying.calls) == 0


async def test_two_schedulers_share_one_durable_claim(setup):
    results = await asyncio.gather(
        setup.scheduler.run_once(setup.plan.campaign_id), setup.scheduler.run_once(setup.plan.campaign_id)
    )
    assert sum(map(len, results)) == 3
    assert setup.evaluator.calls == 1 and len(setup.underlying.calls) == 2


async def test_budget_denial_remains_no_dispatch_not_success(setup):
    setup.budget.deny = True
    outcomes = await setup.scheduler.run_once(setup.plan.campaign_id)
    assert [x.status for x in outcomes] == ["BASELINE", "BUDGET_BLOCKED", "BUDGET_BLOCKED"]
    assert not setup.underlying.calls


async def test_unknown_reservation_retains_cost_bound_no_provider_or_retry(setup):
    setup.budget.unknown_reserve = True
    outcomes = await setup.scheduler.run_once(setup.plan.campaign_id)
    assert [x.status for x in outcomes] == ["BASELINE", "UNKNOWN", "UNKNOWN"]
    denominator = await setup.repository.seal_denominator(setup.plan.campaign_id)
    assert denominator.outstanding_reservation_microusd > 0
    assert denominator.unknown_budget_operation_count == 2
    assert not setup.underlying.calls
    await setup.scheduler.run_once(setup.plan.campaign_id)
    assert not setup.underlying.calls


async def test_timeout_actual_failed_cost_held_no_restart_dispatch(setup):
    setup.underlying.error = LLMTimeout("offline fixture timeout")
    outcomes = await setup.scheduler.run_once(setup.plan.campaign_id)
    assert [x.status for x in outcomes] == ["BASELINE", "CALL_FAILED", "CALL_FAILED"]
    denominator = await setup.repository.seal_denominator(setup.plan.campaign_id)
    assert denominator.outstanding_reservation_microusd > 0
    assert denominator.committed_cost_microusd == 0
    await setup.scheduler.run_once(setup.plan.campaign_id)
    assert len(setup.underlying.calls) == 2


async def test_terminal_outage_stays_unknown_actual_cost_preserved(setup):
    setup.connection.fail_terminal = True
    outcomes = await setup.scheduler.run_once(setup.plan.campaign_id)
    assert outcomes[-1].status == "UNKNOWN"
    denominator = await setup.repository.seal_denominator(setup.plan.campaign_id)
    assert denominator.unknown_attempt_count == 1 and denominator.committed_cost_microusd > 0
    setup.connection.fail_terminal = False
    await setup.scheduler.run_once(setup.plan.campaign_id)
    assert len(setup.underlying.calls) == 2


async def test_late_review_response_blocks_new_proposal_without_retry(setup):
    setup.underlying.delay_ms = 5_000
    outcomes = await setup.scheduler.run_once(setup.plan.campaign_id)
    assert [x.status for x in outcomes] == ["BASELINE", "LATE", "MISSED"]
    assert len(setup.underlying.calls) == 1


async def test_evaluator_failure_all_arms_recorded_without_provider(setup):
    setup.evaluator.error = True
    outcomes = await setup.scheduler.run_once(setup.plan.campaign_id)
    assert {x.status for x in outcomes} == {"EVALUATOR_FAILED"}
    assert not setup.underlying.calls


async def test_denominator_cannot_drop_missing_scheduled_arms(setup):
    with pytest.raises(MessageIdentityConflict):
        await setup.repository.seal_denominator(setup.plan.campaign_id)


async def test_forged_success_without_completion_is_rejected(setup):
    claim = await setup.repository.claim_next(setup.plan.campaign_id)
    with pytest.raises(MessageIdentityConflict, match="invents"):
        await setup.repository.record_outcome(
            ResearchArmOutcomeV1(
                campaign_id=setup.plan.campaign_id,
                sample_id="sample-1",
                arm_id="llm-proposal-research",
                claim_id=claim.claim_id,
                status="PROPOSAL",
                observed_at_ts_ms=setup.connection.now,
            )
        )


async def test_preregistration_cannot_adopt_past_window(setup):
    setup.connection.rows["sim_adaptive_campaign_plans"].clear()
    with pytest.raises(MessageIdentityConflict, match="past"):
        await setup.repository.register(setup.plan)


def test_offline_evaluator_cannot_impersonate_arbitrary_artifact():
    with pytest.raises(ValueError, match="impersonate"):
        _NoIntentFixture(SimpleNamespace(evaluator_sha256="a" * 64))
    fixture = _NoIntentFixture(
        SimpleNamespace(
            evaluator_sha256=OFFLINE_NO_INTENT_EVALUATOR_SHA256, strategy_id="fixture", strategy_revision="v1"
        )
    )
    assert fixture.artifact_sha256 == OFFLINE_NO_INTENT_EVALUATOR_SHA256


def test_cli_missing_explicit_db_is_sanitized_zero_calls(monkeypatch, capsys):
    monkeypatch.delenv("KAIROS_RESEARCH_CAMPAIGN_DATABASE_URL", raising=False)
    assert main(["verify", "--campaign-id", "fixture"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"] == "BLOCKED" and payload["provider_or_venue_calls"] == 0


async def test_native_three_arm_scheduler_and_restart_on_explicit_disposable_campaign_db(setup):
    """No migration/provider: parent acceptance must install the exact 027 schema first."""
    url = os.getenv("KAIROS_RESEARCH_CAMPAIGN_TEST_DATABASE_URL")
    if not url:
        pytest.skip("explicit disposable RESEARCH_CAMPAIGN test DB required")
    name = urlsplit(url).path.removeprefix("/")
    prefix = "kairos_sim_test_campaign_"
    if not name.startswith(prefix) or len(name) != len(prefix) + 32:
        raise RuntimeError("refusing non-disposable native scheduler target")
    namespace = UUID(hex=name.removeprefix(prefix))
    if namespace.version != 4 or namespace.hex != name.removeprefix(prefix):
        raise RuntimeError("native campaign requires exact UUID4 database identity")
    require_database_target_url(url, name, local_only=True)
    database = Database(
        PersistenceSettings(
            _env_file=None, database_url=url, pool_min_size=1, pool_max_size=3, command_timeout_s=5
        ),
        migration_profile=MigrationProfile.RESEARCH_CAMPAIGN,
    )
    await connect_verified_database(database, name, local_only=True)
    try:
        await database.verify_schema()
        repository = ResearchCampaignRepository(database)
        now = await repository.clock()
        values = setup.schedule.identity_payload()
        values.update(
            campaign_id="native-three-arm-" + uuid4().hex[:12],
            windows=[
                {
                    **setup.schedule.windows[0].model_dump(mode="json"),
                    "market_as_of_ts_ms": now + 2_000,
                    "paired_at_ts_ms": now + 5_000,
                    "sample_deadline_ts_ms": now + 10_000,
                }
            ],
        )
        schedule = ResearchObservationScheduleV1.model_validate(values)
        protocol = AdaptiveCandidateProtocolV1.model_validate(
            {
                **setup.protocol.identity_payload(),
                "campaign_id": schedule.campaign_id,
                "schedule_digest": schedule.schedule_digest,
            }
        )
        plan = ResearchCampaignPlanV1.model_validate(
            {
                **setup.plan.model_dump(mode="json"),
                "receipt_sha256": None,
                "campaign_id": schedule.campaign_id,
                "schedule_digest": schedule.schedule_digest,
                "candidate_protocol_digest": protocol.protocol_digest,
                "maximum_clock_skew_ms": 2_000,
            }
        )
        await ResearchObservationScheduleRepository(database).register(schedule)
        await ResearchAdaptiveCandidateProtocolRepository(database).register(protocol)
        assert await repository.register(plan)
        for requirement in plan.required_sources:
            await repository.capture_source(
                campaign_id=plan.campaign_id,
                sample_id="sample-1",
                source_kind=requirement.source_kind,
                source_name=requirement.source_name,
                reference=requirement.source_name,
                source_as_of_ts_ms=now,
                content={"symbol": "BTCUSDT", "close": 100},
            )
        await asyncio.sleep(
            max(0, (schedule.windows[0].market_as_of_ts_ms - await repository.clock()) / 1_000)
        )

        class NativeFixtureGateway(_Gateway):
            async def complete(self, **kwargs):
                arm = (
                    "strategy-review"
                    if kwargs["schema"] is ResearchReviewOutputV1
                    else "llm-proposal-research"
                )
                state = await repository.window_state(plan.campaign_id, "sample-1")
                assert any(kind == "start" and current_arm == arm for kind, current_arm, _ in state)
                # Only the response generator uses a double. Admission/fences,
                # independent costs, causal replay and denominator use real PG.
                self.connection.rows = {"sim_adaptive_campaign_receipts": [{"kind": "start", "arm_id": arm}]}
                return await super().complete(**kwargs)

        gateway = NativeFixtureGateway(SimpleNamespace(rows={}, now=0))
        budget = _Budget()
        scheduler = AdaptiveCampaignScheduler(
            repository=repository,
            gateway=BudgetedLLMGateway(gateway, budget),
            evaluator=_Evaluator(),
            review_prompt=setup.scheduler.review_prompt,
            proposal_prompt=setup.scheduler.proposal_prompt,
            clock=lambda: time.time_ns() // 1_000_000,
        )
        results = await scheduler.run_once(plan.campaign_id)
        assert [x.status for x in results] == ["BASELINE", "ALLOW", "PROPOSAL"]
        assert len({x.bundle_receipt_sha256 for x in results}) == 1
        denominator = await repository.seal_denominator(plan.campaign_id)
        assert denominator.expected_outcomes == 3 and denominator.committed_cost_microusd > 0
        assert not denominator.economic_qualification and not denominator.live_orders_allowed
        assert await scheduler.run_once(plan.campaign_id) == ()
        assert len(gateway.calls) == len(budget.reserves) == len(budget.commits) == 2
    finally:
        await database.close()
