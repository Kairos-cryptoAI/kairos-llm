"""Bounded opt-in three-arm orchestration over an actual durable SIM journal.

There is no provider factory, service wiring, order/risk conversion or automatic
retry. A caller must supply a frozen evaluator and an existing budgeted gateway.
Missing/late/unknown observations remain in the scheduled denominator; they are
never rewritten as zero-cost successful model decisions. The denominator is
engineering coverage only and cannot attest economic or PAPER/LIVE readiness.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol, cast

from kairos_core import EvidenceReferenceV1, canonical_sha256
from kairos_persistence.research_campaign import (
    ResearchArmOutcomeV1,
    ResearchCampaignRepository,
    ResearchCostReceiptV1,
    ResearchReviewOutputV1,
    ResearchReviewReceiptV1,
    ResearchWindowClaimV1,
)
from kairos_persistence.research_evidence import ResearchLLMAttemptStartV1, ResearchSourceReceiptV1

from .budget import BudgetedLLMGateway, LLMUsageBudget
from .errors import LLMBudgetError
from .gateway import LLMGateway
from .proposals import LLMProposalOutputV1, proposal_evidence_id
from .research import (
    RESEARCH_INPUT_FEATURE_SHA256,
    ResearchEvidenceError,
    ResearchPromptArtifactV1,
    ResearchProposalCoordinator,
    _AttemptBudget,
    _ObservedGateway,
)

CAMPAIGN_SCHEDULER_SHA256 = canonical_sha256(
    {
        "contract_version": "adaptive-campaign-scheduler.v2",
        "claims": "durable-once-no-reclaim",
        "capture": "db-observed-before-decision-cutoff.v1",
        "arms": "fixed-three-matched",
        "denominator": "all-scheduled-including-missing-late-unknown",
        "backend": "requested-and-resolved-exact-preregistered.v1",
        "completion_clock": "pg-recorded-at-authoritative-with-frozen-local-skew.v1",
        "authority": "SIM_RESEARCH_ONLY",
    }
)


class CampaignBaselineEvaluator(Protocol):
    """Frozen independent evaluator; no default strategy or profitability assertion."""

    artifact_sha256: str
    strategy_id: str
    strategy_revision: str

    async def evaluate(self, *, window, sources: tuple[ResearchSourceReceiptV1, ...]) -> Any: ...


class _IndependentBudgetObserver:
    def __init__(self, budget: LLMUsageBudget, repository: ResearchCampaignRepository, claim, arm_id):
        self.budget, self.repository, self.claim, self.arm_id = budget, repository, claim, arm_id

    async def reserve(self, **kwargs) -> None:
        await self._save(kwargs["reservation_id"], "RESERVATION_REQUESTED", kwargs["reserved_microusd"])
        try:
            await self.budget.reserve(**kwargs)
        except LLMBudgetError:
            await self._save(kwargs["reservation_id"], "RESERVATION_DENIED", 0)
            raise
        await self._save(kwargs["reservation_id"], "RESERVED", kwargs["reserved_microusd"])

    async def commit(self, **kwargs) -> None:
        await self._save(kwargs["reservation_id"], "COMMIT_REQUESTED", kwargs["actual_microusd"])
        await self.budget.commit(**kwargs)
        await self._save(kwargs["reservation_id"], "COMMITTED", kwargs["actual_microusd"])

    async def _save(self, attempt_id: str, stage, amount: int) -> None:
        await self.repository.record_cost(
            ResearchCostReceiptV1(
                campaign_id=self.claim.campaign_id,
                sample_id=self.claim.sample_id,
                arm_id=self.arm_id,
                attempt_id=attempt_id,
                stage=stage,
                amount_microusd=amount,
                observed_at_ts_ms=await self.repository.clock(),
            )
        )


class AdaptiveCampaignScheduler:
    """Actual preregistered bounded executor; claims and receipts live in PostgreSQL.

    run_once does not create a campaign, load credentials or run migrations.
    Its gateway is explicitly injected and must already obey shared budgets,
    frozen routes and max_retries=0. CLI use below is offline-fixture only.
    """

    scheduler_sha256 = CAMPAIGN_SCHEDULER_SHA256

    def __init__(
        self,
        *,
        repository: ResearchCampaignRepository,
        gateway: BudgetedLLMGateway,
        evaluator: CampaignBaselineEvaluator,
        review_prompt: ResearchPromptArtifactV1,
        proposal_prompt: ResearchPromptArtifactV1,
        clock: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(repository, ResearchCampaignRepository):
            raise ValueError("scheduler requires the independently enrolled campaign repository")
        if not isinstance(gateway, BudgetedLLMGateway) or gateway.settings.max_retries != 0:
            raise ValueError("scheduler requires an explicitly injected once-only budgeted gateway")
        self.repository, self.gateway, self.evaluator = repository, gateway, evaluator
        self.review_prompt, self.proposal_prompt = review_prompt, proposal_prompt
        self.clock = clock or (lambda: time.time_ns() // 1_000_000)

    async def run_once(self, campaign_id: str) -> tuple[ResearchArmOutcomeV1, ...]:
        plan, schedule, protocol = await self._validate(campaign_id)
        outcomes: list[ResearchArmOutcomeV1] = []
        # Restart accounting is read-only with respect to evaluator/provider.
        # A running prior call may finish late; its actual costs/terminal remain
        # append-only even when this at-cutoff outcome is UNKNOWN.
        outcomes.extend(await self.reconcile(campaign_id))
        for _ in range(plan.maximum_windows_per_tick):
            claim = await self.repository.claim_next(campaign_id)
            if claim is None:
                break
            window = next(x for x in schedule.windows if x.sample_id == claim.sample_id)
            if claim.claimed_at_ts_ms >= window.paired_at_ts_ms:
                outcomes.extend(await self._finalize(claim, fallback="MISSED"))
                continue
            try:
                bundle = await self.repository.freeze_bundle(claim)
                sources = await self.repository.resolve_causal_sources(
                    campaign_id=campaign_id,
                    sample_id=claim.sample_id,
                    source_receipt_sha256s=bundle.source_receipt_sha256s,
                )
            except Exception:
                outcomes.extend(await self._finalize(claim, fallback="SOURCE_MISSING"))
                continue
            try:
                async with asyncio.timeout(plan.maximum_call_seconds):
                    # Frozen models contain mutable nested JSON. A transform in
                    # the evaluator must never alter another arm's source bytes.
                    result = await self._evaluate_window(
                        window=window, sources=tuple(source.model_copy(deep=True) for source in sources)
                    )
                evaluation = await self._record_evaluation(claim, bundle, result)
            except asyncio.CancelledError:
                raise
            except Exception:
                outcomes.extend(await self._finalize(claim, fallback="EVALUATOR_FAILED"))
                continue
            if await self._evaluation_is_late(evaluation, window):
                outcomes.extend(await self._finalize(claim, fallback="LATE"))
                continue
            if evaluation.intent is not None:
                await self._review(claim, bundle, evaluation, sources, plan, schedule, protocol)
            if await self.repository.clock() >= window.paired_at_ts_ms:
                outcomes.extend(await self._finalize(claim, fallback="MISSED"))
                continue
            # A proposal is still observed when strategy emits NO_INTENT. The
            # disagreement is preserved, not promoted into an executable trade.
            proposal_gateway = self._gateway(claim, "llm-proposal-research")
            coordinator = ResearchProposalCoordinator(proposal_gateway, self.repository, clock=self.clock)
            try:
                async with asyncio.timeout(plan.maximum_call_seconds):
                    observation = await coordinator.observe(
                        schedule=schedule,
                        protocol=protocol,
                        sample_id=claim.sample_id,
                        prompt=self.proposal_prompt,
                        source_receipt_sha256s=bundle.source_receipt_sha256s,
                        workload=self.proposal_prompt.workload,
                    )
                if (
                    observation.terminal is not None
                    and not observation.unresolved
                    and observation.terminal.observed_at_ts_ms <= window.paired_at_ts_ms
                    and await self.repository.decision_recorded_at(
                        campaign_id=claim.campaign_id,
                        sample_id=claim.sample_id,
                        arm_id="llm-proposal-research",
                        attempt_id=observation.start.attempt_id,
                    )
                    <= window.paired_at_ts_ms
                ):
                    await self._replay_proposal_sample(
                        coordinator,
                        observation,
                        schedule=schedule,
                        protocol=protocol,
                        sample_id=claim.sample_id,
                        prompt=self.proposal_prompt,
                        source_receipt_sha256s=bundle.source_receipt_sha256s,
                        workload=self.proposal_prompt.workload,
                        evaluation_receipt_sha256=str(evaluation.receipt_sha256),
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                # Durable START determines UNKNOWN vs no-dispatch budget denial;
                # no exception text/body is stored and no retry is attempted.
                outcomes.extend(await self._finalize(claim, fallback="BUDGET_BLOCKED"))
                continue
            outcomes.extend(await self._finalize(claim, fallback="BUDGET_BLOCKED"))
        return tuple(outcomes)

    async def reconcile(self, campaign_id: str) -> tuple[ResearchArmOutcomeV1, ...]:
        """Account expired claims without ever evaluating or calling a gateway."""
        plan, schedule, _ = await self.repository.load_campaign(campaign_id)
        if plan.scheduler_sha256 != self.scheduler_sha256:
            raise ResearchEvidenceError("scheduler artifact differs from preregistered version")
        now = await self.repository.clock()
        outcomes: list[ResearchArmOutcomeV1] = []
        for claim in await self.repository.pending_claims(campaign_id, limit=plan.maximum_windows_per_tick):
            window = next(x for x in schedule.windows if x.sample_id == claim.sample_id)
            if now >= window.paired_at_ts_ms:
                outcomes.extend(await self._finalize(claim, fallback="MISSED"))
        return tuple(outcomes)

    async def _validate(self, campaign_id):
        plan, schedule, protocol = await self.repository.load_campaign(campaign_id)
        if plan.scheduler_sha256 != self.scheduler_sha256:
            raise ResearchEvidenceError("scheduler artifact differs from preregistered version")
        if (self.evaluator.artifact_sha256, self.evaluator.strategy_id, self.evaluator.strategy_revision) != (
            schedule.evaluator_sha256,
            schedule.strategy_id,
            schedule.strategy_revision,
        ):
            raise ResearchEvidenceError("evaluator differs from independently frozen source identity")
        for arm, prompt, schema in (
            (protocol.arms[1], self.review_prompt, ResearchReviewOutputV1),
            (protocol.arms[2], self.proposal_prompt, LLMProposalOutputV1),
        ):
            route = self.gateway.router.resolve(workload=prompt.workload)
            if (
                arm.prompt_sha256,
                arm.schema_sha256,
                arm.input_feature_sha256,
                arm.provider,
                arm.model,
                prompt.reasoning_effort,
                prompt.provider_effort,
                prompt.max_output_tokens,
            ) != (
                prompt.prompt_sha256,
                canonical_sha256(schema.model_json_schema()),
                self._input_feature_sha256(arm.arm_id),
                route.choice.provider.value,
                route.choice.model,
                route.effort.value,
                route.choice.provider_effort,
                min(route.max_output_tokens, self.gateway.settings.max_output_tokens),
            ):
                raise ResearchEvidenceError("research arm route/prompt/schema differs from preregistration")
        return plan, schedule, protocol

    def _input_feature_sha256(self, arm_id: str) -> str:
        return RESEARCH_INPUT_FEATURE_SHA256

    async def _evaluate_window(self, *, window, sources):
        return await self.evaluator.evaluate(window=window, sources=sources)

    async def _evaluation_is_late(self, evaluation, window):
        return evaluation.evaluated_at_ts_ms > window.paired_at_ts_ms

    async def _record_evaluation(self, claim, bundle, intent):
        return await self.repository.record_evaluation(claim=claim, bundle=bundle, intent=intent)

    async def _replay_proposal_sample(self, coordinator, observation, **kwargs):
        return await coordinator.replay_sample(**kwargs)

    def _review_input_metadata(self, evaluation, sources, window) -> dict[str, Any]:
        return {}

    def _gateway(self, claim, arm_id) -> BudgetedLLMGateway:
        return BudgetedLLMGateway(
            self.gateway.gateway,
            _IndependentBudgetObserver(self.gateway.budget, self.repository, claim, arm_id),
            prices=self.gateway.prices,
            monthly_budgets_microusd=self.gateway.monthly_budgets_microusd,
        )

    async def _review(self, claim, bundle, evaluation, sources, plan, schedule, protocol) -> None:
        arm = protocol.arms[1]
        window = next(x for x in schedule.windows if x.sample_id == claim.sample_id)
        attempt_id = "kairos-review:" + canonical_sha256(
            {
                "plan_receipt_sha256": plan.receipt_sha256,
                "sample_id": claim.sample_id,
                "bundle_receipt_sha256": bundle.receipt_sha256,
                "evaluation_receipt_sha256": evaluation.receipt_sha256,
                "prompt_sha256": self.review_prompt.prompt_sha256,
            }
        )
        # A terminal, review or bare START is a permanent no-resend fence.
        if await self.repository.find_attempt(attempt_id) is not None:
            return
        references = tuple(
            EvidenceReferenceV1(
                kind=x.source_kind,
                reference=x.reference,
                content_sha256=x.content_sha256,
                observed_at_ms=x.observed_at_ts_ms,
            )
            for x in sources
        )
        evidence_ids = {proposal_evidence_id(x) for x in references}
        payload = {
            "contract_version": "adaptive-review-input.v1",
            "bundle_receipt_sha256": bundle.receipt_sha256,
            "evaluation_receipt_sha256": evaluation.receipt_sha256,
            "strategy_intent": evaluation.intent.to_payload(),
            "sources": [
                {
                    "receipt_sha256": x.receipt_sha256,
                    "evidence_id": proposal_evidence_id(ref),
                    "evidence": ref.model_dump(mode="json"),
                    "content": x.content,
                }
                for x, ref in zip(sources, references, strict=True)
            ],
            **self._review_input_metadata(evaluation, sources, window),
        }
        user = (
            self.review_prompt.user_prefix
            + "\n"
            + json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        )

        def make_start(started):
            return ResearchLLMAttemptStartV1(
                attempt_id=attempt_id,
                budget_reservation_id=attempt_id,
                campaign_id=claim.campaign_id,
                sample_id=claim.sample_id,
                arm_id="strategy-review",
                schedule_digest=schedule.schedule_digest,
                candidate_protocol_digest=protocol.protocol_digest,
                arm_protocol_digest=protocol.arm_digest("strategy-review"),
                symbol=window.symbol,
                timeframe=window.timeframe,
                market_as_of_ts_ms=window.market_as_of_ts_ms,
                market_snapshot_sha256=bundle.market_snapshot_sha256,
                sample_deadline_ts_ms=window.sample_deadline_ts_ms,
                provider=arm.provider,
                requested_model=arm.model,
                prompt_sha256=self.review_prompt.prompt_sha256,
                attempt_started_at_ts_ms=started,
            )

        budget = _AttemptBudget(
            _IndependentBudgetObserver(self.gateway.budget, self.repository, claim, "strategy-review"),
            self.repository,
            make_start,
            self.clock,
        )
        observed = _ObservedGateway(self.gateway.gateway, self.clock)
        gateway = BudgetedLLMGateway(
            cast(LLMGateway, observed),
            budget,
            prices=self.gateway.prices,
            monthly_budgets_microusd=self.gateway.monthly_budgets_microusd,
        )
        try:
            async with asyncio.timeout(plan.maximum_call_seconds):
                result = await gateway.complete(
                    system=self.review_prompt.system,
                    user=user,
                    workload=self.review_prompt.workload,
                    schema=ResearchReviewOutputV1,
                )
            if budget.start is None or observed.observed_at_ts_ms is None:
                raise ResearchEvidenceError("review lacks independently observed dispatch/response bounds")
            output = ResearchReviewOutputV1.model_validate_json(result.content)
            if set(output.evidence_ids) - evidence_ids or result.parsed != output:
                raise ResearchEvidenceError(
                    "review cited unprovided evidence or inconsistent parsed response"
                )
            if result.resolved_model != arm.model or not result.request_id:
                raise ResearchEvidenceError("review completion differs from frozen model/request identity")
            result = replace(
                result,
                budget_reservation_id=attempt_id,
                attempt_started_at_ts_ms=budget.start.attempt_started_at_ts_ms,
                response_observed_at_ts_ms=observed.observed_at_ts_ms,
            )
            await self.repository.record_review(
                ResearchReviewReceiptV1(
                    campaign_id=claim.campaign_id,
                    sample_id=claim.sample_id,
                    attempt_id=attempt_id,
                    start_receipt_sha256=budget.start.receipt_sha256,
                    bundle_receipt_sha256=bundle.receipt_sha256,
                    evaluation_receipt_sha256=evaluation.receipt_sha256,
                    output=output,
                    response_sha256=hashlib.sha256(result.content.encode("utf-8")).hexdigest(),
                    requested_model=result.model,
                    resolved_model=result.resolved_model,
                    request_id=result.request_id,
                    prompt_sha256=self.review_prompt.prompt_sha256,
                    observed_at_ts_ms=observed.observed_at_ts_ms,
                )
            )
        except BaseException as exc:
            if budget.start is not None:
                coordinator = ResearchProposalCoordinator(self.gateway, self.repository, clock=self.clock)
                terminal = coordinator._failure_terminal(budget.start, observed, exc)
                try:
                    await coordinator._finish(terminal)
                except ResearchEvidenceError:
                    # Unrecordable/ambiguous completion leaves the durable START
                    # and observed costs intact. Account UNKNOWN, never resend.
                    pass
            if isinstance(exc, asyncio.CancelledError) or not isinstance(exc, Exception):
                raise

    async def _finalize(
        self, claim: ResearchWindowClaimV1, *, fallback: str
    ) -> tuple[ResearchArmOutcomeV1, ...]:
        """Reconstruct statuses from independent facts, never accept caller model outcomes."""
        _, schedule, _ = await self.repository.load_campaign(claim.campaign_id)
        window = next(x for x in schedule.windows if x.sample_id == claim.sample_id)
        # window_state independently decodes each tagged receipt to its exact
        # model; heterogeneous tagged lookup is intentionally typed as Any here.
        state: dict[tuple[str, str, str], Any] = cast(
            dict[tuple[str, str, str], Any],
            await self.repository.window_state(claim.campaign_id, claim.sample_id),
        )
        existing = {x.arm_id: x for x in await self.repository.outcomes(claim.campaign_id, claim.sample_id)}
        bundle = state.get(("bundle", "all", "one"))
        evaluation = state.get(("evaluation", "all", "one"))
        results = []
        for arm_id in ("strategy-only", "strategy-review", "llm-proposal-research"):
            if arm_id in existing:
                results.append(existing[arm_id])
                continue
            status = fallback
            attempt = next(
                (x for (kind, arm, _), x in state.items() if kind == "start" and arm == arm_id), None
            )
            decision = None
            causal = state.get(("sample", arm_id, "one"))
            if evaluation is not None and await self._evaluation_is_late(evaluation, window):
                status = "LATE"
            elif arm_id == "strategy-only" and evaluation is not None:
                status = "BASELINE" if evaluation.intent is not None else "NO_INTENT"
            elif (
                arm_id == "strategy-review"
                and evaluation is not None
                and evaluation.intent is None
                and attempt is None
            ):
                status = "NO_INTENT"
            elif attempt is not None:
                terminal = state.get(("terminal", arm_id, attempt.attempt_id))
                review = state.get(("review", arm_id, attempt.attempt_id))
                decision = review or terminal
                if decision is None or (terminal is not None and terminal.terminal_status == "UNRESOLVED"):
                    status = "UNKNOWN"
                elif (
                    decision.observed_at_ts_ms > window.paired_at_ts_ms
                    or (
                        await self.repository.decision_recorded_at(
                            campaign_id=claim.campaign_id,
                            sample_id=claim.sample_id,
                            arm_id=arm_id,
                            attempt_id=attempt.attempt_id,
                        )
                    )
                    > window.paired_at_ts_ms
                ):
                    status = "LATE"
                elif review is not None:
                    status = review.output.action
                elif terminal is not None and terminal.terminal_status == "FAILED":
                    status = "CALL_FAILED"
                else:
                    status = "PROPOSAL"
            elif await self.repository.clock() >= window.paired_at_ts_ms and status == "BUDGET_BLOCKED":
                status = "MISSED"
            if attempt is None and arm_id != "strategy-only":
                cost_stages = {
                    x.stage for (kind, arm, _), x in state.items() if kind == "cost" and arm == arm_id
                }
                if "RESERVATION_REQUESTED" in cost_stages and "RESERVATION_DENIED" not in cost_stages:
                    status = "UNKNOWN"
            outcome = ResearchArmOutcomeV1(
                campaign_id=claim.campaign_id,
                sample_id=claim.sample_id,
                arm_id=arm_id,
                claim_id=claim.claim_id,
                status=status,
                bundle_receipt_sha256=bundle.receipt_sha256 if bundle is not None else None,
                evaluation_receipt_sha256=evaluation.receipt_sha256 if evaluation is not None else None,
                attempt_id=attempt.attempt_id if attempt is not None else None,
                decision_receipt_sha256=decision.receipt_sha256 if decision is not None else None,
                causal_sample_receipt_sha256=causal.receipt_sha256 if causal is not None else None,
                observed_at_ts_ms=await self.repository.clock(),
            )
            await self.repository.record_outcome(outcome)
            results.append(outcome)
        return tuple(results)
