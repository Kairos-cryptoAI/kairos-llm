"""Credential-safe availability, latency, quota, cost, and contract probes."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .config import LLMSettings
from .gateway import LLMGateway
from .models import DEFAULT_WORKLOAD_ROUTES, LLMWorkload, Provider
from .pricing import PriceTable
from .schemas import LLMResult, TokenUsage

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore


class ProbeStatus(StrEnum):
    PASS = "PASS"  # nosec B105
    BLOCKED = "BLOCKED"
    FAIL = "FAIL"


class ProbePayload(BaseModel):
    protocol: Literal["KAIROS_LLM_PROBE_V1"]
    arithmetic: Literal[42]
    decision: Literal["NO_TRADE"]


@dataclass(frozen=True)
class QuotaObservation:
    provider: str
    status: ProbeStatus
    http_status: int | None
    latency_ms: float | None
    headers: dict[str, str]
    detail: str


@dataclass(frozen=True)
class ModelCallObservation:
    workload: str
    provider: str
    requested_model: str
    requested_effort: str | None
    sample: int
    status: ProbeStatus
    latency_s: float | None
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    resolved_model: str | None
    system_fingerprint: str | None
    detail: str


@dataclass(frozen=True)
class WorkloadSummary:
    workload: str
    provider: str
    requested_model: str
    samples_requested: int
    successful_calls: int
    availability: float
    quality_rate: float
    p95_latency_s: float | None
    estimated_cost_usd: float
    status: ProbeStatus
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class LLMQualificationReport:
    schema_version: int
    generated_at: str
    samples_per_workload: int
    thresholds: dict[str, float]
    quotas: tuple[QuotaObservation, ...]
    calls: tuple[ModelCallObservation, ...]
    workloads: tuple[WorkloadSummary, ...]
    planned_cost_ceiling_usd: float = 0.0
    maximum_planned_cost_usd: float = 0.0
    live_orders_allowed: bool = False

    @property
    def status(self) -> ProbeStatus:
        statuses = {item.status for item in self.quotas} | {item.status for item in self.workloads}
        if ProbeStatus.FAIL in statuses:
            return ProbeStatus.FAIL
        if ProbeStatus.BLOCKED in statuses:
            return ProbeStatus.BLOCKED
        return ProbeStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "samples_per_workload": self.samples_per_workload,
            "thresholds": dict(sorted(self.thresholds.items())),
            "planned_cost_ceiling_usd": self.planned_cost_ceiling_usd,
            "maximum_planned_cost_usd": self.maximum_planned_cost_usd,
            "status": self.status.value,
            "live_orders_allowed": False,
            "quotas": [asdict(item) for item in self.quotas],
            "calls": [asdict(item) for item in self.calls],
            "workloads": [asdict(item) for item in self.workloads],
        }


Runner = Callable[[LLMWorkload], Awaitable[LLMResult]]
QuotaProbe = Callable[[Provider, str], Awaitable[QuotaObservation]]

DEFAULT_THRESHOLDS = {
    "minimum_availability": 1.0,
    "minimum_quality_rate": 1.0,
    "maximum_p95_latency_s": 30.0,
    "maximum_total_estimated_cost_usd": 0.25,
}
QUALIFICATION_MAX_INPUT_TOKENS = 2_048
QUALIFICATION_MAX_OUTPUT_TOKENS = 128
DEFAULT_MAXIMUM_PLANNED_COST_USD = 0.05
QUALIFICATION_SYSTEM_PROMPT = (
    "You are a deterministic API qualification probe. Return the requested JSON object exactly. "
    "Do not use tools and do not add fields."
)
QUALIFICATION_USER_PROMPT = (
    'Return JSON {"protocol":"KAIROS_LLM_PROBE_V1","arithmetic":42,"decision":"NO_TRADE"}.'
)


def _selected_workloads(workloads: Sequence[LLMWorkload] | None) -> tuple[LLMWorkload, ...]:
    selected = tuple(DEFAULT_WORKLOAD_ROUTES) if workloads is None else tuple(workloads)
    if not selected:
        raise ValueError("at least one workload must be selected")
    if any(not isinstance(workload, LLMWorkload) for workload in selected):
        raise ValueError("workloads must contain only LLMWorkload values")
    if len(set(selected)) != len(selected):
        raise ValueError("workloads must not contain duplicates")
    return selected


def planned_cost_ceiling_usd(
    *,
    workloads: Sequence[LLMWorkload] | None,
    samples_per_workload: int,
) -> float:
    """Return a conservative qualification allowance before any provider call.

    Output tokens are hard-capped by the provider request. The input allowance is
    deliberately much larger than the fixed qualification prompt and schema.
    """
    if samples_per_workload <= 0:
        raise ValueError("samples_per_workload must be positive")
    usage = TokenUsage(
        input_tokens=QUALIFICATION_MAX_INPUT_TOKENS,
        output_tokens=QUALIFICATION_MAX_OUTPUT_TOKENS,
    )
    prices = PriceTable()
    per_sample = math.fsum(
        prices.cost(DEFAULT_WORKLOAD_ROUTES[workload].choice.model, usage)
        for workload in _selected_workloads(workloads)
    )
    return per_sample * samples_per_workload


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _quota_headers(headers: Mapping[str, str]) -> dict[str, str]:
    prefixes = ("ratelimit", "x-ratelimit", "retry-after")
    return {key.lower(): str(value) for key, value in headers.items() if key.lower().startswith(prefixes)}


def _safe_error(exc: Exception, secrets: Sequence[str]) -> str:
    rendered = f"{type(exc).__name__}: {exc}"
    for secret in secrets:
        if secret:
            rendered = rendered.replace(secret, "[REDACTED]")
    return rendered[:500]


def _validate_result(result: LLMResult, workload: LLMWorkload) -> None:
    route = DEFAULT_WORKLOAD_ROUTES[workload]
    if result.model != route.choice.model:
        raise ValueError("gateway returned a different requested model")
    if result.workload != workload.value:
        raise ValueError("gateway returned a different workload identity")
    if not isinstance(result.parsed, ProbePayload):
        raise ValueError("gateway did not return the exact qualification schema")
    if result.resolved_model is None or not result.resolved_model.strip():
        raise ValueError("provider omitted resolved model identity")
    usage = result.usage
    if usage.input_tokens <= 0 or usage.output_tokens <= 0:
        raise ValueError("provider omitted positive token usage")
    if usage.cached_input_tokens < 0 or usage.cached_input_tokens > usage.input_tokens:
        raise ValueError("provider returned invalid cached-token usage")
    if not math.isfinite(result.latency_s) or result.latency_s <= 0:
        raise ValueError("gateway returned invalid latency")
    if not math.isfinite(result.cost_usd) or result.cost_usd < 0:
        raise ValueError("gateway returned invalid estimated cost")


async def qualify_llms(
    *,
    samples_per_workload: int,
    available_keys: Mapping[Provider, str | None],
    runner: Runner,
    quota_probe: QuotaProbe,
    workloads: Sequence[LLMWorkload] | None = None,
    thresholds: dict[str, float] | None = None,
    now: datetime | None = None,
    planned_cost_ceiling: float = 0.0,
    maximum_planned_cost: float = 0.0,
) -> LLMQualificationReport:
    if samples_per_workload <= 0:
        raise ValueError("samples_per_workload must be positive")
    policy = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    if policy.keys() != DEFAULT_THRESHOLDS.keys():
        raise ValueError("thresholds must contain the exact qualification policy")
    selected = _selected_workloads(workloads)
    providers = tuple(dict.fromkeys(DEFAULT_WORKLOAD_ROUTES[item].choice.provider for item in selected))
    quotas: list[QuotaObservation] = []
    for provider in providers:
        key = available_keys.get(provider)
        if not key:
            quotas.append(
                QuotaObservation(
                    provider=provider.value,
                    status=ProbeStatus.BLOCKED,
                    http_status=None,
                    latency_ms=None,
                    headers={},
                    detail="provider key file was not supplied; no request was made",
                )
            )
        else:
            try:
                quotas.append(await quota_probe(provider, key))
            except Exception as exc:
                quotas.append(
                    QuotaObservation(
                        provider=provider.value,
                        status=ProbeStatus.FAIL,
                        http_status=None,
                        latency_ms=None,
                        headers={},
                        detail=_safe_error(exc, [key]),
                    )
                )

    calls: list[ModelCallObservation] = []
    inference_quota_headers: dict[Provider, dict[str, str]] = {}
    for workload in selected:
        route = DEFAULT_WORKLOAD_ROUTES[workload]
        key = available_keys.get(route.choice.provider)
        for sample in range(1, samples_per_workload + 1):
            if not key:
                calls.append(
                    ModelCallObservation(
                        workload=workload.value,
                        provider=route.choice.provider.value,
                        requested_model=route.choice.model,
                        requested_effort=route.choice.provider_effort,
                        sample=sample,
                        status=ProbeStatus.BLOCKED,
                        latency_s=None,
                        input_tokens=0,
                        cached_input_tokens=0,
                        output_tokens=0,
                        estimated_cost_usd=0,
                        resolved_model=None,
                        system_fingerprint=None,
                        detail="provider key file was not supplied; no billable request was made",
                    )
                )
                continue
            try:
                result = await runner(workload)
                _validate_result(result, workload)
                inference_quota_headers.setdefault(route.choice.provider, {}).update(
                    result.rate_limit_headers
                )
                calls.append(
                    ModelCallObservation(
                        workload=workload.value,
                        provider=route.choice.provider.value,
                        requested_model=route.choice.model,
                        requested_effort=route.choice.provider_effort,
                        sample=sample,
                        status=ProbeStatus.PASS,
                        latency_s=result.latency_s,
                        input_tokens=result.usage.input_tokens,
                        cached_input_tokens=result.usage.cached_input_tokens,
                        output_tokens=result.usage.output_tokens,
                        estimated_cost_usd=result.cost_usd,
                        resolved_model=result.resolved_model,
                        system_fingerprint=result.system_fingerprint,
                        detail="exact structured contract validated",
                    )
                )

            except Exception as exc:
                calls.append(
                    ModelCallObservation(
                        workload=workload.value,
                        provider=route.choice.provider.value,
                        requested_model=route.choice.model,
                        requested_effort=route.choice.provider_effort,
                        sample=sample,
                        status=ProbeStatus.FAIL,
                        latency_s=None,
                        input_tokens=0,
                        cached_input_tokens=0,
                        output_tokens=0,
                        estimated_cost_usd=0,
                        resolved_model=None,
                        system_fingerprint=None,
                        detail=_safe_error(exc, [key]),
                    )
                )

    successful_providers = {Provider(item.provider) for item in calls if item.status is ProbeStatus.PASS}
    reconciled_quotas: list[QuotaObservation] = []
    for observation in quotas:
        provider = Provider(observation.provider)
        inference_headers = inference_quota_headers.get(provider, {})
        if inference_headers:
            reconciled_quotas.append(
                replace(
                    observation,
                    status=ProbeStatus.PASS,
                    headers=dict(sorted(inference_headers.items())),
                    detail=f"observed inference quota headers: {sorted(inference_headers)}",
                )
            )
        elif observation.status is ProbeStatus.FAIL and provider in successful_providers:
            reconciled_quotas.append(
                replace(
                    observation,
                    status=ProbeStatus.BLOCKED,
                    detail=(
                        "inference authentication succeeded but effective quota headers remain unverified"
                    ),
                )
            )
        else:
            reconciled_quotas.append(observation)
    quotas = reconciled_quotas

    summaries: list[WorkloadSummary] = []
    for workload in selected:
        route = DEFAULT_WORKLOAD_ROUTES[workload]
        observations = [item for item in calls if item.workload == workload.value]
        successful = [item for item in observations if item.status is ProbeStatus.PASS]
        availability = len(successful) / samples_per_workload
        quality_rate = len(successful) / samples_per_workload
        latency = _percentile([item.latency_s for item in successful if item.latency_s is not None], 0.95)
        cost = math.fsum(item.estimated_cost_usd for item in observations)
        reasons: list[str] = []
        if not available_keys.get(route.choice.provider):
            reasons.append("provider_key_missing")
        if availability < policy["minimum_availability"]:
            reasons.append("availability_below_threshold")
        if quality_rate < policy["minimum_quality_rate"]:
            reasons.append("quality_below_threshold")
        if latency is None or latency > policy["maximum_p95_latency_s"]:
            reasons.append("latency_above_threshold")
        status = (
            ProbeStatus.PASS
            if not reasons
            else ProbeStatus.BLOCKED
            if reasons
            == [
                "provider_key_missing",
                "availability_below_threshold",
                "quality_below_threshold",
                "latency_above_threshold",
            ]
            else ProbeStatus.FAIL
        )
        summaries.append(
            WorkloadSummary(
                workload=workload.value,
                provider=route.choice.provider.value,
                requested_model=route.choice.model,
                samples_requested=samples_per_workload,
                successful_calls=len(successful),
                availability=availability,
                quality_rate=quality_rate,
                p95_latency_s=latency,
                estimated_cost_usd=cost,
                status=status,
                reasons=tuple(reasons),
            )
        )
    total_cost = math.fsum(item.estimated_cost_usd for item in calls)
    if total_cost > policy["maximum_total_estimated_cost_usd"]:
        summaries = [
            WorkloadSummary(
                **{
                    **asdict(item),
                    "status": ProbeStatus.FAIL,
                    "reasons": (*item.reasons, "total_cost_above_threshold"),
                }
            )
            for item in summaries
        ]
    return LLMQualificationReport(
        schema_version=2,
        generated_at=(now or datetime.now(UTC)).astimezone(UTC).isoformat(),
        samples_per_workload=samples_per_workload,
        thresholds=policy,
        quotas=tuple(quotas),
        calls=tuple(calls),
        workloads=tuple(summaries),
        planned_cost_ceiling_usd=planned_cost_ceiling,
        maximum_planned_cost_usd=maximum_planned_cost,
    )


async def qualify_live_llms(
    *,
    openai_api_key: str | None,
    deepseek_api_key: str | None,
    samples_per_workload: int,
    workloads: Sequence[LLMWorkload] | None = None,
    maximum_planned_cost_usd: float = DEFAULT_MAXIMUM_PLANNED_COST_USD,
) -> LLMQualificationReport:
    if not math.isfinite(maximum_planned_cost_usd) or maximum_planned_cost_usd <= 0:
        raise ValueError("maximum_planned_cost_usd must be finite and positive")
    selected = _selected_workloads(workloads)
    planned_cost = planned_cost_ceiling_usd(
        workloads=selected,
        samples_per_workload=samples_per_workload,
    )
    if planned_cost > maximum_planned_cost_usd:
        raise ValueError(
            f"planned qualification cost ceiling ${planned_cost:.8f} exceeds "
            f"the configured ${maximum_planned_cost_usd:.8f} limit"
        )
    settings = LLMSettings(
        openai_api_key=openai_api_key,
        deepseek_api_key=deepseek_api_key,
        max_retries=0,
        max_output_tokens=QUALIFICATION_MAX_OUTPUT_TOKENS,
        request_timeout_s=30,
    )
    gateway = LLMGateway(settings)

    async def runner(workload: LLMWorkload) -> LLMResult:
        return await gateway.complete(
            system=QUALIFICATION_SYSTEM_PROMPT,
            user=QUALIFICATION_USER_PROMPT,
            workload=workload,
            schema=ProbePayload,
        )

    async def quota_probe(provider: Provider, key: str) -> QuotaObservation:
        if aiohttp is None:  # pragma: no cover
            raise RuntimeError("aiohttp is required for quota qualification")
        base = (
            "https://api.openai.com/v1"
            if provider is Provider.OPENAI and not settings.openai_base_url
            else settings.openai_base_url
            if provider is Provider.OPENAI
            else settings.deepseek_base_url
        )
        if base is None:  # pragma: no cover - guarded by the configured defaults
            raise ValueError(f"{provider.value} base URL is missing")
        url = f"{base.rstrip('/')}/models"
        timeout = aiohttp.ClientTimeout(total=settings.request_timeout_s)
        started = asyncio.get_running_loop().time()
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers={"Authorization": f"Bearer {key}"}) as response:
                await response.read()
                latency_ms = (asyncio.get_running_loop().time() - started) * 1000
                headers = _quota_headers(response.headers)
                if response.status != 200:
                    return QuotaObservation(
                        provider.value,
                        ProbeStatus.FAIL,
                        response.status,
                        latency_ms,
                        headers,
                        "authenticated model-list probe did not return HTTP 200",
                    )
                return QuotaObservation(
                    provider.value,
                    ProbeStatus.PASS if headers else ProbeStatus.BLOCKED,
                    response.status,
                    latency_ms,
                    headers,
                    (
                        f"observed quota headers: {sorted(headers)}"
                        if headers
                        else "provider emitted no quota headers; effective quota is unverified"
                    ),
                )

    try:
        return await qualify_llms(
            samples_per_workload=samples_per_workload,
            available_keys={
                Provider.OPENAI: openai_api_key,
                Provider.DEEPSEEK: deepseek_api_key,
            },
            runner=runner,
            quota_probe=quota_probe,
            workloads=selected,
            planned_cost_ceiling=planned_cost,
            maximum_planned_cost=maximum_planned_cost_usd,
        )
    finally:
        await gateway.close()


def _read_secret(path: Path | None, name: str) -> str | None:
    if path is None:
        return None
    value = path.resolve().read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"{name} key file is empty")
    return value


def _write_report(path: Path, report: LLMQualificationReport, *, overwrite: bool) -> None:
    resolved = path.resolve()
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite LLM qualification report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qualify all Kairos LLM workload routes")
    parser.add_argument("--openai-key-file", type=Path)
    parser.add_argument("--deepseek-key-file", type=Path)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument(
        "--workload",
        action="append",
        choices=[item.value for item in LLMWorkload],
        help="qualify only this workload; repeat to select more than one",
    )
    parser.add_argument(
        "--maximum-planned-cost-usd",
        type=float,
        default=DEFAULT_MAXIMUM_PLANNED_COST_USD,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = asyncio.run(
        qualify_live_llms(
            openai_api_key=_read_secret(args.openai_key_file, "OpenAI"),
            deepseek_api_key=_read_secret(args.deepseek_key_file, "DeepSeek"),
            samples_per_workload=args.samples,
            workloads=(tuple(LLMWorkload(item) for item in args.workload) if args.workload else None),
            maximum_planned_cost_usd=args.maximum_planned_cost_usd,
        )
    )
    _write_report(args.output, report, overwrite=args.overwrite)
    print(f"LLM qualification: {report.status.value}; live_orders_allowed=false")
    return 0 if report.status is ProbeStatus.PASS else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
