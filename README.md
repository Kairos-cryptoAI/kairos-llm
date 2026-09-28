# kairos-llm

Paid qualification requires the pinned `qualification` extra
(`uv sync --locked --extra qualification`) and an already reconciled
`kairos-dev-qualification-v1` database campaign. `kairos-llm-qualify` uses the
shared durable provider budget before every inference and never auto-registers
an allowance or migrates a database. Programmatic `qualify_live_llms` requires
an injected `LLMUsageBudget`; missing budget is a startup error. Existing
per-run planned limits remain additional bounds, not fresh monthly allowances.

The Kairos LLM gateway is the single provider boundary for Text Scouts, Aggregator and Macro
Strategist. It owns role-aware routing, strict structured output, token-cost accounting,
transient retries and health telemetry; no analytical service talks to a provider directly.

## Role routes

Routing is explicit by workload so two components cannot become coupled merely because they use
the same logical reasoning effort.

| workload | model | provider mode | price / 1M input · cached · cache write · output |
| --- | --- | --- | --- |
| `TEXT_SCOUTS` | `deepseek-flash` | non-thinking | $0.30 · $0.006 · $0.30 · $1.20 peak |
| `AGGREGATOR_NORMAL` | `gpt-6-luna` | `medium` | $0.10 · $0.01 · $0.125 · $0.50 |
| `AGGREGATOR_CONFLICT` | `gpt-6-sol` | `high` | $2.00 · $0.20 · $2.50 · $10.00 |
| `MACRO_STRATEGIST` | `gpt-6-sol` | `xhigh` | $2.00 · $0.20 · $2.50 · $10.00 |

The original effort-only API remains supported and maps `low`, `medium`, `high`, and `xhigh` to
the same four routes. New callers should provide `LLMWorkload`; workload overrides and legacy
effort overrides are intentionally independent.

OpenAI models use the Responses API with SDK-native Pydantic Structured Outputs. DeepSeek uses
the official OpenAI-compatible Chat Completions API, explicitly disables thinking for the Text
Scouts route, and validates JSON locally against the caller's Pydantic schema.

## Usage

```python
from kairos_llm import LLMGateway, LLMWorkload

gateway = LLMGateway()
result = await gateway.complete(
    system=SYSTEM_PROMPT,
    user=compact_json,
    workload=LLMWorkload.AGGREGATOR_CONFLICT,
    schema=TacticalOutput,
)
print(result.parsed, result.cost_usd, result.model)
```

Existing callers remain valid:

```python
from kairos_core.enums import ReasoningEffort

result = await gateway.complete(
    system=SYSTEM_PROMPT,
    user=compact_json,
    effort=ReasoningEffort.HIGH,
)
```

## Provider resolution telemetry

`LLMResult.model` is the stable model ID sent in the request. `provider`, `request_id`,
`resolved_model` and `system_fingerprint` preserve the values returned by the provider. A
`BudgetedLLMGateway` also attaches the exact durable `budget_reservation_id` after the
reservation is committed. It stamps the call start and observed response time around
the provider request; direct unbudgeted results leave these fields unset. This distinction matters for
aliases: Text Scouts sends `deepseek-flash`, while telemetry records the resolved backend
returned by DeepSeek without hard-coding a provider snapshot as an API model ID.
The provider/model fields are included in the `llm.response` structured log event; callers use
the complete result to persist paid-review provenance without copying secrets or prompts.

## Research-only LLM proposals

`LLMProposalOutputV1` is a closed-world schema for an experimental model hypothesis. It permits only
`LONG_BIAS`, `SHORT_BIAS`, `VOLATILITY_ALERT`, `NO_PROPOSAL`, or `DEFER`, a bounded rationale,
and IDs for evidence already
provided by the caller. Fields that could set execution, sizing, venue, campaign scope, or provenance
are rejected. `VOLATILITY_ALERT` cites timestamped input evidence and asserts no trade direction.
`build_llm_trade_proposal` binds the validated output to trusted caller context and the
completed gateway result, including provider/model resolution, request ID, response hash, and the
durable budget reservation.
`build_llm_proposal_completion_receipt` requires those trusted timing fields and
an exact recomputation of the proposal from the gateway result. The separate
receipt allows a SIM research pair to reject a model answer that arrived after
its scheduled decision time; the model cannot supply its own timing metadata.

The adapter returns only `LLMTradeProposalV1`, a research record—not `StrategyIntentV1`,
`CandidateReviewV1`, `RiskTradeDecisionV1`, or an order. The opt-in
`build_and_publish_llm_trade_proposal` helper publishes an already completed, validated result
only to `Topics.LLM_TRADE_PROPOSAL`; it makes no provider call and has no conversion to strategy,
risk, or execution contracts. An uncertain publish is not retried automatically: callers must
reconcile the stable proposal/message ID first. Risk Manager and Execution must not subscribe to
this research topic. This is an event-transport boundary, not a strategy evaluation or trading
authorization.

An attempted call that fails is not a model `DEFER` or `NO_PROPOSAL` response.
The separate `LLMCallFailureV1` research receipt requires caller-attested attempt,
reservation, failure-class, and actual observation-time metadata. This adapter
does not currently receive all of that metadata from `BudgetedLLMGateway` on an
exception, so it cannot honestly construct or publish such a receipt. Late or
missing failure receipts fail closed at pairing; a pre-registered schedule and
durable attempt/timeout ledger are still required for a complete matched A/B
denominator. No provider call is made merely to create a receipt.

## Failure semantics

The gateway retries only connection failures, timeouts and explicitly transient HTTP responses
(`408`, `409`, `429`, and `5xx`). Authentication, bad requests, invalid structured output and
programming errors fail immediately. Health telemetry is best-effort and cannot repeat an
otherwise successful, potentially billable provider call.

## Cost scenario

With the existing planning call/token volumes, no cache hits and conservative DeepSeek peak
pricing, the role-aware table produces an estimated $58.30/month API scenario at ordinary input
rates, or $61.25 if every OpenAI input token incurs the higher cache-write rate. Off-peak
DeepSeek requests are billed lower, but the local ledger deliberately does not depend on
dispatch time. The budget wrapper reserves at the higher cache-write rate and, when the
provider omits cache-write usage, accounts every noncached input token as a write. These are
tested estimates, not a guaranteed budget; actual usage, retries, long-context multipliers
and provider prices must be monitored.

Production callers must wrap `LLMGateway(max_retries=0)` in `BudgetedLLMGateway` and attach a
durable `LLMUsageBudget`. During shadow qualification the shared provider ceilings are exactly
`$1.00` for DeepSeek and `$12.00` for OpenAI, leaving the rest of the funded balances outside
this stage's authority. Before every
provider call the wrapper reserves a conservative prompt/schema/output allowance. Successful
usage is committed in whole microdollars; failures and ambiguous outcomes keep the reservation
open so a retry cannot silently spend the same capacity twice. Paid calls are denied when no
durable backend is attached.

Workload output ceilings are also fixed before the request: Text Scouts 1,024 tokens, Luna
normal aggregation 2,048, Sol conflict handling 4,096, and Sol macro strategy 8,192. A lower
global setting remains authoritative.

## Provider qualification

Keep provider keys in local secret files and run the non-trading contract probe:

```powershell
uv run --locked kairos-llm-qualify `
  --openai-key-file D:\Kairos\secrets\openai_api_key `
  --deepseek-key-file D:\Kairos\secrets\deepseek_api_key `
  --samples 2 `
  --output $env:TEMP\kairos-llm-qualification.json `
  --overwrite
```

The tool checks every workload route with an exact structured `NO_TRADE` response,
records requested and resolved model identities, fingerprint, token usage, latency and
estimated price-table cost, and probes each provider's authenticated model-list endpoint
for observable quota headers. Keys are never accepted on argv or written to evidence.
Missing keys make zero billable calls and produce `BLOCKED`; malformed output, model
alias drift, absent usage, excessive latency/cost, or provider errors fail closed. The
probe validates API mechanics only—not market reasoning quality—and its report always
contains `live_orders_allowed=false`.

Paid diagnostics can select one route so an already-qualified model is not called
again. The command also refuses to start when its conservative planned allowance
exceeds the supplied ceiling; the default ceiling is `$0.05`:

```powershell
uv run --locked kairos-llm-qualify `
  --deepseek-key-file D:\Kairos\secrets\deepseek_api_key `
  --workload text_scouts `
  --samples 1 `
  --maximum-planned-cost-usd 0.002 `
  --output $env:TEMP\kairos-deepseek-qualification.json
```

## Local development

```powershell
python -m uv sync --locked
python -m uv run --locked ruff check kairos_llm tests
python -m uv run --locked ruff format --check kairos_llm tests
python -m uv run --locked mypy kairos_llm
python -m uv run --locked bandit -q -r kairos_llm -x tests
python -m uv run --locked pytest -q --tb=short
python -m uv build --no-sources
```

Part of the [Kairos](https://github.com/Kairos-cryptoAI/kairos) system. MIT licensed.
