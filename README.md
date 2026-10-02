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
| `TEXT_SCOUTS` | `gpt-6-luna` | `low` | $0.10 · $0.01 · $0.125 · $0.50 |
| `AGGREGATOR_NORMAL` | `gpt-6-luna` | `medium` | $0.10 · $0.01 · $0.125 · $0.50 |
| `AGGREGATOR_CONFLICT` | `gpt-6.1-sol` | `high` | $2.00 · $0.10 · $2.50 · $10.00 |
| `MACRO_STRATEGIST` | `gpt-6.1-sol` | `xhigh` | $2.00 · $0.10 · $2.50 · $10.00 |

The original effort-only API remains supported and maps `low`, `medium`, `high`, and `xhigh` to
the same four routes. New callers should provide `LLMWorkload`; workload overrides and legacy
effort overrides are intentionally independent.

All default workload routes use OpenAI and the Responses API with SDK-native Pydantic Structured
Outputs. The DeepSeek adapter remains available only for explicit, non-default overrides; it is
not selected by any default workload route. Provider support is not evidence of qualification.

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
aliases: explicit DeepSeek overrides may resolve to a provider backend name, while telemetry
records the resolved backend without hard-coding a provider snapshot as an API model ID.
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

For a preregistered adaptive-candidate SIM campaign,
`build_preregistered_adaptive_llm_proposal` additionally binds proposal creation
to the frozen `AdaptiveCandidateProtocolV1`: campaign and arm, exact provider and
requested/resolved model, prompt hash, output-schema hash, and caller-supplied
feature hash must match the preregistered LLM-proposal arm. The supplied
`ResearchObservationScheduleV1` must have the protocol's exact digest and the
proposal must match one scheduled sample's symbol, timeframe, decision clock,
deadline, and (when precommitted) market snapshot hash. A changed route,
prompt, schema, or feature set requires a new protocol revision rather than
being silently mixed into the campaign. This check makes no provider call and
does not admit results to the campaign by itself; durable sample recording and
coverage sealing remain Persistence's responsibility.

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

The previously recorded `$58.30`/`$61.25` monthly scenario used the former mixed-provider
defaults and is not an estimate for the current OpenAI-only defaults. Recalculate a campaign
estimate from its frozen call/token volumes and route-specific prices before enabling paid
shadow requests. The budget wrapper reserves conservatively and, when a provider omits cache-write
usage, treats noncached input as cache-write usage. Any estimate is not a guarantee; actual usage,
retries, long-context multipliers and provider prices must be monitored.

Production callers must wrap `LLMGateway(max_retries=0)` in `BudgetedLLMGateway` and attach a
durable `LLMUsageBudget`. During shadow qualification the shared provider ceilings are exactly
`$1.00` for DeepSeek and `$12.00` for OpenAI, leaving the rest of the funded balances outside
this stage's authority. The DeepSeek ceiling is not used by default routes. Before every
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
  --expected-database-name kairos `
  --openai-key-file D:\Kairos\secrets\openai_api_key `
  --samples 2 `
  --output $env:TEMP\kairos-llm-qualification.json `
  --overwrite
```

The tool checks every workload route with an exact structured `NO_TRADE` response,
records requested and resolved model identities, fingerprint, token usage, latency and
estimated price-table cost. OpenAI's authenticated model-list check requires positive
remaining request capacity in its rate-limit headers; DeepSeek's read-only balance check
uses only its `is_available` flag and never records balance amounts. Credential-bearing
qualification requests are pinned to the official provider HTTPS endpoints even if a
service `.env` overrides its normal gateway URL. Keys are never accepted on argv or written
to evidence.
The CLI requires an explicit `KAIROS_PERSISTENCE_DATABASE_URL`, checks the literal and
connected database name, verifies the complete current migration profile, and requires
the registered shared campaign before reading keys or sending requests. The operator
must still confirm that this URL and container/network refer to the authoritative
`kairos-shadow-gate` database, not PAPER or a restored clone; a matching SQL database
name by itself does not prove that operational identity.
Missing keys make zero billable calls and produce `BLOCKED`; malformed output, model
resolution changes within one probe, absent usage, excessive latency/cost, or provider
errors fail closed. A new resolved backend between runs still requires explicit review; this
probe does not maintain a cross-run allow-list. A failed or ambiguous inference can retain
a durable reservation even when the report has no measured cost, so reconcile
`campaign_usage` after every attempt. The
probe validates API mechanics only—not market reasoning quality—and its report always
contains `live_orders_allowed=false`.

Paid diagnostics can select one route so an already-qualified model is not called
again. The command also refuses to start when its conservative planned allowance
exceeds the supplied ceiling; the default ceiling is `$0.05`:

Qualification keeps the legacy `128` output-token default. An explicit
`--max-output-tokens` allowance from `128` through `4096` can be selected before
a new run; each workload's lower role cap still applies. The cost admission,
durable reservation, provider request and report all use that allowance.
[Reasoning tokens share the output budget](https://developers.openai.com/api/docs/guides/reasoning#allocating-space-for-reasoning),
so a short cap can produce an incomplete response before visible JSON while
still incurring costs. Increasing the allowance is not an automatic retry or a
qualification PASS; failed/ambiguous reservations remain for reconciliation.

```powershell
uv run --locked kairos-llm-qualify `
  --expected-database-name kairos `
  --openai-key-file D:\Kairos\secrets\openai_api_key `
  --workload text_scouts `
  --samples 1 `
  --maximum-planned-cost-usd 0.05 `
  --output $env:TEMP\kairos-openai-text-scout-qualification.json
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
