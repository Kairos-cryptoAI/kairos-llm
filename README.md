# kairos-llm

The Kairos LLM gateway is the single provider boundary for Text Scouts, Aggregator and Macro
Strategist. It owns role-aware routing, strict structured output, token-cost accounting,
transient retries and health telemetry; no analytical service talks to a provider directly.

## Role routes

Routing is explicit by workload so two components cannot become coupled merely because they use
the same logical reasoning effort.

| workload | model | provider mode | price / 1M input · cached · output |
| --- | --- | --- | --- |
| `TEXT_SCOUTS` | `deepseek-v4-flash` | non-thinking | $0.14 · $0.0028 · $0.28 |
| `AGGREGATOR_NORMAL` | `gpt-5.6-luna` | `medium` | $0.20 · $0.02 · $1.20 |
| `AGGREGATOR_CONFLICT` | `gpt-5.6-terra` | `high` | $2.00 · $0.20 · $12.00 |
| `MACRO_STRATEGIST` | `gpt-5.6-sol` | `xhigh` | $5.00 · $0.50 · $30.00 |

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

`LLMResult.model` is the stable model ID sent in the request. `resolved_model` and
`system_fingerprint` preserve the values returned by the provider. This distinction matters for
aliases: Text Scouts continues to send `deepseek-v4-flash`, while telemetry can identify a
resolved backend such as the 0731 snapshot without hard-coding that snapshot as an API model ID.
The same fields are included in the `llm.response` structured log event.

## Failure semantics

The gateway retries only connection failures, timeouts and explicitly transient HTTP responses
(`408`, `409`, `429`, and `5xx`). Authentication, bad requests, invalid structured output and
programming errors fail immediately. Health telemetry is best-effort and cannot repeat an
otherwise successful, potentially billable provider call.

## Cost scenario

With the existing planning call/token volumes and no cache hits, the role-aware table produces
an estimated $72.20/month API scenario. This is tested arithmetic, not a guaranteed budget;
actual usage, retries, long-context multipliers and provider prices must be monitored.

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
