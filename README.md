# kairos-llm

The **LLM gateway**: a single choke-point for every model call in Kairos. No layer talks
to OpenAI directly — Text Scouts, the Aggregator and the Macro-Strategist all go through it.

## Responsibilities
- **Effort → model routing.** Maps `ReasoningEffort` (`low`/`medium`/`high`/`xhigh`) to a
  concrete model. The default split is cost-optimised per the spec: cheap models carry the
  routine flow, `gpt-5.6-sol` is reserved for `high`/`xhigh`.
- **Cost accounting.** Tracks spend with the spec's tariff ($5 / $30 per 1M in/out, $0.50
  cached) so budget alerts are trivial.
- **Resilience.** Owns the timeout + retry budget and raises typed `LLMServerError` /
  `LLMTimeout` so the Risk Manager's **circuit breaker** can detach the LLM on 5xx/timeouts.
  Retries are limited to connection/timeouts and explicitly transient HTTP responses
  (`408`, `409`, `429`, and `5xx`). Authentication, bad-request, invalid structured-output,
  and programming errors fail immediately. Health telemetry is best-effort and can never
  repeat an otherwise successful, potentially billable provider call.
- **Structured output.** OpenAI uses Responses API Structured Outputs with SDK-native
  Pydantic parsing. DeepSeek JSON is validated locally against the same schema.

## Usage
```python
from kairos_core.enums import ReasoningEffort
from kairos_llm import LLMGateway

gw = LLMGateway()
res = await gw.complete(system=SYSTEM_PROMPT, user=compact_json, effort=ReasoningEffort.HIGH)
print(res.parsed, res.cost_usd, res.model)
```

## Local development

```powershell
winget install --id astral-sh.uv --exact
uv sync --locked
uv run --locked ruff check kairos_llm tests
uv run --locked ruff format --check kairos_llm tests
uv run --locked mypy kairos_llm
uv run --locked bandit -q -r kairos_llm -x tests
uv run --locked pytest -q --tb=short
uv build --no-sources
```

## Cost model
| effort | default model | typical use |
| --- | --- | --- |
| low | deepseek-v4-flash (thinking disabled) | Text Scouts sentiment |
| medium | deepseek-v4-pro (thinking disabled) | Aggregator, calm market |
| high | gpt-5.6-sol | Aggregator, signal conflict |
| xhigh | gpt-5.6-sol | Macro-Strategist, regime change |

---
Part of the [Kairos](https://github.com/Kairos-cryptoAI/kairos) system. MIT licensed.
