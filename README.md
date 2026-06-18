# kairos-llm

The **LLM gateway**: a single choke-point for every model call in Kairos. No layer talks
to OpenAI directly — Text Scouts, the Aggregator and the Macro-Strategist all go through it.

## Responsibilities
- **Effort → model routing.** Maps `ReasoningEffort` (`low`/`medium`/`high`/`xhigh`) to a
  concrete model. The default split is cost-optimised per the spec: cheap models carry the
  routine flow, the flagship `gpt-5.5` is reserved for `high`/`xhigh`.
- **Cost accounting.** Tracks spend with the spec's tariff ($5 / $30 per 1M in/out, $0.50
  cached) so budget alerts are trivial.
- **Resilience.** Owns the timeout + retry budget and raises typed `LLMServerError` /
  `LLMTimeout` so the Risk Manager's **circuit breaker** can detach the LLM on 5xx/timeouts.
- **Structured output.** Always requests `json_object`; optionally validates against a
  Pydantic schema and raises `LLMBadOutput` on a malformed response.

## Usage
```python
from kairos_core.enums import ReasoningEffort
from kairos_llm import LLMGateway

gw = LLMGateway()
res = await gw.complete(system=SYSTEM_PROMPT, user=compact_json, effort=ReasoningEffort.HIGH)
print(res.parsed, res.cost_usd, res.model)
```

## Cost model
| effort | default model | typical use |
| --- | --- | --- |
| low | gpt-5.5-mini | Text Scouts sentiment |
| medium | gpt-5.5-mini | Aggregator, calm market |
| high | gpt-5.5 | Aggregator, signal conflict |
| xhigh | gpt-5.5 | Macro-Strategist, regime change |

---
Part of the [Kairos](https://github.com/Kairos-cryptoAI/kairos) system. MIT licensed.
