# Opt-in proposal observation and replay

`kairos_llm.research.ResearchProposalCoordinator` is an explicitly constructed
SIM-only engineering adapter. Install the persistence `qualification` extra;
the default package/gateway does not import this module or start research.
It supports only `llm-proposal-research`, not a new production route or a
review policy. There is no order, Risk Manager, venue, or PnL interface.

Before using it, an operator must separately preregister and enroll a new
schedule and adaptive candidate protocol in the SIM database. Do not adopt
Trial 15, V4/V5, historical results, or an unfinished attempt into a new run.

Freeze the actual `ResearchPromptArtifactV1.prompt_sha256`, the versioned
`RESEARCH_INPUT_FEATURE_SHA256`, the strict proposal schema digest, and the
provider/model arm. The prompt artifact includes workload, logical/provider
effort and the effective output cap: these are checked before reservation.
This input renderer projects independently stored causal JSON as-is; it does
not compute indicators, qualify news availability, or implement an adaptive
feature pipeline. No campaign is created or frozen by this module.

The journal must implement the independent `ResearchEvidenceRepository`
interface, not a caller-asserted hash lookup. Sources are loaded by receipt
identity and reparsed canonically; exactly one market snapshot is required.
The provider prompt receives saved content and stable evidence IDs, not a
remote fetch path or a caller-substituted source. All inputs must have been
observed by the scheduled market clock. Content must be sanitized before it
is stored; this module does not handle credentials or fetch sources.

The admission order is:

1. Verify frozen inputs and load any existing attempt.
2. Reserve against the existing cumulative provider/campaign budget using a
   deterministic attempt ID.
3. Commit append-only START. Only a successful new insert permits dispatch.
4. Invoke the provider once through the existing no-retry budgeted gateway.
5. Commit actual cost on success, then save the adapted proposal/completion,
   or save the sanitized observed failure/unresolved terminal.

START is an admission fence, not proof of remote delivery. A crash after START,
ambiguous terminal write, unclassifiable failure, or invalid output without a
trustworthy original response hash stays unresolved. It is never treated as
`NOT_CALLED`, `NO_PROPOSAL`, or an invented transport failure. Failed/timeout/
cancelled calls retain their reservations. Post-response adaptation failure
may already have settled actual cost, but still does not create an admissible
proposal. Cancellation remains cancellation even if terminal storage fails.
Restart/duplicate observation replays saved facts; it never automatically
reissues a provider request or obtains a new reservation ID. Changed source
sets cannot override the journal's unique campaign/arm/sample fence.

`replay_sample` makes no provider or budget request. It loads and resolves the
real saved strategy evaluation/source receipts, uses existing causal pairing,
and calls `record_verified_sample` for a second independent persistence check.
Unresolved attempts and receipts observed after the decision/deadline cannot
be paired. A missing attempt may become `NOT_CALLED` only if the independent
journal also verifies that no attempt exists for that campaign/arm/sample.

Offline tests cover ordering, success replay, real receipt lookup, changed
prompt/route/content, budget and enrollment denial, exact duplicate admission,
timeouts, cancellation, ambiguous storage and late receipts. These tests do
not demonstrate durable database deployment, provider semantic quality,
market profit, full adaptive campaign coverage, or scientific alpha. Readiness
and trading policy are unchanged.
