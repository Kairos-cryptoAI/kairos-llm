"""Optional installed-consumer contract checks; no network/provider construction.

The normal LLM wheel does not depend on Text/Macro/Router. Qualification must
install their pinned published wheels before these tests count as evidence;
an absent package is an explicit skip, never a contract PASS.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from kairos_core import ImpactDirection, Side
from kairos_core.enums import CandidateReviewTier, StrategicTrigger

from kairos_llm.models import LLMWorkload


class _SchemaGateway:
    def __init__(self, output):
        self.output, self.calls = output, []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed=self.output)


async def test_installed_text_schema_workload_and_provenance():
    sentiment = pytest.importorskip("kairos_text.sentiment")
    models = pytest.importorskip("kairos_text.models")
    schemas = pytest.importorskip("kairos_text.schemas")
    item = models.NewsItem(
        title="Offline bearish fixture",
        source="fixture.invalid",
        source_kind="rss",
        url="https://fixture.invalid/news",
        published_at=datetime(2026, 10, 3, tzinfo=UTC),
        timestamp_is_estimated=False,
    )
    output = schemas.SentimentBatch(
        signals=[
            schemas.ExtractedSentiment(
                topic="BTCUSDT",
                sentiment=-0.8,
                impact=ImpactDirection.BEARISH,
                confidence=0.9,
                summary="Offline evidence only",
                item_ids=[1],
            )
        ]
    )
    gateway = _SchemaGateway(output)
    result = await sentiment.SentimentExtractor(gateway).extract([item])
    assert gateway.calls[0]["workload"] is LLMWorkload.TEXT_SCOUTS
    assert gateway.calls[0]["schema"] is schemas.SentimentBatch
    assert len(result) == 1 and result[0].confidence <= 0.65
    assert result[0].sources == ["https://fixture.invalid/news"]


async def test_installed_text_rejects_nonexistent_news_id():
    sentiment = pytest.importorskip("kairos_text.sentiment")
    models = pytest.importorskip("kairos_text.models")
    schemas = pytest.importorskip("kairos_text.schemas")
    gateway = _SchemaGateway(
        schemas.SentimentBatch(
            signals=[
                schemas.ExtractedSentiment(
                    topic="BTCUSDT",
                    sentiment=-0.8,
                    impact=ImpactDirection.BEARISH,
                    confidence=0.9,
                    summary="Invented citation",
                    item_ids=[99],
                )
            ]
        )
    )
    assert (
        await sentiment.SentimentExtractor(gateway).extract(
            [models.NewsItem(title="Offline fixture", source="fixture.invalid")]
        )
        == []
    )


@pytest.mark.parametrize(
    "weights,expected",
    [
        ([{"strategy_name": "baseline", "weight": 0.2}], {"baseline": 0.2}),
        ([{"strategy_name": "invented", "weight": 0.2}], {}),
    ],
)
async def test_installed_macro_frozen_ids_and_failclosed_schema(weights, expected):
    macro = pytest.importorskip("kairos_macro.strategist")
    gateway = _SchemaGateway(
        dict(
            regime="BEAR",
            stable_reserve_pct=0.8,
            strategy_weights=weights,
            max_gross_leverage=1.0,
            rationale="Offline fixture, not capital allocation authority",
        )
    )
    allocation = await macro.MacroStrategist(gateway, allowed_strategy_ids=("baseline",)).allocate(
        "{}", trigger=StrategicTrigger.SCHEDULE
    )
    assert gateway.calls[0]["workload"] is LLMWorkload.MACRO_STRATEGIST
    assert gateway.calls[0]["schema"] is macro.AllocationOutput
    assert allocation.strategy_weights == expected
    if not expected:
        assert allocation.stable_reserve_pct == 1.0 and allocation.max_gross_leverage == 1.0


def test_installed_router_opposite_stale_future_and_duplicates():
    candidate = pytest.importorskip("kairos_router.candidate")
    aggregation = pytest.importorskip("kairos_router.aggregation")
    assert candidate.classify_candidate(Side.LONG, Side.SHORT) is CandidateReviewTier.CONFLICT
    assert candidate.classify_candidate(Side.LONG, Side.FLAT) is CandidateReviewTier.NORMAL
    with pytest.raises(ValueError):
        candidate.classify_candidate(Side.FLAT, Side.SHORT)
    window = aggregation.SignalWindow(sentiment_ttl_s=5)
    now = datetime(2026, 10, 3, tzinfo=UTC)
    for identity, produced in (("stale", now - timedelta(seconds=6)), ("future", now + timedelta(seconds=1))):
        window.add_sentiment(
            "BTCUSDT",
            message_id=identity,
            produced_at=produced,
            sentiment=-1.0,
            impact=ImpactDirection.BEARISH,
            confidence=0.9,
        )
    # Event-time policy must not turn stale/future news into a fresh conflict.
    result = window.aggregate("BTCUSDT", as_of=now)
    assert result.bias is Side.FLAT and result.sentiment_ids == ()
    window.add_sentiment(
        "BTCUSDT",
        message_id="fresh",
        produced_at=now,
        sentiment=-1.0,
        impact=ImpactDirection.BEARISH,
        confidence=0.9,
    )
    window.add_sentiment(
        "BTCUSDT",
        message_id="fresh",
        produced_at=now,
        sentiment=1.0,
        impact=ImpactDirection.BULLISH,
        confidence=0.9,
    )
    result = window.aggregate("BTCUSDT", as_of=now)
    assert result.bias is Side.SHORT and result.sentiment_ids == ("fresh",)
