"""Aggregation functions for consensus responses."""

from __future__ import annotations

import difflib
import logging
from typing import Any

from stateloom.consensus.confidence import extract_confidence
from stateloom.consensus.models import DebaterResponse
from stateloom.consensus.prompts import JUDGE_SYNTHESIS_TEMPLATE

logger = logging.getLogger("stateloom.consensus.aggregation")


def compute_agreement(responses: list[DebaterResponse]) -> float:
    """Compute pairwise agreement score across responses (0.0-1.0).

    Uses difflib.SequenceMatcher ratio averaged over all pairs.
    """
    if len(responses) < 2:
        return 1.0
    scores: list[float] = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            ratio = difflib.SequenceMatcher(
                None, responses[i].content, responses[j].content
            ).ratio()
            scores.append(ratio)
    return sum(scores) / len(scores) if scores else 1.0


def majority_vote(responses: list[DebaterResponse]) -> tuple[str, float]:
    """Select the most common answer by semantic similarity grouping.

    Returns (best_answer, confidence).
    """
    if not responses:
        return ("", 0.0)
    if len(responses) == 1:
        return (responses[0].content, responses[0].confidence)

    # Group similar responses using pairwise similarity
    groups: list[list[DebaterResponse]] = []
    threshold = 0.6  # similarity threshold for grouping

    for resp in responses:
        placed = False
        for group in groups:
            ratio = difflib.SequenceMatcher(None, resp.content, group[0].content).ratio()
            if ratio >= threshold:
                group.append(resp)
                placed = True
                break
        if not placed:
            groups.append([resp])

    # Find the largest group
    groups.sort(key=len, reverse=True)
    best_group = groups[0]

    # Pick the response with the highest confidence from the best group
    best = max(best_group, key=lambda r: r.confidence)
    vote_confidence = len(best_group) / len(responses)
    return (best.content, max(vote_confidence, best.confidence))


def confidence_weighted(responses: list[DebaterResponse]) -> tuple[str, float]:
    """Select the answer with the highest confidence-weighted score.

    Returns (best_answer, weighted_confidence).
    """
    if not responses:
        return ("", 0.0)
    if len(responses) == 1:
        return (responses[0].content, responses[0].confidence)

    best = max(responses, key=lambda r: r.confidence)
    total_conf = sum(r.confidence for r in responses)
    if total_conf > 0:
        weighted = best.confidence / total_conf * len(responses)
        weighted = min(1.0, weighted / len(responses) + best.confidence * 0.5)
    else:
        weighted = 0.5
    return (best.content, weighted)


async def judge_synthesis(
    responses: list[DebaterResponse],
    judge_model: str,
    gate: Any,
    parent_session: Any,
) -> tuple[str, float]:
    """Use a judge model to synthesize a final answer from all responses.

    Returns (synthesized_answer, confidence).
    """
    from stateloom.chat import Client

    # Build transcript
    transcript_parts: list[str] = []
    for resp in responses:
        transcript_parts.append(
            f"**{resp.model}** (Round {resp.round_number}, "
            f"Confidence: {resp.confidence:.2f}):\n{resp.content}"
        )
    transcript = "\n\n---\n\n".join(transcript_parts)

    prompt = JUDGE_SYNTHESIS_TEMPLATE.format(transcript=transcript)

    client = Client(
        session_id=f"{parent_session.id}-judge",
        parent=parent_session.id,
        durable=True,
        name="consensus-judge",
    )
    try:
        result = await client.achat(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = result.content
        confidence = extract_confidence(text)
        return (text, confidence)
    finally:
        await client.aclose()
