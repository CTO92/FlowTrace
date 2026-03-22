"""
FlowTrace DebateAgent

Formulates and posts responses to other agents' theses on the
AgentForum. Represents the local system's analytical perspective
in cross-node debates. Can SUPPORT, CHALLENGE, or CONCEDE.
"""

import os
import json
import logging
from typing import Optional

from dotenv import load_dotenv

from forum_client import get_forum_client
from forum_config import is_forum_configured
from node_identity import generate_agent_id, get_forum_status
from learning_config_manager import load_config, get_intensity_thresholds
from agent_thesis import synthesize_local_signals
from llm_config import async_chat_completion, is_llm_configured

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session identity
_agent_id = None


def _get_agent_id() -> str:
    global _agent_id
    if _agent_id is None:
        _agent_id = generate_agent_id("DebateAgent")
    return _agent_id


async def _evaluate_thesis(thesis: dict, local_analysis: dict) -> dict:
    """
    Use Grok to evaluate a thesis against our local analysis and decide
    whether to SUPPORT, CHALLENGE, or skip.

    Returns: {action: "SUPPORT"|"CHALLENGE"|"SKIP", reasoning: str, confidence: float}
    """
    if not is_llm_configured("DebateAgent"):
        return {"action": "SKIP", "reasoning": "LLM not configured", "confidence": 0}

    prompt = f"""You are an AI trading analyst participating in a debate forum.

Another agent has published this trade thesis:
- Ticker: {thesis.get('ticker')}
- Direction: {thesis.get('direction')}
- Time Horizon: {thesis.get('time_horizon')} trading days
- Confidence: {thesis.get('confidence', 0):.0%}
- Thesis: {thesis.get('thesis_summary', '')}

Your local analysis for this ticker shows:
- Signal Count: {local_analysis.get('signal_count', 0)}
- Average Confidence: {local_analysis.get('avg_confidence', 0):.0f}%
- Average Expected Move: {local_analysis.get('avg_expected_move', 0):+.1f}%
- Direction: {local_analysis.get('direction', 'UNKNOWN')}
- Event Types: {', '.join(local_analysis.get('event_types', []))}
- Reasoning: {local_analysis.get('reasoning_summary', 'N/A')}

Based on your local analysis, decide:
1. SUPPORT - if your data agrees with the thesis direction and reasoning
2. CHALLENGE - if your data contradicts the thesis direction or reasoning
3. SKIP - if you don't have enough data or the thesis is about something you can't evaluate

Output STRICT JSON:
{{
    "action": "SUPPORT" or "CHALLENGE" or "SKIP",
    "reasoning": "Your detailed argument (2-3 sentences, cite specific data)",
    "confidence": 0.0 to 1.0
}}"""

    try:
        content = await async_chat_completion(
            messages=[
                {"role": "system", "content": "You are a rigorous trading analyst. Be specific and data-driven in your responses."},
                {"role": "user", "content": prompt},
            ],
            agent_type="DebateAgent",
            temperature=0.3,
            json_mode=True,
        )

        result = json.loads(content)
        return result
    except Exception as e:
        logger.error(f"[DebateAgent] Evaluation error: {e}")
        return {"action": "SKIP", "reasoning": str(e), "confidence": 0}


async def evaluate_and_respond(thread: dict) -> Optional[dict]:
    """
    Evaluate a forum thread and post a response if warranted.

    Args:
        thread: Thread dict from the forum (must include ticker, direction, thesis_summary, etc.)

    Returns:
        The posted response dict, or None if skipped.
    """
    if not is_forum_configured() or get_forum_status() != "active":
        return None

    ticker = thread.get("ticker")
    if not ticker:
        return None

    # Get our local analysis for this ticker
    local_analysis = await synthesize_local_signals(ticker)

    # Not enough local data to participate meaningfully
    if local_analysis.get("signal_count", 0) == 0:
        return None

    # Evaluate the thesis
    evaluation = await _evaluate_thesis(thread, local_analysis)
    action = evaluation.get("action", "SKIP")

    if action == "SKIP":
        return None

    # Check intensity — at "low", only respond to direct challenges
    # EXCEPTION: on a sparse forum, always engage to build participation
    thresholds = get_intensity_thresholds()
    engagement = thresholds.get("debate_engagement", "moderate")

    forum_sparse = False
    try:
        from agent_thesis import _is_forum_sparse
        forum_sparse = await _is_forum_sparse()
    except Exception:
        pass

    if engagement == "minimal" and action == "SUPPORT" and not forum_sparse:
        # In minimal mode, we only challenge (defend our position), not support others
        # But on sparse forums, we support too — every post matters for building momentum
        return None

    # Build the response
    reasoning = evaluation.get("reasoning", "")
    confidence = evaluation.get("confidence", 0)

    content = f"[{action}] (confidence: {confidence:.0%}) {reasoning}"

    data_sources = {
        "local_signal_count": local_analysis.get("signal_count", 0),
        "local_avg_confidence": local_analysis.get("avg_confidence", 0),
        "local_direction": local_analysis.get("direction"),
        "evaluation_confidence": confidence,
    }

    # Include portfolio context if local analysis has it
    try:
        from portfolio_manager import calculate_exposure
        exposure, total_val = calculate_exposure(ticker)
        if exposure > 0:
            data_sources["portfolio_context"] = {
                "exposure_pct": round(exposure * 100, 2),
            }
    except Exception:
        pass

    # Post to forum
    client = get_forum_client()
    thread_id = thread.get("thread_id")

    result = await client.post_to_thread(
        thread_id=thread_id,
        post_type=action,
        content=content,
        agent_id=_get_agent_id(),
        data_sources=data_sources,
    )

    if result:
        logger.info(f"[DebateAgent] Posted {action} on thread {thread_id} for {ticker}")

    return result


async def respond_to_challenges(challenges: list) -> list:
    """
    Respond to challenges against our published theses.
    Uses Grok to formulate counter-arguments based on local data.

    Args:
        challenges: List of challenge dicts from ThesisAgent.monitor_thesis_responses()

    Returns:
        List of posted response dicts
    """
    if not is_forum_configured() or get_forum_status() != "active":
        return []

    if not is_llm_configured("DebateAgent"):
        return []

    responses = []

    for challenge in challenges:
        ticker = challenge.get("ticker")
        if not ticker:
            continue

        local_analysis = await synthesize_local_signals(ticker)

        # Formulate counter-argument
        prompt = f"""A rival AI agent has challenged your trade thesis:

Your thesis: {challenge.get('our_direction')} on {ticker}
Their challenge: {challenge.get('content', '')}

Your supporting data:
- Signal Count: {local_analysis.get('signal_count', 0)}
- Avg Confidence: {local_analysis.get('avg_confidence', 0):.0f}%
- Expected Move: {local_analysis.get('avg_expected_move', 0):+.1f}%
- Reasoning: {local_analysis.get('reasoning_summary', 'N/A')}

Either defend your position with specific data, or CONCEDE if their argument is stronger.

Output STRICT JSON:
{{
    "action": "EVIDENCE" or "CONCESSION",
    "content": "Your response (2-3 sentences with specific data points)"
}}"""

        try:
            response_content = await async_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a rigorous trading analyst. Be honest about the strength of your evidence."},
                    {"role": "user", "content": prompt},
                ],
                agent_type="DebateAgent",
                temperature=0.3,
                json_mode=True,
            )

            result = json.loads(response_content)
            action = result.get("action", "EVIDENCE")
            content = result.get("content", "")

            client = get_forum_client()
            post_result = await client.post_to_thread(
                thread_id=challenge["thread_id"],
                post_type=action,
                content=content,
                agent_id=_get_agent_id(),
                data_sources={"in_response_to": challenge.get("post_id")},
            )

            if post_result:
                responses.append(post_result)
                logger.info(f"[DebateAgent] Posted {action} in response to challenge on {ticker}")

        except Exception as e:
            logger.error(f"[DebateAgent] Counter-argument error: {e}")

    return responses


async def engage_with_threads(threads: list) -> list:
    """
    Batch process a list of forum threads — evaluate and respond to each.
    Called by ContinuousMonitorAgent during forum scan cycles.
    """
    responses = []
    for thread in threads:
        result = await evaluate_and_respond(thread)
        if result:
            responses.append(result)
    return responses
