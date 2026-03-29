"""Backtest runner — replay-based counterfactual testing."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from stateloom.core.types import SessionStatus
from stateloom.experiment.models import VariantConfig

if TYPE_CHECKING:
    from stateloom.experiment.manager import ExperimentManager
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.experiment")


@dataclass
class BacktestResult:
    """Result of a single backtest run (one session x one variant)."""

    source_session_id: str
    variant_name: str
    replay_session_id: str
    total_cost: float = 0.0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    call_count: int = 0
    latency_ms: float = 0.0
    status: str = ""
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_session_id": self.source_session_id,
            "variant_name": self.variant_name,
            "replay_session_id": self.replay_session_id,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "call_count": self.call_count,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "metadata": self.metadata,
        }


class BacktestRunner:
    """Replay-based counterfactual testing with different variant configs.

    The runner re-executes user-provided agent code inside a session that has
    experiment variant overrides applied via the middleware pipeline.  For the
    first ``mock_until_step`` steps the ReplayEngine returns cached LLM
    responses; steps beyond that boundary execute live — with the variant's
    model/config overrides in effect.

    ``agent_fn`` receives the replay ``Session`` object and should run the
    same logic the original session executed.  Any LLM calls it makes will be
    intercepted by the middleware pipeline (including ExperimentMiddleware).
    """

    def __init__(
        self,
        gate: Gate,
        experiment_manager: ExperimentManager | None = None,
    ) -> None:
        self._gate = gate
        self._experiment_manager = experiment_manager

    def run_backtest(
        self,
        source_session_ids: list[str],
        variants: list[dict[str, Any] | VariantConfig],
        agent_fn: Callable[..., Any],
        *,
        mock_until_step: int | None = None,
        strict: bool = False,
        evaluator: Callable[[BacktestResult], float | str | None] | None = None,
    ) -> list[BacktestResult]:
        """Run backtest: replay sessions with different variant configs.

        Args:
            source_session_ids: Session IDs whose recorded steps provide the
                cached responses for the mocked portion of the replay.
            variants: Variant configs to test.  Each source session is
                replayed once per variant.
            agent_fn: A callable that re-executes the agent logic.  It
                receives the replay ``Session`` as its only argument.  All
                LLM calls made inside ``agent_fn`` flow through the
                middleware pipeline where ExperimentMiddleware applies the
                variant overrides.
            mock_until_step: Mock steps 1..N with cached responses from the
                source session.  Defaults to *all* recorded steps (pure
                offline evaluation — only config/model changes take effect).
            strict: If True, block outbound HTTP calls during mocked steps.
            evaluator: Optional callback that scores each result.  Receives
                a ``BacktestResult`` and returns a float (0-1), a rating
                string ("success"/"failure"/"partial"), or None to skip.

        Returns:
            List of ``BacktestResult`` — one per (session, variant) pair.
        """
        resolved_variants = [
            VariantConfig.from_dict(v) if isinstance(v, dict) else v for v in variants
        ]

        results: list[BacktestResult] = []

        for session_id in source_session_ids:
            source_session = self._gate.store.get_session(session_id)
            if source_session is None:
                logger.warning("Source session '%s' not found, skipping", session_id)
                continue

            step_limit = (
                mock_until_step if mock_until_step is not None else source_session.step_counter
            )

            for variant in resolved_variants:
                result = self._run_single(session_id, variant, step_limit, strict, agent_fn)
                results.append(result)

                if evaluator is not None and self._experiment_manager is not None:
                    try:
                        eval_result = evaluator(result)
                        if eval_result is not None:
                            if isinstance(eval_result, (int, float)):
                                self._experiment_manager.record_feedback(
                                    result.replay_session_id,
                                    rating="success" if eval_result >= 0.5 else "failure",
                                    score=float(eval_result),
                                )
                            elif isinstance(eval_result, str):
                                self._experiment_manager.record_feedback(
                                    result.replay_session_id,
                                    rating=eval_result,
                                )
                    except Exception:
                        logger.debug(
                            "Evaluator failed for %s", result.replay_session_id, exc_info=True
                        )

        return results

    def _run_single(
        self,
        source_session_id: str,
        variant: VariantConfig,
        mock_until_step: int,
        strict: bool,
        agent_fn: Callable[..., Any],
    ) -> BacktestResult:
        """Run a single backtest: set up replay context, then execute agent_fn."""
        from stateloom.core.context import get_current_session, set_current_session

        replay_session_id = f"backtest-{source_session_id}-{variant.name}"

        # Create replay session with variant config in metadata
        session = self._gate.session_manager.create(
            session_id=replay_session_id,
            name=f"Backtest {source_session_id} / {variant.name}",
        )
        session.metadata = {
            "experiment_variant_config": variant.to_dict(),
            "backtest_source": source_session_id,
            "variant_name": variant.name,
        }
        self._gate.store.save_session(session)

        previous_session = get_current_session()
        output = None
        try:
            from stateloom.replay.engine import ReplayEngine

            engine = ReplayEngine(
                self._gate,
                session_id=source_session_id,
                mock_until_step=mock_until_step,
                strict=strict,
            )
            # Context manager ensures engine.stop() is always called,
            # even if agent_fn or session setup raises.
            with engine:
                # Override the session context so LLM calls are recorded
                # against our backtest session (with experiment variant metadata).
                set_current_session(session)

                # Actually re-execute the agent code
                try:
                    output = agent_fn(session)
                except Exception:
                    logger.debug(
                        "agent_fn raised during backtest %s/%s",
                        source_session_id,
                        variant.name,
                        exc_info=True,
                    )
        except Exception:
            logger.debug(
                "Backtest replay setup failed for %s/%s",
                source_session_id,
                variant.name,
                exc_info=True,
            )
        finally:
            # End the session and restore prior context
            session.end(SessionStatus.COMPLETED)
            self._gate.store.save_session(session)
            set_current_session(previous_session)

        # Collect metrics from the (now updated) replay session
        updated = self._gate.store.get_session(replay_session_id)
        if updated is None:
            updated = session

        return BacktestResult(
            source_session_id=source_session_id,
            variant_name=variant.name,
            replay_session_id=replay_session_id,
            total_cost=updated.total_cost,
            total_tokens=updated.total_tokens,
            total_prompt_tokens=updated.total_prompt_tokens,
            total_completion_tokens=updated.total_completion_tokens,
            call_count=updated.call_count,
            status=updated.status.value,
            output=output,
            metadata=updated.metadata,
        )
