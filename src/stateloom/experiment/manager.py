"""Experiment manager — central orchestrator for the experiment framework."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from stateloom.core.event import FeedbackEvent
from stateloom.core.types import AssignmentStrategy, ExperimentStatus, FeedbackRating
from stateloom.experiment.assigner import ExperimentAssigner
from stateloom.experiment.models import (
    Experiment,
    ExperimentAssignment,
    SessionFeedback,
    VariantConfig,
    _utcnow,
)

if TYPE_CHECKING:
    from stateloom.store.base import Store

logger = logging.getLogger("stateloom.experiment")


class ExperimentManager:
    """Central orchestrator for experiment lifecycle, assignment, and metrics."""

    def __init__(self, store: Store) -> None:
        self._store = store
        self._assigner = ExperimentAssigner(store)
        self._lock = threading.Lock()

    def create_experiment(
        self,
        name: str,
        variants: list[dict[str, Any] | VariantConfig],
        *,
        strategy: str | AssignmentStrategy = AssignmentStrategy.RANDOM,
        description: str = "",
        metadata: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> Experiment:
        """Create a new experiment in DRAFT status."""
        resolved_variants = [
            VariantConfig.from_dict(v) if isinstance(v, dict) else v for v in variants
        ]

        if isinstance(strategy, str):
            strategy = AssignmentStrategy(strategy)

        experiment = Experiment(
            name=name,
            description=description,
            strategy=strategy,
            variants=resolved_variants,
            metadata=metadata or {},
            agent_id=agent_id,
        )
        with self._lock:
            self._store.save_experiment(experiment)
        logger.info("Created experiment '%s' (%s)", name, experiment.id)
        return experiment

    def update_experiment(
        self,
        experiment_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        variants: list[dict[str, Any] | VariantConfig] | None = None,
        strategy: str | AssignmentStrategy | None = None,
        metadata: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> Experiment:
        """Update a DRAFT experiment. Raises ValueError if not in DRAFT status."""
        with self._lock:
            experiment = self._store.get_experiment(experiment_id)
            if experiment is None:
                raise ValueError(f"Experiment '{experiment_id}' not found")
            if experiment.status != ExperimentStatus.DRAFT:
                raise ValueError(
                    f"Cannot update experiment in '{experiment.status.value}' status; "
                    "only DRAFT experiments can be edited"
                )

            if name is not None:
                experiment.name = name
            if description is not None:
                experiment.description = description
            if variants is not None:
                experiment.variants = [
                    VariantConfig.from_dict(v) if isinstance(v, dict) else v for v in variants
                ]
            if strategy is not None:
                if isinstance(strategy, str):
                    strategy = AssignmentStrategy(strategy)
                experiment.strategy = strategy
            if metadata is not None:
                experiment.metadata = metadata
            if agent_id is not None:
                experiment.agent_id = agent_id

            experiment.updated_at = _utcnow()
            self._store.save_experiment(experiment)

        logger.info("Updated experiment '%s' (%s)", experiment.name, experiment.id)
        return experiment

    def start_experiment(self, experiment_id: str) -> Experiment:
        """Start an experiment — register with assigner, set RUNNING."""
        with self._lock:
            experiment = self._store.get_experiment(experiment_id)
            if experiment is None:
                raise ValueError(f"Experiment '{experiment_id}' not found")
            experiment.start()
            self._store.save_experiment(experiment)
            self._assigner.register(experiment)
        logger.info("Started experiment '%s'", experiment.name)
        return experiment

    def pause_experiment(self, experiment_id: str) -> Experiment:
        """Pause an experiment — unregister, set PAUSED."""
        with self._lock:
            experiment = self._store.get_experiment(experiment_id)
            if experiment is None:
                raise ValueError(f"Experiment '{experiment_id}' not found")
            experiment.pause()
            self._store.save_experiment(experiment)
            self._assigner.unregister(experiment_id)
        logger.info("Paused experiment '%s'", experiment.name)
        return experiment

    def conclude_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Conclude an experiment — unregister, set CONCLUDED, return final metrics."""
        with self._lock:
            experiment = self._store.get_experiment(experiment_id)
            if experiment is None:
                raise ValueError(f"Experiment '{experiment_id}' not found")
            experiment.conclude()
            self._store.save_experiment(experiment)
            self._assigner.unregister(experiment_id)
        logger.info("Concluded experiment '%s'", experiment.name)
        return self.get_metrics(experiment_id)

    def assign_session(
        self,
        session_id: str,
        experiment_id: str | None = None,
        variant_name: str | None = None,
    ) -> ExperimentAssignment | None:
        """Assign a session to an experiment variant."""
        return self._assigner.assign(
            session_id=session_id,
            experiment_id=experiment_id,
            variant_name=variant_name,
        )

    def record_feedback(
        self,
        session_id: str,
        rating: str,
        score: float | None = None,
        comment: str = "",
    ) -> None:
        """Record feedback for a session and emit a FeedbackEvent."""
        valid_ratings = {r.value for r in FeedbackRating}
        if rating not in valid_ratings:
            raise ValueError(
                f"Invalid rating '{rating}'. Must be one of: {', '.join(sorted(valid_ratings))}"
            )
        feedback = SessionFeedback(
            session_id=session_id,
            rating=rating,
            score=score,
            comment=comment,
        )
        self._store.save_feedback(feedback)

        event = FeedbackEvent(
            session_id=session_id,
            rating=rating,
            score=score,
            comment=comment,
        )
        self._store.save_event(event)
        logger.debug("Recorded feedback for session '%s': %s", session_id, rating)

    def get_metrics(self, experiment_id: str) -> dict[str, Any]:
        """Get per-variant aggregated metrics for an experiment."""
        return self._store.get_experiment_metrics(experiment_id)

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """Get cross-experiment variant ranking sorted by success_rate desc, avg_cost asc."""
        experiments = self._store.list_experiments()
        entries: list[dict[str, Any]] = []

        for exp in experiments:
            metrics = self._store.get_experiment_metrics(exp.id)
            for vname, vmetrics in metrics.get("variants", {}).items():
                entries.append(
                    {
                        "experiment_id": exp.id,
                        "experiment_name": exp.name,
                        "experiment_status": exp.status.value,
                        "variant_name": vname,
                        **vmetrics,
                    }
                )

        entries.sort(key=lambda e: (-e.get("success_rate", 0), e.get("avg_cost", float("inf"))))
        return entries

    def restore_running_experiments(self) -> None:
        """Re-register RUNNING experiments from store after restart."""
        experiments = self._store.list_experiments(status="running")
        for exp in experiments:
            self._assigner.register(exp)
            logger.info("Restored running experiment '%s'", exp.name)
