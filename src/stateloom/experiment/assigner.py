"""Experiment assigner — assigns sessions to experiment variants."""

from __future__ import annotations

import hashlib
import logging
import random
import threading
from typing import TYPE_CHECKING

from stateloom.core.types import AssignmentStrategy, ExperimentStatus
from stateloom.experiment.models import Experiment, ExperimentAssignment, VariantConfig

if TYPE_CHECKING:
    from stateloom.store.base import Store

logger = logging.getLogger("stateloom.experiment")


class ExperimentAssigner:
    """Assigns sessions to experiment variants using configured strategies."""

    def __init__(self, store: Store) -> None:
        self._store = store
        self._running: dict[str, Experiment] = {}
        self._lock = threading.Lock()

    def register(self, experiment: Experiment) -> None:
        """Register a running experiment for assignment."""
        with self._lock:
            self._running[experiment.id] = experiment

    def unregister(self, experiment_id: str) -> None:
        """Unregister an experiment (paused/concluded)."""
        with self._lock:
            self._running.pop(experiment_id, None)

    def assign(
        self,
        session_id: str,
        experiment_id: str | None = None,
        variant_name: str | None = None,
    ) -> ExperimentAssignment | None:
        """Assign a session to an experiment variant.

        Returns None if no running experiment matches or if session already assigned.
        Thread-safe: the entire check-and-assign flow is serialized.
        """
        with self._lock:
            # Check for existing assignment (inside lock to prevent TOCTOU race).
            # Only return early if the assignment belongs to the same experiment —
            # a session may be reused across different experiments.
            existing = self._store.get_assignment(session_id)
            if existing is not None and (
                experiment_id is None or existing.experiment_id == experiment_id
            ):
                return existing

            if not self._running:
                return None

            # Pick the experiment
            if experiment_id:
                experiment = self._running.get(experiment_id)
                if experiment is None:
                    return None
            else:  # no explicit experiment_id — use the first running
                experiment = next(iter(self._running.values()))

            if experiment.status != ExperimentStatus.RUNNING:
                return None

            # Pick the variant
            variant = self._pick_variant(experiment, session_id, variant_name)
            if variant is None:
                return None

            # Create the assignment with a config snapshot
            assignment = ExperimentAssignment.create(
                session_id=session_id,
                experiment_id=experiment.id,
                variant=variant,
            )

            # Snapshot agent version overrides at assignment time (immutable)
            if variant.agent_version_id:
                try:
                    version = self._store.get_agent_version(variant.agent_version_id)
                    if version:
                        assignment.variant_config["_resolved_agent_overrides"] = {
                            "model": version.model,
                            "system_prompt": version.system_prompt,
                            "request_overrides": version.request_overrides or {},
                        }
                        assignment.variant_config["_agent_meta"] = {
                            "agent_id": version.agent_id,
                            "agent_version_id": version.id,
                            "agent_version_number": version.version_number,
                        }
                except Exception:
                    logger.debug("Agent snapshot failed at assignment time", exc_info=True)

            # Persist and record
            self._store.save_assignment(assignment)
            experiment.record_assignment(variant.name)
            self._store.save_experiment(experiment)

            return assignment

    def _pick_variant(
        self,
        experiment: Experiment,
        session_id: str,
        variant_name: str | None = None,
    ) -> VariantConfig | None:
        """Pick a variant using the experiment's strategy."""
        variants = experiment.variants
        if not variants:
            return None

        strategy = experiment.strategy

        if strategy == AssignmentStrategy.MANUAL:
            if variant_name is None:
                return None
            for v in variants:
                if v.name == variant_name:
                    return v
            return None

        if variant_name:
            # Explicit override regardless of strategy
            for v in variants:
                if v.name == variant_name:
                    return v
            return None

        if strategy == AssignmentStrategy.RANDOM:
            weights = [v.weight for v in variants]
            return random.choices(variants, weights=weights, k=1)[0]

        if strategy == AssignmentStrategy.HASH:
            key = f"{session_id}:{experiment.id}"
            h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
            total_weight = sum(v.weight for v in variants)
            point = (h % 10000) / 10000 * total_weight
            cumulative = 0.0
            for v in variants:
                cumulative += v.weight
                if point < cumulative:
                    return v
            return variants[-1]

        return None
