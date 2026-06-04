"""
MLflow experiment tracking for RAG queries.

``ExperimentTracker`` logs each query as an MLflow run containing
parameters (question, mode, sources), metrics (latency, tokens,
confidence), and tags (cache_hit, username).

If the ``mlflow`` package is not installed or the tracking server is
unreachable, every method degrades to a no-op with a warning.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe MLflow import
# ---------------------------------------------------------------------------

try:
    import mlflow  # type: ignore[import-untyped]
    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_AVAILABLE = False
    logger.info("mlflow not installed – ExperimentTracker will no-op")


class ExperimentTracker:
    """Log RAG queries and configuration to MLflow.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking server URI (local path or HTTP URL).
    experiment_name:
        Name of the MLflow experiment.  Created if it does not exist.
    """

    def __init__(
        self,
        tracking_uri: str = "./mlruns",
        experiment_name: str = "psi-rag",
    ) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._ready = False

        if not _MLFLOW_AVAILABLE:
            return

        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._ready = True
            logger.info(
                "MLflow tracking initialised (uri=%s, experiment=%s)",
                tracking_uri,
                experiment_name,
            )
        except Exception as exc:
            logger.warning("MLflow setup failed – tracking disabled: %s", exc)

    # -- public API ----------------------------------------------------------

    def log_query(
        self,
        question: str,
        answer: str,
        confidence: float,
        tokens: int,
        latency_ms: float,
        cache_hit: bool = False,
        mode: str = "rag",
        sources: Optional[Sequence[str]] = None,
    ) -> None:
        """Log a single RAG query as an MLflow run.

        Parameters
        ----------
        question:
            User question text.
        answer:
            Generated answer text.
        confidence:
            Model confidence score (0-1).
        tokens:
            Total tokens consumed.
        latency_ms:
            End-to-end latency in milliseconds.
        cache_hit:
            Whether the answer came from cache.
        mode:
            Response mode (``rag``, ``direct``, ``conversational``…).
        sources:
            Source document names used for retrieval.
        """
        if not self._ready:
            return

        try:
            with mlflow.start_run(nested=True):
                # Parameters (string-valued)
                mlflow.log_param("question", question[:250])
                mlflow.log_param("mode", mode)
                mlflow.log_param("cache_hit", str(cache_hit))
                if sources:
                    mlflow.log_param("sources", ", ".join(sources[:10]))

                # Metrics (numeric)
                mlflow.log_metric("confidence", confidence)
                mlflow.log_metric("tokens", tokens)
                mlflow.log_metric("latency_ms", latency_ms)

                # Tags
                mlflow.set_tag("answer_preview", answer[:200])

        except Exception as exc:
            logger.warning("MLflow log_query failed: %s", exc)

    def log_config(self, settings_dict: Dict[str, Any]) -> None:
        """Log application configuration as MLflow params.

        Only safe (non-secret) keys are logged.  Any key containing
        ``key``, ``secret``, or ``password`` (case-insensitive) is
        automatically redacted.

        Parameters
        ----------
        settings_dict:
            Flat dict of settings, e.g. ``settings.model_dump()``.
        """
        if not self._ready:
            return

        _REDACT_KEYWORDS = {"key", "secret", "password", "token"}

        try:
            with mlflow.start_run(nested=True, run_name="config-snapshot"):
                for k, v in settings_dict.items():
                    # Redact sensitive values
                    if any(word in k.lower() for word in _REDACT_KEYWORDS):
                        continue
                    # MLflow params must be strings ≤ 500 chars
                    mlflow.log_param(k, str(v)[:500])
        except Exception as exc:
            logger.warning("MLflow log_config failed: %s", exc)
