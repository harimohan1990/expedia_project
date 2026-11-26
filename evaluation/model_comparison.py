"""Common MLflow experiment tracking, model comparison, and registry promotion framework."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from configs.settings import PlatformConfig, get_config, get_logger

logger = get_logger(__name__)


@dataclass
class CandidateResult:
    model_name: str
    task_type: str  # "regression" or "classification"
    metrics: dict[str, float]
    params: dict[str, Any]
    model_obj: Any
    run_id: str | None = None


class ModelEvaluator:
    """Reusable evaluation & MLflow tracking orchestrator across all forecasting models."""

    def __init__(self, experiment_name: str, config: PlatformConfig | None = None) -> None:
        self.config = config or get_config()
        self.experiment_name = experiment_name
        tracking_uri = self.config.paths.mlflow_tracking_uri
        if tracking_uri.startswith("sqlite:///"):
            db_rel = tracking_uri.replace("sqlite:///", "")
            db_abs = self.config.resolve_path(db_rel)
            tracking_uri = f"sqlite:///{db_abs}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def log_and_compare(
        self,
        candidates: list[CandidateResult],
        primary_metric: str,
        higher_is_better: bool = False,
        registered_model_name: str | None = None,
        local_save_dir: Path | None = None,
        extra_artifacts: dict[str, Any] | None = None,
    ) -> tuple[CandidateResult, pd.DataFrame]:
        """Log all candidate models to MLflow, rank them, save best model, and register/promote it."""
        rows: list[dict[str, Any]] = []
        save_dir = local_save_dir or (
            self.config.resolve_path(self.config.paths.models_dir) / self.experiment_name
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        for cand in candidates:
            with mlflow.start_run(run_name=cand.model_name) as run:
                cand.run_id = run.info.run_id
                mlflow.log_params({k: str(v) for k, v in cand.params.items()})
                mlflow.log_metrics(cand.metrics)

                artifact_file = save_dir / f"{cand.model_name}.pkl"
                with open(artifact_file, "wb") as f:
                    pickle.dump(cand.model_obj, f)
                mlflow.log_artifact(str(artifact_file), artifact_path="model_pickle")

                if extra_artifacts:
                    for name, data in extra_artifacts.items():
                        art_path = save_dir / name
                        if isinstance(data, pd.DataFrame):
                            data.to_csv(art_path, index=False)
                        else:
                            art_path.write_text(
                                json.dumps(data, indent=2, default=str), encoding="utf-8"
                            )
                        mlflow.log_artifact(str(art_path), artifact_path="evaluation_artifacts")

            row = {"model_name": cand.model_name, "run_id": cand.run_id, **cand.metrics}
            rows.append(row)

        comparison_df = (
            pd.DataFrame(rows)
            .sort_values(primary_metric, ascending=not higher_is_better)
            .reset_index(drop=True)
        )

        best_name = comparison_df.iloc[0]["model_name"]
        best_cand = next(c for c in candidates if c.model_name == best_name)

        # Save canonical best_model.pkl + comparison report locally for FastAPI/Dashboard loading
        with open(save_dir / "best_model.pkl", "wb") as f:
            pickle.dump(best_cand.model_obj, f)

        summary = {
            "experiment_name": self.experiment_name,
            "best_model": best_cand.model_name,
            "best_run_id": best_cand.run_id,
            "primary_metric": primary_metric,
            "best_metrics": best_cand.metrics,
            "comparison": comparison_df.to_dict(orient="records"),
        }
        (save_dir / "evaluation_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        comparison_df.to_csv(save_dir / "model_comparison.csv", index=False)

        # Register in MLflow Model Registry & set alias "champion"
        if registered_model_name and best_cand.run_id:
            try:
                model_uri = f"runs:/{best_cand.run_id}/model_pickle"
                mv = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
                self.client.set_registered_model_alias(
                    name=registered_model_name, alias="champion", version=mv.version
                )
                logger.info(
                    "Registered '%s' version=%s (alias=champion, %s=%.4f)",
                    registered_model_name,
                    mv.version,
                    primary_metric,
                    best_cand.metrics[primary_metric],
                )
            except Exception as exc:
                logger.warning("MLflow model registry step skipped (%s)", exc)

        return best_cand, comparison_df
