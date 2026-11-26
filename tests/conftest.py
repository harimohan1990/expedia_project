"""Shared Pytest fixtures for fast isolated unit and integration tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from configs.settings import PlatformConfig
from scripts.generate_synthetic_data import generate_all_datasets


@pytest.fixture(scope="session")
def test_config(tmp_path_factory: pytest.TempPathFactory) -> PlatformConfig:
    """Create an isolated temporary PlatformConfig for end-to-end tests."""
    root = tmp_path_factory.mktemp("expedia_ad_test")
    cfg = PlatformConfig()
    cfg.paths.synthetic_data_dir = str(root / "synthetic")
    cfg.paths.raw_data_dir = str(root / "raw")
    cfg.paths.processed_data_dir = str(root / "processed")
    cfg.paths.models_dir = str(root / "artifacts" / "models")
    cfg.paths.reports_dir = str(root / "artifacts" / "reports")
    cfg.paths.mlflow_tracking_uri = f"sqlite:///{root / 'mlflow.db'}"
    cfg.aws.s3_local_fallback_dir = str(root / "s3_datalake")
    cfg.database.sqlite_fallback_path = str(root / "processed" / "test_ads.db")
    cfg.data_generation.num_inventory_days = 180
    return cfg


@pytest.fixture(scope="session")
def synthetic_dataset_paths(test_config: PlatformConfig) -> dict[str, Path]:
    """Generate a scaled-down realistic synthetic dataset once per test session."""
    return generate_all_datasets(config=test_config, scale=0.02)


@pytest.fixture(scope="session")
def experiment_df(synthetic_dataset_paths: dict[str, Path]) -> pd.DataFrame:
    return pd.read_csv(synthetic_dataset_paths["experiment_events"])
