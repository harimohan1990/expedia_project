"""Centralized SparkSession builder configured for Java 17/21 and local/S3 execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

from configs.settings import PlatformConfig, get_config, get_logger

logger = get_logger(__name__)


def _ensure_java_home() -> None:
    """Ensure JAVA_HOME points to a valid JDK on macOS/Linux if unset."""
    if os.environ.get("JAVA_HOME") and Path(os.environ["JAVA_HOME"]).exists():
        return
    candidates = [
        "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home",
        "/opt/homebrew/opt/openjdk@21",
        "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
        "/usr/lib/jvm/java-17-openjdk-amd64",
    ]
    for cand in candidates:
        if Path(cand).exists():
            os.environ["JAVA_HOME"] = cand
            os.environ["PATH"] = f"{cand}/bin:" + os.environ.get("PATH", "")
            logger.info("Configured JAVA_HOME=%s", cand)
            return


def create_spark_session(
    config: PlatformConfig | None = None,
    app_name: str | None = None,
) -> SparkSession:
    """Build or retrieve a configured SparkSession with Adaptive Query Execution."""
    _ensure_java_home()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    cfg = config or get_config()
    java_opts = (
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
    )
    builder = (
        SparkSession.builder.appName(app_name or cfg.spark.app_name)
        .master(cfg.spark.master)
        .config("spark.driver.memory", cfg.spark.driver_memory)
        .config("spark.sql.shuffle.partitions", str(cfg.spark.shuffle_partitions))
        .config("spark.sql.adaptive.enabled", str(cfg.spark.sql_adaptive_enabled).lower())
        .config("spark.driver.extraJavaOptions", java_opts)
        .config("spark.executor.extraJavaOptions", java_opts)
        .config("spark.ui.showConsoleProgress", "false")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
