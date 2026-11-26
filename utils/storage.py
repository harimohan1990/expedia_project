"""Storage abstractions for PostgreSQL (with SQLite fallback), S3 (with local data lake fallback), and Redis."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import redis
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from configs.settings import PlatformConfig, get_config, get_logger

logger = get_logger(__name__)


class DatabaseClient:
    """Unified relational DB client supporting PostgreSQL with automatic local SQLite fallback."""

    def __init__(self, config: PlatformConfig | None = None) -> None:
        self.config = config or get_config()
        self.engine, self.backend = self._create_engine()

    def _create_engine(self) -> tuple[Engine, str]:
        pg_url = self.config.database.postgres_url
        try:
            engine = create_engine(pg_url, pool_pre_ping=True, connect_args={"connect_timeout": 2})
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Connected to PostgreSQL at %s", self.config.database.postgres_host)
            return engine, "postgresql"
        except Exception:
            sqlite_path = self.config.resolve_path(self.config.database.sqlite_fallback_path)
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(f"sqlite:///{sqlite_path}")
            logger.info("PostgreSQL unavailable; using local SQLite store at %s", sqlite_path)
            return engine, "sqlite"

    def write_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
        chunksize: int = 10_000,
    ) -> int:
        df.to_sql(
            table_name,
            con=self.engine,
            if_exists=if_exists,
            index=False,
            chunksize=chunksize,
            method="multi" if self.backend == "postgresql" else None,
        )
        logger.info("Stored %d rows into %s table '%s'", len(df), self.backend, table_name)
        return len(df)

    def read_table(self, table_name: str, limit: int | None = None) -> pd.DataFrame:
        query = f"SELECT * FROM {table_name}"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        return pd.read_sql(query, con=self.engine)

    def execute(self, sql: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(sql))


class S3DataLakeClient:
    """S3 client with transparent local filesystem data lake fallback when AWS credentials are absent."""

    def __init__(self, config: PlatformConfig | None = None) -> None:
        self.config = config or get_config()
        self.bucket = self.config.aws.s3_bucket
        self.local_root = self.config.resolve_path(self.config.aws.s3_local_fallback_dir)
        self.local_root.mkdir(parents=True, exist_ok=True)
        self._s3 = self._init_s3()

    def _init_s3(self) -> Any | None:
        try:
            session = boto3.Session(region_name=self.config.aws.region)
            creds = session.get_credentials()
            if creds is None:
                return None
            client = session.client("s3")
            client.head_bucket(Bucket=self.bucket)
            logger.info("Connected to AWS S3 bucket s3://%s", self.bucket)
            return client
        except Exception:
            return None

    def upload_path(self, local_path: Path, s3_prefix: str) -> str:
        """Upload a file or directory to S3 or mirror it into the local S3 data lake directory."""
        local_path = Path(local_path)
        if self._s3 is not None:
            if local_path.is_file():
                key = f"{s3_prefix.rstrip('/')}/{local_path.name}"
                self._s3.upload_file(str(local_path), self.bucket, key)
                return f"s3://{self.bucket}/{key}"
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    rel = file_path.relative_to(local_path)
                    key = f"{s3_prefix.rstrip('/')}/{rel.as_posix()}"
                    self._s3.upload_file(str(file_path), self.bucket, key)
            return f"s3://{self.bucket}/{s3_prefix.rstrip('/')}"

        target = self.local_root / self.bucket / s3_prefix.strip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        if local_path.is_file():
            target.mkdir(parents=True, exist_ok=True)
            dest = target / local_path.name
            shutil.copy2(local_path, dest)
            return str(dest)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(local_path, target)
        return str(target)


class CacheClient:
    """Redis cache client with graceful in-memory fallback for local development and tests."""

    def __init__(self, config: PlatformConfig | None = None) -> None:
        self.config = config or get_config()
        self._memory: dict[str, str] = {}
        self._redis = self._connect()

    def _connect(self) -> redis.Redis | None:
        try:
            client = redis.Redis(
                host=self.config.redis.host,
                port=self.config.redis.port,
                db=self.config.redis.db,
                socket_connect_timeout=1,
                decode_responses=True,
            )
            client.ping()
            logger.info(
                "Connected to Redis at %s:%s", self.config.redis.host, self.config.redis.port
            )
            return client
        except Exception:
            logger.info("Redis unavailable; using in-process memory cache fallback")
            return None

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "in-memory"

    def get_json(self, key: str) -> Any | None:
        raw = self._redis.get(key) if self._redis is not None else self._memory.get(key)
        return json.loads(raw) if raw is not None else None

    def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        payload = json.dumps(value)
        ttl = ttl_seconds or self.config.redis.cache_ttl_seconds
        if self._redis is not None:
            self._redis.setex(key, ttl, payload)
        else:
            self._memory[key] = payload
