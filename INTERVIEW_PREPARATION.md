# Senior Applied Scientist (ML) Interview Preparation Guide
### Connected to the Expedia Travel Advertising Intelligence & Forecasting Platform

---

## Section 1: Python & SQL (50 Questions)

1. **How do you prevent SQL N+1 or row-explosion bugs when joining `campaigns` with `ad_impressions` and `targeting_signals`?**  
   Aggregate or deduplicate child tables to the target grain first (e.g., latest signal per `traveler_id` via `ROW_NUMBER() OVER (PARTITION BY traveler_id ORDER BY signal_timestamp DESC)` as implemented in `spark/transform.py`) before joining.

2. **Write a SQL query to compute 7-day rolling available inventory per `(country, placement)` without lookahead.**  
   ```sql
   SELECT country, placement, date, available_impressions,
          AVG(available_impressions) OVER (
              PARTITION BY country, placement
              ORDER BY date
              ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
          ) AS rolling_7d_prev_avg
   FROM ad_inventory;
   ```

3. **Why did we use `ROWS BETWEEN` instead of `RANGE BETWEEN` in window functions for daily inventory?**  
   `ROWS BETWEEN` counts exact physical row offsets even if dates have gaps, whereas `RANGE` operates on value intervals. When daily continuity is verified in ETL, `ROWS` is deterministic and faster.

4. **How does `pandas.DataFrame.shift(1).expanding().mean()` prevent leakage in `features/audience_features.py`?**  
   `shift(1)` excludes the current campaign row from its own expanding historical average, guaranteeing only strictly prior campaigns contribute to `historical_unique_reach`.

5. **What is the difference between `RANK()`, `DENSE_RANK()`, and `ROW_NUMBER()` when picking top campaigns by ROAS?**  
   `ROW_NUMBER()` assigns unique integers (arbitrary tie-break), `RANK()` leaves gaps on ties (1,1,3), and `DENSE_RANK()` leaves no gaps (1,1,2).

6. **How do you vectorize synthetic generation of 1,000,000 rows in NumPy without Python loops?**  
   In `scripts/generate_synthetic_data.py`, we index NumPy arrays via vectorized integer indices (`rng.choice`, boolean masks, `np.where`, `np.power`) instead of iterating rows, generating 1M impressions in seconds.

7. **Explain why `defaultdict` or `lru_cache` can cause subtle bugs in long-running FastAPI services.**  
   `@lru_cache` on `get_config()` or `_load_reach_bundle()` retains objects in memory indefinitely; if a new `@champion` model is promoted in MLflow without restarting or invalidating the cache, stale models stay in RAM.

8. **How do you compute weighted average CPM in SQL safely avoiding division-by-zero?**  
   `SUM(impression_cost) / NULLIF(COUNT(*), 0) * 1000.0`.

9. **What is the difference between `COUNT(*)`, `COUNT(col)`, and `COUNT(DISTINCT col)`?**  
   `COUNT(*)` counts all rows including NULLs; `COUNT(col)` ignores NULLs in `col`; `COUNT(DISTINCT col)` counts unique non-null values (used for `unique_reach`).

10. **Why use Pydantic `BaseSettings` (`configs/settings.py`) instead of `os.getenv` scattered across files?**  
    Pydantic validates types at startup, centralizes defaults from `configs/config.yaml`, and supports nested env overrides (`EXPEDIA_AD_DATABASE__POSTGRES_HOST`).

11. **Write a SQL CTE to detect advertisers whose CTR dropped >20% week-over-week.**  
    Compute weekly `(clicks, impressions, ctr)` in a `weekly_perf` CTE, apply `LAG(ctr) OVER (PARTITION BY advertiser_id ORDER BY week_start)`, and filter `WHERE ctr < 0.8 * prev_ctr`.

12. **Explain Python's GIL and why XGBoost/LightGBM/PySpark still parallelize across CPU cores.**  
    C/C++ extensions (`xgboost`, `lightgbm`, `numpy`) release the GIL during OpenMP/BLAS multi-threaded computation, and PySpark executes JVM tasks in separate worker processes.

13. **How do you stream large CSVs in Python without exhausting memory?**  
    Use `pd.read_csv(..., chunksize=50_000)` or generators (`yield`), or delegate directly to Spark/PyArrow lazy execution.

14. **Difference between `LEFT JOIN ... WHERE right.id IS NULL` and `NOT EXISTS` in SQL?**  
    `NOT EXISTS` short-circuits on the first match and handles NULLs safely without pitfalls of `NOT IN` (which returns empty if any NULL exists).

15. **How do you compute campaign reach saturation (`unique_reach / impressions`) cleanly in SQL?**  
    `COUNT(DISTINCT traveler_id) * 1.0 / NULLIF(COUNT(impression_id), 0) AS reach_efficiency`.

16. **Why does `to_sql(..., method="multi", chunksize=10_000)` speed up PostgreSQL loads in `utils/storage.py`?**  
    It batches 10,000 rows into multi-value `INSERT INTO ... VALUES (...), (...)` statements, minimizing network round-trips.

17. **What are Python dataclasses vs Pydantic BaseModels?**  
    `dataclasses` (`DataQualityReport` in `spark/transform.py`) have near-zero overhead for internal typed containers; Pydantic `BaseModel` (`api/schemas/models.py`) provides runtime coercion, validation, and OpenAPI schema generation.

18. **How do you write an idempotent SQL upsert for daily forecast tables?**  
    `INSERT INTO pred_inventory (...) VALUES (...) ON CONFLICT (date, country, placement) DO UPDATE SET predicted_available_impressions = EXCLUDED.predicted_available_impressions;`.

19. **Explain generator expressions vs list comprehensions in Python memory profiling.**  
    List comprehensions materialize all elements in RAM (`O(N)`); generators yield lazily (`O(1)`), e.g., `sum(1 for _ in open(src_path, "rb"))` in `ingestion/batch_ingestion.py`.

20. **How do you calculate percentiles (median, p95 API latency) in PostgreSQL?**  
    `PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)`.

21. **Why sort dataframes before `groupby().expanding()` in Pandas?**  
    Pandas `expanding()` respects current row order, not index/date values; failing to `sort_values(["start_date", "campaign_id"])` causes future rows to leak into earlier averages.

22. **How do you debug a slow SQL query on a 100M-row `ad_impressions` table?**  
    Run `EXPLAIN (ANALYZE, BUFFERS)`, inspect sequential scans vs index/partition pruning, check join order & cardinality estimates, and add composite B-tree/BRIN indexes on `(event_date, campaign_id)`.

23. **What is the purpose of `from __future__ import annotations` across our modules?**  
    Enables PEP 563 postponed evaluation of type annotations, supporting modern `X | Y` union syntax cleanly and avoiding circular import overhead.

24. **How do you implement deterministic hashing for Redis cache keys (`api/routes/forecasting.py`)?**  
    Serialize payload with `json.dumps(payload, sort_keys=True)` and hash with `hashlib.sha256` so key order variations hit the same cache entry.

25. **Write a SQL query to find the top 3 placements by conversion rate within each country.**  
    Aggregate `(country, placement)` CVR in a CTE, rank with `DENSE_RANK() OVER (PARTITION BY country ORDER BY cvr DESC) AS rnk`, and filter `WHERE rnk <= 3`.

26. **What is a Python context manager and where do we use it?**  
    Implements `__enter__`/`__exit__` (or `@contextmanager`); used for SQLAlchemy transactions (`with self.engine.begin() as conn:`), MLflow runs (`with mlflow.start_run():`), and file I/O.

27. **How do you handle timezone-naive vs timezone-aware timestamps in Python?**  
    Standardize internal timestamps to UTC (`datetime.now(timezone.utc)`) or explicit `pd.to_datetime(..., utc=True)` before storing or comparing.

28. **Explain SQL `COALESCE` vs `NULLIF`.**  
    `COALESCE(a, b)` returns the first non-null argument (used to impute default `travel_intent`); `NULLIF(a, 0)` returns NULL if `a == 0` (prevents ZeroDivisionError).

29. **How do you avoid `SettingWithCopyWarning` in Pandas?**  
    Use `.copy()` explicitly when subsetting (`train_df = ordered[mask].copy()`) and `.loc[rows, col] = val` for assignments.

30. **What is the time complexity of `set(impressions["traveler_id"]).issubset(set(travelers["traveler_id"]))`?**  
    `O(N + M)` average time using hash sets, versus `O(N * M)` with naive list membership checks.

31. **How do you structure a SQL query to find users exposed to >5 impressions without clicking (ad fatigue)?**  
    `SELECT traveler_id, campaign_id FROM ad_impressions GROUP BY 1, 2 HAVING COUNT(*) > 5 AND SUM(clicked) = 0;`

32. **Explain Python `time.perf_counter()` vs `time.time()` for API latency measurement.**  
    `perf_counter()` is monotonic and highest-resolution (unaffected by NTP wall-clock jumps), making it ideal for measuring `latency_ms` in FastAPI middleware.

33. **How do you efficiently filter a multi-gigabyte Parquet dataset in Python without Spark?**  
    Use PyArrow predicate pushdown & column projection (`pd.read_parquet(..., columns=[...], filters=[("country", "==", "US")])`).

34. **What does `np.clip(arr, min_val, max_val)` prevent in probability/CTR predictions?**  
    Ensures linear/tree outputs never yield impossible probabilities (`< 0` or `> 1`) or negative impression counts.

35. **Write SQL to compute Campaign ROAS and filter campaigns spending > \$10,000 with ROAS < 1.0.**  
    ```sql
    SELECT campaign_id, SUM(impression_cost) AS spend,
           SUM(CASE WHEN converted = 1 THEN 145.0 ELSE 0 END) AS rev,
           SUM(CASE WHEN converted = 1 THEN 145.0 ELSE 0 END) / NULLIF(SUM(impression_cost), 0) AS roas
    FROM ad_impressions
    GROUP BY campaign_id
    HAVING SUM(impression_cost) > 10000 AND rev < spend;
    ```

36. **Why use `typing.Any` sparingly and prefer `TypedDict` or `BaseModel`?**  
    Strong types catch schema mismatches in IDE/CI static checks before production runtime failures.

37. **How do SQL CTEs differ from temporary tables in query optimization?**  
    CTEs improve readability and may be inlined by the planner; materialized temp tables persist intermediate results with statistics/indexes for multi-step reuse.

38. **Explain `pd.qcut` vs `pd.cut` used in uplift decile analysis (`causal_inference/uplift_modeling.py`).**  
    `qcut` creates equal-frequency quantile bins (10% of users per decile); `cut` creates equal-width value intervals.

39. **How do you safely deserialize JSON in high-throughput APIs?**  
    Validate strictly through Pydantic schemas with `min_length`/`max_length` bounds (`items: list[...] = Field(..., max_length=500)`) to prevent OOM payloads.

40. **What is the difference between `INNER JOIN`, `LEFT SEMI JOIN`, and `LEFT ANTI JOIN`?**  
    `INNER` returns matched columns from both; `LEFT SEMI` filters left rows having a match without duplicating columns (used in `clean_bidding_data`); `LEFT ANTI` returns orphan rows for data-quality alerting.

41. **How do you test functions that interact with external DBs or S3 without cloud credentials?**  
    Dependency injection of `PlatformConfig` pointing to pytest `tmp_path_factory` (`tests/conftest.py`) and local SQLite/filesystem fallbacks (`utils/storage.py`).

42. **Explain `async def` vs sync `def` route handlers in FastAPI.**  
    FastAPI runs sync `def` endpoints (like CPU-bound sklearn/XGBoost inference) in an AnyIO threadpool so they do not block the main asyncio event loop!

43. **How do you compute cumulative campaign spend over time in SQL?**  
    `SUM(impression_cost) OVER (PARTITION BY campaign_id ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)`.

44. **Why set `random_state` / `np.random.default_rng(seed)` everywhere?**  
    Ensures exact reproducibility across synthetic data generation, train/test splitting, and tree subsampling.

45. **What is the difference between `isinstance` and `issubclass`, and why prefer duck typing / protocols?**  
    Protocols define structural interfaces (e.g., `.fit()` / `.predict()`) without forcing rigid inheritance across `sklearn`, `xgboost`, `lightgbm`, and `catboost`.

46. **How do you avoid Cartesian joins in SQL when joining daily inventory and campaign budgets?**  
    Aggregate campaigns to `(country, date)` grain first (`camp_daily` in `features/inventory_features.py`) before merging onto inventory.

47. **Explain why `pickle` has security & version-coupling risks in model serving.**  
    Pickle executes arbitrary code on untrusted files and breaks across major `scikit-learn`/`numpy` upgrades; mitigate via trusted internal MLflow artifact stores, pinned dependencies, and formats like ONNX/XGBoost JSON where possible.

48. **How do you calculate Year-over-Year inventory growth in SQL?**  
    `LAG(available_impressions, 365) OVER (PARTITION BY country, placement ORDER BY date)` or self-join on `date = prev.date + INTERVAL '365 days'`.

49. **Why use `Path` (`pathlib`) instead of string path concatenation?**  
    OS-independent path handling, cleaner `.mkdir(parents=True, exist_ok=True)`, `.exists()`, `.glob()`, and `.relative_to()`.

50. **How do you profile a slow Python feature engineering script?**  
    Use `cProfile` + `snakeviz`, or `py-spy` sampling profiler, and replace row-wise `.apply(axis=1)` with vectorized NumPy/Pandas or PySpark operations.

---

## Section 2: PySpark & Distributed Data Engineering (40 Questions)

51. **Why define explicit `StructType` schemas (`spark/extract.py`) instead of `inferSchema=True`?**  
    `inferSchema=True` triggers an extra full pass over raw CSVs, slows ingestion, and can silently infer wrong types (e.g., `int` instead of `double` or `string` IDs) causing downstream pipeline breaks.

52. **Explain Spark's Adaptive Query Execution (`spark.sql.adaptive.enabled=true`) in `spark/spark_session.py`.**  
    AQE dynamically coalesces small shuffle partitions, converts sort-merge joins to broadcast joins at runtime, and optimizes skewed joins using actual stage runtime statistics.

53. **Narrow vs. Wide transformations in our ETL (`spark/transform.py`)?**  
    Narrow (`filter`, `withColumn`, `to_timestamp`) process each partition independently without network shuffle; Wide (`dropDuplicates`, `join`, `groupBy`, `Window.partitionBy`) trigger shuffle across executors.

54. **How does `left_semi` join verify referential integrity efficiently in `clean_bidding_data`?**  
    It checks existence of `campaign_id` in `campaigns_df` without carrying right-table columns or multiplying rows, minimizing memory and shuffle payload.

55. **Why partition `ad_impressions_enriched` Parquet by `(year, month)` and `ad_inventory` by `country`?**  
    Enables **partition pruning** during time-range or market-specific queries, skipping irrelevant directories on S3/disk.

56. **How would you handle data skew if `US` accounts for 30%+ of all impressions in a large Spark cluster?**  
    Enable AQE skew join optimization (`spark.sql.adaptive.skewJoin.enabled=true`), salt skewed join keys (`concat(country, '_', floor(rand()*10))`), or broadcast small dimension tables (`campaigns`, `travelers` if small).

57. **Explain `broadcast()` join vs Sort-Merge Join in Spark.**  
    Broadcast join sends a small dimension table (<10MB default, tunable via `autoBroadcastJoinThreshold`) to every executor, eliminating the expensive shuffle of the 1M+/1B+ fact table (`ad_impressions`).

58. **How does Spark Structured Streaming handle late-arriving ad click events (`spark/streaming.py`)?**  
    Using `.withWatermark("event_timestamp", "10 minutes")`: Spark tracks max observed event time and retains window state for 10 minutes past window end, dropping events older than the watermark while updating late-arriving ones within the boundary.

59. **Compare `append`, `update`, and `complete` output modes in Structured Streaming.**  
    `update` (used in `spark/streaming.py`) emits only window rows updated in the micro-batch; `append` waits until watermark passes window end before emitting finalized rows; `complete` outputs the entire aggregated table every trigger.

60. **What is the role of `checkpointLocation` in Structured Streaming?**  
    Persists Kafka offsets and intermediate aggregation state to fault-tolerant storage (S3/HDFS) for **exactly-once** end-to-end recovery after driver/executor failures.

61. **Why use `approx_count_distinct("traveler_id")` (HyperLogLog) in streaming instead of exact `countDistinct`?**  
    Exact `countDistinct` over unbounded users requires huge state memory; HyperLogLog estimates unique reach with ~1-2% relative error in constant `O(1)` memory per window.

62. **How do you make a batch PySpark ETL pipeline idempotent (`spark/run_etl.py`)?**  
    Track processed watermarks (`_etl_watermark.json`), use deterministic primary-key deduplication, and write to partitioned targets using `mode("overwrite")` or dynamic partition overwrite (`partitionOverwriteMode=dynamic`).

63. **What causes `OutOfMemoryError` on the Spark driver when calling `.toPandas()`?**  
    `.toPandas()` collects all partitions to the single driver JVM/Python process. In `spark/run_etl.py`, we only call `.toPandas()` on small **aggregated** tables, never on raw 1M+ `ad_impressions`.

64. **Explain Catalyst Optimizer's logical vs physical planning phases.**  
    Unresolved Logical Plan -> Analyzed Logical Plan (Catalog resolution) -> Optimized Logical Plan (predicate pushdown, constant folding, projection pruning) -> Multiple Physical Plans -> Cost-based selection -> Whole-Stage Codegen.

65. **How does `spark.sql.shuffle.partitions` affect performance on local vs EMR clusters?**  
    Default `200` creates 200 tiny tasks on local/medium datasets (high scheduling overhead). Setting `16` locally (`configs/config.yaml`) speeds up small/medium jobs; on EMR, scale to ~2–4× total executor cores.

66. **Compare `cache()` / `persist()` vs recomputation in Spark.**  
    Since `transformed["ad_impressions_enriched"]` feeds 7 different business aggregations in `spark/aggregations.py`, caching it in `MEMORY_AND_DISK` avoids re-reading and re-joining raw CSVs 7 times.

67. **How do you handle corrupt/malformed CSV records in Spark `extract.py`?**  
    Use `.option("mode", "PERMISSIVE")` with a `_corrupt_record` column (or filter invalid types/ranges explicitly in `transform.py` and track counts in `DataQualityReport`).

68. **Why prefer Parquet over CSV in an S3 data lake?**  
    Parquet is columnar, compressed (Snappy/Zstd), preserves schema metadata, and supports column pruning + min/max statistics pushdown, cutting S3 I/O by 5–20×.

69. **Explain `repartition(n)` vs `coalesce(n)`.**  
    `repartition` triggers a full shuffle to evenly distribute data (can increase or decrease partitions); `coalesce` merges existing partitions without a full shuffle (only decreases, ideal before writing final files).

70. **How do you avoid the "small files problem" in S3 when running Structured Streaming?**  
    Increase trigger interval (e.g., 5 minutes), `coalesce` before writing, or run periodic S3 compaction jobs (Iceberg/Delta `OPTIMIZE` or Spark batch rewrite).

71. **What is the difference between `F.row_number()` and `F.dense_rank()` in `clean_targeting_signals`?**  
    `row_number()` guarantees strictly 1 row per `traveler_id` even if two signals share the identical latest timestamp.

72. **How do you implement incremental batch processing in PySpark (`spark/run_etl.py`)?**  
    Store `max_impression_timestamp` in `_etl_watermark.json` after each successful run, filter incoming facts `WHERE timestamp > prev_watermark`, and merge updated partitions.

73. **Why did we configure `--add-opens=java.base/...` in `spark/spark_session.py`?**  
    Java 17/21 enforces strong module encapsulation (JEP 403); PySpark's Kryo/Arrow/Netty internals require `--add-opens` flags to access `java.nio` and `java.lang` internals cleanly.

74. **Explain Tungsten memory management and off-heap execution.**  
    Tungsten stores binary rows directly in `sun.misc.Unsafe` / off-heap byte arrays instead of Java objects, eliminating GC pauses and improving CPU cache locality.

75. **How does `foreachBatch` in `spark/streaming.py` enable ML model scoring in Structured Streaming?**  
    `foreachBatch` exposes each micro-batch as a standard static `DataFrame`, allowing batch scoring, logging to Parquet/PostgreSQL, or invoking an MLflow/FastAPI endpoint.

76. **What is dynamic allocation in Spark on AWS EMR?**  
    `spark.dynamicAllocation.enabled=true` automatically scales executors up during heavy shuffle stages and down during idle periods to save EC2 costs.

77. **Explain `when(...).otherwise(...)` vs Python UDFs in PySpark.**  
    Built-in `pyspark.sql.functions` (`when`, `coalesce`, `datediff`) execute natively inside JVM Tungsten codegen; Python UDFs serialize rows across Py4J sockets to Python workers, incurring 10–100× slowdown.

78. **How do you join two streaming DataFrames in Spark Structured Streaming?**  
    Require watermarks on both streams AND an event-time range condition (e.g., `imp_ts BETWEEN sig_ts AND sig_ts + INTERVAL 1 HOUR`) so Spark can bound state store eviction.

79. **How do you debug a Spark stage stuck at 99% (199/200 tasks complete)?**  
    Classic straggler/data skew: inspect Spark UI Stage Task Metrics (Max vs Median duration & Shuffle Read Size), identify the skewed key, and apply salting or AQE skew join.

80. **What is Apache Iceberg / Delta Lake and why add it on top of S3 Parquet in production?**  
    Adds ACID transactions, time travel, schema evolution, and `MERGE INTO` upserts on S3 object storage without rewriting entire partitions.

81. **How does `F.expr("percentile_approx(bid_amount, 0.5)")` work in `spark/aggregations.py`?**  
    Uses a memory-bounded streaming quantile sketch (Greenwald-Khanna / t-digest style) to compute approximate medians without sorting all 500K+ bids on a single executor.

82. **Why read S3 via `s3a://` instead of `s3n://` or `s3://` in modern Spark?**  
    `s3a://` (Hadoop AWS connector) supports multi-part uploads, large files (>5GB), IAM instance profiles, and higher parallel throughput.

83. **How do you test PySpark transformations quickly in CI?**  
    Create a session-scoped local SparkSession (`master="local[*]"`, `shuffle.partitions=4`) and run on scaled synthetic datasets (`scale=0.02` in `tests/conftest.py`).

84. **What happens if a Spark job fails halfway through `mode("overwrite")` to S3?**  
    Without a transactional table format (Iceberg/Delta) or staging directory swap, partial files can leave corrupted data; S3A magic committer or staging-to-atomic-rename mitigates partial writes.

85. **How do you compute sessionization (grouping traveler clicks within 30-min inactivity gaps) in Spark?**  
    Use `LAG(timestamp)` over `Window.partitionBy("traveler_id").orderBy("timestamp")`, flag `gap > 1800s` as `1`, and take cumulative `SUM(new_session_flag)` as `session_id`.

86. **Explain `sortWithinPartitions` vs `orderBy` in Spark.**  
    `orderBy` performs a global sort across all partitions (expensive range partitioner); `sortWithinPartitions` sorts locally inside each partition for better Parquet compression.

87. **How does Spark decide between Shuffle Hash Join and Sort-Merge Join?**  
    Sort-Merge Join is default for large tables (scales gracefully by spilling to disk); Shuffle Hash Join is chosen only if one side is small enough to build a hash table per partition and `prefersSortMergeJoin=false`.

88. **Why do we separate `extract.py`, `transform.py`, and `aggregations.py`?**  
    Separation of concerns allows independent unit testing of business logic, schema evolution in `extract.py`, and reuse of cleaned silver tables across multiple downstream gold aggregations.

89. **How do you monitor PySpark job SLAs in AWS EMR?**  
    Emit CloudWatch custom metrics (row counts, DQ drop rates, job duration) at the end of `run_etl.py` and configure Spark History Server / EMR CloudWatch alarms.

90. **What is the difference between `map` and `mapPartitions` in RDD/Spark?**  
    `map` invokes a function per row; `mapPartitions` invokes it once per partition, allowing one-time setup of DB connections or loaded ML models per partition.

---

## Section 3: Machine Learning (40 Questions)

91. **Why train Linear Regression alongside Random Forest, XGBoost, and LightGBM for Audience Reach Forecasting (`models/audience_reach/train.py`)?**  
    Linear Regression establishes an interpretable baseline; if GBDTs fail to beat it on RMSE/WAPE, it signals feature insufficiency or noise. Comparing linear vs non-linear models also quantifies the gain from modeling reach saturation curves.

92. **Why does unique traveler reach saturate sub-linearly as impressions grow?**  
    Due to repeated exposures to the same active travelers (birthday-paradox / coupon-collector dynamics controlled by `frequency_cap` and `target_audience_size`). Tree models capture this via `budget_to_audience_ratio` and `frequency_cap` interactions.

93. **Explain the difference between XGBoost (depth-wise) and LightGBM (leaf-wise) tree growth.**  
    XGBoost grows trees level-by-level (balanced depth); LightGBM grows the leaf with maximum loss reduction (`best-first`), achieving faster training and lower error on large tabular data, though requiring `max_depth` / `num_leaves` regularization against overfitting.

94. **How does CatBoost prevent target leakage when encoding categorical features (`target_country`, `advertising_channel`)?**  
    CatBoost uses **ordered target statistics** computed from random permutations (using only preceding rows) and builds **oblivious (symmetric) trees**, reducing overfitting on high-cardinality categoricals.

95. **Why evaluate Audience Reach using WAPE alongside MAE, RMSE, and R² (`evaluation/metrics.py`)?**  
    RMSE penalizes large-budget campaign outliers heavily, whereas WAPE ($\sum |y - \hat{y}| / \sum |y|$) is volume-weighted, scale-invariant across markets, and directly interpretable by business stakeholders as percentage reach error.

96. **How do we compute 95% prediction confidence intervals in `models/audience_reach/train.py`?**  
    Using the validation residual standard deviation ($\hat{y} \pm 1.96 \cdot \hat{\sigma}_{\text{resid}}$), clamped at zero. In higher-skew production settings, quantile regression (`objective="reg:quantileerror"`) or split conformal prediction yields heteroscedastic intervals.

97. **What is data leakage in pre-launch campaign performance prediction (`models/campaign_performance/train.py`)?**  
    Using post-launch metrics (actual impressions, actual spend, observed CTR) to predict pre-launch revenue. We restrict features strictly to pre-launch inputs (`budget`, `campaign_duration_days`, `target_country`, `planned_avg_bid`) and prior chronological history (`shift(1).expanding()`).

98. **Why use `ColumnTransformer` inside a scikit-learn `Pipeline` instead of fitting `StandardScaler` and `OneHotEncoder` on the full dataset?**  
    Fitting transformers on the full dataset leaks test distribution mean/variance and unseen category frequencies. Encapsulating inside `Pipeline` ensures `.fit()` sees only training data.

99. **How does `OneHotEncoder(handle_unknown="ignore")` protect FastAPI serving?**  
    If a new placement or country appears in production requests that wasn't seen in training, the encoder outputs an all-zero vector instead of crashing the API with `ValueError`.

100. **When is ROC-AUC misleading for imbalanced ad click/conversion classification, and what should replace it?**  
     When CTR/CVR is 1–4%, ROC-AUC can look optimistically high because True Negatives dominate FPR. Pair it with **PR-AUC**, **Precision/Recall at operational threshold**, and **Log-Loss (Calibration)**.

101. **Explain Bias-Variance tradeoff in Random Forest vs Gradient Boosted Trees.**  
     Random Forest averages deep independent trees (reduces variance via bagging); Boosting sequentially fits shallow trees to residuals (reduces bias, can overfit if learning rate/trees are too high without early stopping).

102. **How do you calibrate predicted probabilities for ad click/conversion models?**  
     Apply Platt Scaling (logistic regression on logits) or Isotonic Regression on a held-out calibration split, ensuring $\mathbb{E}[\hat{p}] \approx \text{actual CTR}$ for accurate expected-value bidding ($\text{eCPM} = \text{pCTR} \times \text{bid}$).

103. **Why did we use Ridge (`L2`) regression for baseline campaign revenue regression while using Logistic Regression for high-performance classification?**  
     Ridge stabilizes correlated budget features (`budget`, `daily_budget`) via L2 shrinkage, while Logistic Regression outputs calibrated log-odds for binary classification.

104. **What is the difference between bagging, boosting, and stacking?**  
     Bagging trains parallel models on bootstrap samples (RF); boosting trains sequentially on pseudo-residuals (XGB/LGBM/CatBoost); stacking trains a meta-learner on out-of-fold predictions of diverse base models.

105. **How do you handle cold-start campaigns or new advertisers with zero historical performance (`features/campaign_features.py`)?**  
     Fall back hierarchically to channel-level historical priors (`channel_hist_ctr`, `channel_hist_cvr`) and target segment sizes (`target_segment_size`, `target_avg_booking_freq`) when `advertiser_hist_campaigns == 0`.

106. **Why does `root_mean_squared_error` give greater weight to large campaigns than `mean_absolute_error`?**  
     Squaring errors amplifies large residuals quadratically, pushing the model to prioritize accurate predictions on high-budget campaigns.

107. **Explain how feature importance in tree models (gain vs split vs permutation/SHAP) can mislead.**  
     Split count favors high-cardinality numeric features; gain can favor correlated proxies. Permutation importance and TreeSHAP on the validation set give faithful attribution.

108. **What happens to tree-based extrapolation when predicting a campaign with \$500K budget if max training budget was \$220K?**  
     Decision trees predict flat step constants beyond the training range (cannot extrapolate linearly). Log-transforming budget or blending linear/ridge trends with GBDTs mitigates boundary clipping.

109. **How do you tune hyperparameters systematically without overfitting the test set?**  
     Use time-series expanding-window cross-validation on the training partition (Optuna/Bayesian search), reserving the final chronological test set strictly for one-time unbiased evaluation.

110. **Why is `MAPE` dangerous when actual impressions or conversions can be zero or near zero?**  
     Division by zero explodes `MAPE` toward infinity, and `MAPE` asymmetrically penalizes over-forecasts more than under-forecasts; `WAPE` avoids zero-denominator blowups.

111. **Explain the difference between L1 (Lasso) and L2 (Ridge) regularization.**  
     L1 adds $\lambda \sum |\beta_i|$ producing sparse feature selection; L2 adds $\lambda \sum \beta_i^2$ shrinking correlated weights smoothly without zeroing them out.

112. **How do you handle multi-target prediction (`impressions`, `clicks`, `ctr`, `conversions`, `conversion_rate`, `revenue`) consistently?**  
     In `models/campaign_performance/train.py`, we train dedicated target regressors sharing a unified pre-launch feature pipeline; in tighter consistency constraints, one can predict `impressions`, `ctr`, `cvr` and derive `clicks = imp * ctr`, `conversions = imp * cvr`.

113. **What is target encoding and how can it cause catastrophic overfitting?**  
     Replacing a category (`city`) with its mean target value leaks label noise if computed in-sample. Always use out-of-fold or ordered smoothing (`(sum + m * global_mean) / (count + m)`).

114. **How does `subsample=0.85` and `colsample_bytree=0.85` improve XGBoost/LightGBM generalization?**  
     Row and column subsampling decorrelates trees, acting like stochastic regularization and preventing single dominant features from monopolizing every split.

115. **Explain the difference between micro-F1, macro-F1, and binary F1.**  
     Binary F1 evaluates positive-class precision/recall harmonic mean; macro-F1 averages unweighted class F1s; micro-F1 aggregates global TP/FP/FN.

116. **How do you diagnose whether a model suffers from high bias vs high variance?**  
     Compare training error vs validation error: both high = high bias (underfitting -> add features, increase depth); low train error + high validation error = high variance (overfitting -> regularize, add data, reduce depth).

117. **Why clamp predicted `ctr` and `conversion_rate` to `[0.0, 1.0]`?**  
     Unconstrained linear/GBDT regressors can overshoot `[0, 1]` near boundaries; clamping guarantees physically valid rates before downstream revenue calculations.

118. **What is collinearity and why does it affect Linear Regression but barely affect XGBoost?**  
     Collinear features (`budget` and `daily_budget` when duration is fixed) create ill-conditioned $(X^T X)^{-1}$ matrices causing unstable OLS coefficients; trees split on one feature at a time without matrix inversion.

119. **How do you decide the classification threshold in `is_high_performing` beyond default `0.5`?**  
     Sweep thresholds on the validation Precision-Recall curve to maximize expected campaign ROI or satisfy stakeholder precision/recall SLAs.

120. **Explain why `n_jobs=-1` speeds up Random Forest and XGBoost.**  
     Parallelizes independent tree construction (RF) or histogram bin scanning across splits (XGB/LGBM) over all CPU cores.

121. **How would you predict impressions & conversions jointly using a multi-stage funnel model?**  
     Model $\hat{I} = f_1(X)$, $\widehat{\text{CTR}} = f_2(X)$, $\widehat{\text{CVR}}_{\text{click}} = f_3(X)$, then derive clicks and conversions multiplicatively with uncertainty propagation.

122. **What is early stopping in GBDTs and why must it use a temporal validation slice?**  
     Stops boosting when validation loss stops improving for `k` rounds; using a random slice leaks future temporal patterns into the stopping iteration.

123. **How do you represent cyclic calendar features (`day_of_week`, `day_of_year`) for linear and neural models?**  
     Encode with sine/cosine pairs (`sin_doy`, `cos_doy` in `features/inventory_features.py`) so Dec 31 and Jan 1 remain adjacent in Euclidean space.

124. **Why is median imputation (`SimpleImputer(strategy="median")`) preferred over mean imputation for ad budgets?**  
     Travel advertising budgets are right-skewed (log-normal); median is robust to extreme mega-campaigns.

125. **How does `log1p` target transformation help skewed revenue/reach models?**  
     Compresses extreme right-tail budgets/reaches, stabilizes variance (homoscedasticity), and aligns squared-error loss closer to relative percentage error (remembering Jensen's inequality bias correction when inverting `expm1`).

126. **What is conformal prediction and how can it upgrade our confidence intervals?**  
     Uses calibration residuals to provide distribution-free finite-sample coverage guarantees ($P(Y \in [\hat{l}, \hat{u}]) \ge 1 - \alpha$).

127. **How do you prevent feature drift from breaking `StandardScaler` in production?**  
     Monitor input feature PSI/KS (`monitoring/data_drift.py`) and retrain pipelines on sliding windows when drift exceeds critical thresholds (`PSI >= 0.25`).

128. **Explain the difference between `roc_auc_score` and `log_loss`.**  
     ROC-AUC measures rank ordering across all thresholds; Log-Loss measures probability calibration accuracy (crucial for expected revenue estimation).

129. **Why use `scikit-learn` `ColumnTransformer` even when training XGBoost or LightGBM?**  
     Standardizes preprocessing across all 4–5 compared algorithms (`LinearRegression`, `RF`, `XGB`, `LGBM`, `CatBoost`) so every model receives the exact same feature space in `ModelEvaluator`.

130. **How would you extend Audience Reach Forecasting to incorporate cross-channel overlap?**  
     Include advertiser-level concurrent active campaign counts and channel-mix overlap features, or model reach via generalized Bruce/beta-binomial reach-and-frequency curves.

---

## Section 4: Time-Series Forecasting (30 Questions)

131. **Why compare ARIMA, SARIMA, Prophet, XGBoost, and LightGBM on an identical 30-day temporal holdout (`models/inventory_forecasting/train.py`)?**  
     Classical univariate models (ARIMA/SARIMA/Prophet) excel on smooth single-series seasonality, while global GBDTs (XGBoost/LightGBM) share strength across all 72 `(country, placement)` series and incorporate exogenous `advertiser_demand` and `booking_trends`. Benchmarking on the same holdout lets data decide the winner objectively.

132. **Explain the `(p, d, q)` parameters in ARIMA(2, 1, 2).**  
     `p=2`: 2 autoregressive lag terms; `d=1`: first-order differencing to remove non-stationary trend; `q=2`: 2 moving-average error lag terms.

133. **Why did we add seasonal order `(1, 0, 1, 7)` in SARIMA for ad inventory?**  
     Travel search and ad inventory exhibit strong weekly periodicity (`s=7`: weekend vs weekday travel shopping surges).

134. **How does Prophet decompose an inventory time series?**  
     $y(t) = g(t) + s(t) + h(t) + \epsilon_t$, combining piecewise linear/logistic growth trend $g(t)$, Fourier series yearly/weekly seasonality $s(t)$, holiday effects $h(t)$, and Gaussian noise.

135. **How do we generate 7, 14, and 30-day multi-step inventory forecasts with XGBoost/LightGBM (`_generate_recursive_forecasts`)?**  
     Recursively: predict day $t+1$, append $\hat{y}_{t+1}$ into the lag buffer to compute `avail_lag_1`, `avail_lag_7`, `avail_roll_mean_7` for $t+2$, up to $t+30$.

136. **Compare recursive forecasting vs direct multi-horizon forecasting (`target_7d_ahead`, `target_14d_ahead`, `target_30d_ahead`).**  
     Recursive uses one model and feeds predictions forward (can accumulate error over 30 steps); direct trains separate models per horizon (no error accumulation, more models to maintain). Our feature pipeline supports both (`features/inventory_features.py`).

137. **Why must lag/rolling features in `features/inventory_features.py` be shifted by at least 1 day (`shift(1).rolling(7)`)?**  
     Without `shift(1)`, the rolling window at date $t$ includes $y_t$ itself—leaking the target into its own feature!

138. **How do you test a time series for stationarity?**  
     Augmented Dickey-Fuller (ADF) test (null: unit root / non-stationary) and KPSS test (null: stationary), complemented by ACF/PACF plots.

139. **Why is random k-fold cross-validation invalid for inventory forecasting?**  
     Random splits allow future observations ($t+1$) to train models predicting $t$, exploiting autocorrelation and giving wildly optimistic validation metrics that collapse in production.

140. **Explain Time-Series Expanding Window vs Sliding Window Cross-Validation.**  
     Expanding grows the training history forward across multiple cutoffs; sliding keeps a fixed lookback window (preferable when older pre-pandemic/legacy travel regimes become stale).

141. **How do you handle cold-start `(country, placement)` pairs with short history in global GBDTs vs ARIMA?**  
     ARIMA/SARIMA fail or overfit with <30 points; global LightGBM/XGBoost transfers seasonal and day-of-week patterns learned from all other mature markets.

142. **What causes residual autocorrelation in inventory forecasts, and how do you fix it?**  
     Missing lags, unmodeled holidays, or exogenous demand shocks; diagnose with Ljung-Box test on residuals and add lagged residuals, holiday indicators, or `advertiser_demand`.

143. **How do you detect and handle travel seasonality shifts (e.g., Easter moving dates or COVID-level shocks)?**  
     Add dynamic holiday calendars (`holidays` in Prophet, `is_holiday_window` flags in GBDT) and sample weighting / change-point detection.

144. **Compare MAPE vs WAPE vs sMAPE on multi-country ad inventory (`US` vs `IT`).**  
     `US` has ~4× larger inventory than `IT`. MAPE weights each country-day equally (error in small markets dominates), while WAPE weights total impression error across markets proportionally to volume.

145. **Why include both `booked_roll_mean_7` and `booking_trends` (`booked_7d / booked_28d`)?**  
     Captures both level and short-term momentum (acceleration in advertiser demand tightening remaining inventory).

146. **What is hierarchical time-series reconciliation (`country -> placement -> total`)?**  
     Forecasts generated independently per placement may not sum to the country total; bottom-up, top-down, or MinT (Minimum Trace) reconciliation ensures coherent totals across hierarchy levels.

147. **How do prediction intervals widen as the inventory forecast horizon grows from 7 to 30 days?**  
     Multi-step forecast error variance grows with horizon $h$ (approximately $\propto \sqrt{h}$ under random-walk components); bootstrap recursive rollouts or quantile GBDTs capture this widening uncertainty.

148. **Why use `avail_roll_std_14` as a feature?**  
     Measures recent market volatility—higher volatility signals transitional travel periods (e.g., shoulder-to-peak season shifts).

149. **What happens if `advertiser_demand` is unavailable 30 days into the future?**  
     Future exogenous features that aren't known in advance must themselves be forecasted or held at last observed/planned booking schedule (`last_demand` in `_generate_recursive_forecasts`) to prevent training-serving skew.

150. **Explain Fourier terms (`sin_doy`, `cos_doy`) vs one-hot `month` encoding.**  
     One-hot `month` creates sharp step jumps on month boundaries (Jan 31 -> Feb 1); Fourier terms provide smooth continuous annual seasonality with fewer parameters.

151. **How does Prophet handle multiplicative vs additive seasonality?**  
     `seasonality_mode="multiplicative"` scales seasonal swings proportionally to overall trend growth (critical when platform traffic grows 2× YoY).

152. **Why did we pass `y_tr[-365:]` to SARIMAX in `models/inventory_forecasting/train.py`?**  
     State-space SARIMAX optimization scales poorly on very long daily series; capping to the most recent 365 days balances seasonal estimation accuracy with fast training latency.

153. **What is the Diebold-Mariano test?**  
     A formal statistical hypothesis test comparing whether forecast error differences between two models (e.g., LightGBM vs SARIMA) are statistically significant.

154. **How do you handle missing dates in ad inventory logs before computing `shift(7)`?**  
     Reindex every `(country, placement)` group onto a complete `pd.date_range(min_date, max_date, freq="D")` calendar so `shift(7)` always equals exactly 7 calendar days, not 7 rows!

155. **Why log both `benchmark` metrics (12 series) and `full_test_metrics` (all 72 series) in `models/inventory_forecasting/train.py`?**  
     Ensures strict apples-to-apples comparison against ARIMA/SARIMA/Prophet on the identical holdout slice while also reporting global generalization across all 72 combinations.

156. **What is the role of differencing (`d=1`) in ARIMA vs lag features in XGBoost?**  
     ARIMA requires differencing to achieve stationarity; trees cannot extrapolate linear trends without differencing or trend features, so predicting $\Delta y_t = y_t - y_{t-1}$ or including trend/lag ratios helps when strong drift exists.

157. **How do you monitor forecast accuracy in production when ground truth arrives 7–30 days later?**  
     Store issued forecasts with `(forecast_date, target_date, horizon_days)` in PostgreSQL, then run a daily evaluation join once `target_date` materializes to track rolling 7d/14d/30d WAPE and trigger alerts (`monitoring/model_drift.py`).

158. **Explain MASE (Mean Absolute Scaled Error).**  
     Divides model MAE by in-sample naive seasonal forecast MAE; MASE < 1 proves the model beats a simple seasonal naive baseline.

159. **When would you choose Prophet over LightGBM in production?**  
     When interpretability (decomposed trend + holiday + weekly components), sparse/irregular series, or strong non-stationary growth extrapolation is needed without large cross-sectional panel data.

160. **How do you prevent holiday spikes (Black Friday / Christmas travel) from distorting rolling averages in January?**  
     Use median/winsorized rolling features or include explicit post-holiday cooldown indicators.

---

## Section 5: Causal Inference & Experimentation (25 Questions)

161. **Why isn't observational correlation between ML-targeting adoption and higher CVR enough to prove ML targeting works?**  
     Selection bias / confounding: advertisers with larger budgets and travelers with higher intent (`travel_intent`, `booking_frequency`) disproportionately enter ML-targeted campaigns, inflating naive observational lift (`naive_biased_lift` vs `psm_att` in `causal_inference/propensity_matching.py`).

162. **Why did we randomize at the `traveler_id` cluster level instead of individual impression level (`causal_inference/ab_testing.py`)?**  
     Randomizing per impression exposes the *same* traveler to both ML and traditional targeting (intra-user contamination / SUTVA violation) and underestimates variance due to positive intra-user correlation across repeated visits.

163. **Explain Cluster-Robust Standard Errors (`cov_type="cluster"`) and why ignoring clustering inflates Type I error.**  
     Repeated impressions from the same traveler are positively correlated; naive binomial SEs treat $N$ impressions as $N$ independent Bernoulli trials, making confidence intervals artificially narrow and producing false positives.

164. **What is a Sample Ratio Mismatch (SRM) check and why run it in `run_ab_test`?**  
     Chi-square goodness-of-fit test checking whether observed unique user counts match the intended 50/50 split (`srm_p_value >= 0.001`). SRM failure indicates logging bugs, bot traffic imbalance, or broken randomization, invalidating the experiment.

165. **State the 3 core assumptions of causal inference under Potential Outcomes (Rubin Causal Model).**  
     1) **SUTVA** (no interference between units, single treatment version), 2) **Ignorability / Unconfoundedness** ($Y(0), Y(1) \perp T \mid X$), and 3) **Positivity / Common Support** ($0 < P(T=1|X) < 1$).

166. **How does Difference-in-Differences (`causal_inference/difference_in_differences.py`) isolate causal impact without randomization?**  
     Subtracts the pre-to-post change of comparison markets (capturing shared secular trends/seasonality) from the pre-to-post change of ML-rollout markets: $\hat{\tau}_{\text{DiD}} = (\bar{Y}_{\text{treat,post}} - \bar{Y}_{\text{treat,pre}}) - (\bar{Y}_{\text{ctrl,post}} - \bar{Y}_{\text{ctrl,pre}})$.

167. **What is the Parallel Trends assumption in DiD and how do you test it?**  
     Assumes treated and control markets would have evolved identically absent treatment. Validate via pre-treatment event-study plots (leads/placebo tests showing $\beta_{\text{pre}} \approx 0$ before rollout).

168. **Explain Propensity Score Matching (`causal_inference/propensity_matching.py`) and the role of the 0.2 SD caliper.**  
     Matches each treated unit to a control unit with similar estimated $e(X) = P(T=1|X)$ on the logit scale; a caliper of $0.2 \cdot \sigma(\text{logit}(e))$ rejects bad matches outside common support, reducing residual bias.

169. **How do we verify that Propensity Score Matching actually balanced confounders?**  
     Compute Standardized Mean Difference (SMD) for every covariate before and after matching (`smd_before_matching` vs `smd_after_matching`); post-matching SMD $< 0.1$ confirms covariate balance.

170. **What is the difference between ATE, ATT, and CATE?**  
     **ATE**: average effect across the entire population; **ATT**: average effect among units that actually received treatment (PSM & DiD target ATT); **CATE** ($\tau(x)$): conditional effect for subgroup/individual $x$ (Uplift Modeling).

171. **How does EconML's `CausalForestDML` (`causal_inference/uplift_modeling.py`) estimate CATE?**  
     Double/Debiased Machine Learning orthogonalizes both outcome ($Y - \hat{m}(X)$) and treatment ($T - \hat{e}(X)$) via cross-fitted nuisance models (`RandomForestRegressor`/`Classifier`), then fits a causal forest on residuals to estimate heterogeneous treatment effects $\tau(X)$.

172. **Define the four Uplift Personas: Persuadables, Sure Things, Lost Causes, and Sleeping Dogs.**  
     - **Persuadables** ($\tau > 0$): convert *because* of ML targeting -> target aggressively.
     - **Sure Things** ($\tau \approx 0$, high baseline): convert regardless -> save ad spend.
     - **Lost Causes** ($\tau \approx 0$, low baseline): won't convert -> suppress ads.
     - **Sleeping Dogs** ($\tau < 0$): ML targeting annoys/hurts conversion (e.g., saturated Platinum travelers) -> exclude!

173. **What is DoWhy's 4-step causal workflow used in `propensity_matching.py`?**  
     1) **Model** causal graph (`CausalModel`), 2) **Identify** estimand via backdoor criterion, 3) **Estimate** effect (`propensity_score_stratification`), 4) **Refute** estimate (placebo treatment, random common cause).

174. **Why can marketplace ad experiments suffer from network spillover (budget cannibalization) violating SUTVA?**  
     Control and treatment campaigns bid in the *same* ad auction! If ML-targeted treatment campaigns win more auctions, they raise clearing prices or starve control campaigns of impressions, inflating apparent lift.

175. **How would you mitigate auction interference in travel advertising experiments?**  
     Use **budget-split / shadow bidding** experiments, **geo-clustering / switchback** designs, or **cluster randomization at advertiser + market level**.

176. **What is Novelty Effect vs Primacy Effect in A/B testing?**  
     Early spikes when users react temporarily to a new ad format (novelty) that decay over weeks; mitigate by running tests across full travel booking cycles (2–4+ weeks) and checking time-by-treatment interaction.

177. **How do you compute statistical power and minimum sample size before launching an A/B test?**  
     $n \approx 2(z_{1-\alpha/2} + z_{1-\beta})^2 \frac{p(1-p)}{\Delta^2}$ adjusted by the cluster design effect $\text{DEFF} = 1 + (m - 1)\rho$ (where $\rho$ is intra-cluster correlation and $m$ is avg impressions per traveler).

178. **Why is peeking at p-values daily without correction invalid?**  
     Repeated significance testing inflates family-wise Type I error far above 5%. Use sequential testing (Always-Valid Confidence Sequences / alpha-spending) or fixed-horizon evaluation.

179. **What is CUPED (Controlled-experiment Using Pre-Experiment Data) and how does it help?**  
     Subtracts $\theta(X_{\text{pre}} - \bar{X}_{\text{pre}})$ from outcome $Y$ using pre-experiment traveler booking history, reducing variance and shortening required experiment duration by 20–50%.

180. **What are unmeasured confounders and how do sensitivity analyses (e.g., Rosenbaum bounds / E-values) address them?**  
     Observational methods (PSM/DiD) cannot adjust for unobserved variables (like off-platform promotions); sensitivity analysis quantifies how strong an unmeasured confounder would need to be to nullify the observed ATT.

181. **Why evaluate uplift models using Qini curves / uplift deciles instead of RMSE?**  
     Individual ground-truth counterfactuals $Y_i(1) - Y_i(0)$ are never observed for the same user (Fundamental Problem of Causal Inference), so we rank users by $\hat{\tau}(x)$ into deciles and verify observed $(Y|T=1) - (Y|T=0)$ increases monotonically across deciles.

182. **How do we translate uplift scores into budget allocation (`api/routes/forecasting.py` `/targeting/recommendations`)?**  
     Filter to positive-uplift segments ($\hat{\tau} > 0$) and allocate campaign budget proportionally to predicted incremental CATE rather than raw baseline conversion rate.

183. **Difference between Two-Way Fixed Effects DiD and staggered DiD?**  
     When different markets adopt ML targeting at different dates, standard TWFE can produce biased negative weights; use Callaway-Sant'Anna or Sun-Abraham group-time ATT estimators.

184. **Why report both absolute lift and relative lift (`causal_impact_report.md`)?**  
     A +50% relative lift on a 0.2% CVR is only +0.1% absolute lift (possibly unprofitable), whereas absolute lift directly determines incremental conversions per 1,000 impressions.

185. **How do you handle long-tail conversion revenue in A/B tests (extreme luxury bookings)?**  
     Winsorize revenue at the 99th percentile, test conversion rate + conditional AOV separately, or use bootstrap/non-parametric Mann-Whitney checks alongside CR1 SEs.

---

## Section 6: MLOps & Production ML Systems (30 Questions)

186. **How does `ModelEvaluator` (`evaluation/model_comparison.py`) standardize experiment tracking across all three forecasting models?**  
     Every candidate algorithm (`LinearRegression`, `RF`, `XGB`, `LGBM`, `CatBoost`, `ARIMA`, `SARIMA`, `Prophet`) is logged inside a dedicated MLflow run with parameters, standardized metrics, serialized pickle artifacts, and ranked automatically to promote the winner to `@champion`.

187. **Why use MLflow Model Registry aliases (`@champion`) instead of deprecated Stage transitions (`Staging`/`Production`)?**  
     MLflow 2.9+/3.x aliases (`@champion`, `@challenger`) provide mutable named pointers to specific immutable model versions, supporting clean blue/green and shadow deployments.

188. **How do we prevent training-serving skew in this platform?**  
     By bundling the entire preprocessing `ColumnTransformer` (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) together with the estimator inside a single scikit-learn `Pipeline` artifact (`best_model.pkl`), ensuring FastAPI runs identical transformations at inference time.

189. **Explain Population Stability Index (PSI) thresholds (`monitoring/data_drift.py`).**  
     - `PSI < 0.1`: No significant distribution shift.  
     - `0.1 <= PSI < 0.25`: Moderate drift -> **Warning alert** (`configs/config.yaml`).  
     - `PSI >= 0.25`: Severe distribution shift -> **Critical alert** + automated retraining trigger.

190. **How does Evidently AI complement custom PSI/KS checks in `monitoring/data_drift.py`?**  
     Evidently generates interactive HTML visual drift reports (`DataDriftPreset`) for stakeholder inspection while fast NumPy PSI + Scipy two-sample KS tests emit real-time Prometheus gauges (`expedia_ad_feature_psi`).

191. **What is the difference between Data Drift, Concept Drift, and Prediction Drift?**  
     - **Data Drift**: $P(X)$ shifts (e.g., sudden surge in mobile traffic).  
     - **Concept Drift**: $P(Y \mid X)$ changes (e.g., new ad UX changes user click propensity for the same intent score).  
     - **Prediction Drift**: $P(\hat{Y})$ shifts (monitored in `monitoring/model_drift.py`).

192. **Why cache FastAPI model responses in Redis (`api/routes/forecasting.py`) using SHA-256 request hashes?**  
     Repeated dashboard queries or batch campaign simulations for identical parameters hit Redis in <2ms (`PREDICTION_CACHE_HITS`), offloading CPU-bound tree inference and protecting API tail latency.

193. **What happens if Redis or PostgreSQL goes down in development or during a partial outage?**  
     `CacheClient` and `DatabaseClient` (`utils/storage.py`) fall back gracefully (in-memory cache / SQLite or readable error handling) so local testing and core health endpoints remain resilient; in production, monitor `cache_backend` via `/api/v1/health`.

194. **How do we measure API tail latency and throughput (`api/main.py` & `monitoring/prometheus/metrics.py`)?**  
     A FastAPI HTTP middleware records `X-Process-Time-Ms` on every response, and route handlers observe `PREDICTION_LATENCY` Prometheus histograms (`buckets=5ms..2.5s`) so Grafana computes `p95` and `p99` quantiles.

195. **Why include model versioning (`model_version` field in API responses)?**  
     Links every served prediction back to the exact algorithm + MLflow `run_id`, enabling rapid rollback and post-incident auditing when downstream metrics shift.

196. **How would you implement shadow deployment before promoting a new XGBoost reach model?**  
     Route 100% of live `/forecast/audience-reach` traffic asynchronously to `@challenger` in background threads, log predictions alongside `@champion` without returning them to users, and compare latency + actual accuracy once campaign outcomes finalize.

197. **Explain Canary Rollout for the FastAPI serving layer.**  
     Route 5% -> 25% -> 100% of traffic via ECS/ALB weighted target groups while watching `expedia_ad_prediction_errors_total` and `p95` latency alarms in Prometheus/CloudWatch.

198. **How do you trigger automated retraining without thrashing on transient noise?**  
     Require sustained drift or degradation (e.g., 3 consecutive daily checks with `RMSE > 1.25 * baseline_rmse` or `PSI >= 0.25`), run the retraining DAG (`pipelines/batch_pipeline.py`), and promote `@champion` only if the new model beats the incumbent on the latest holdout.

199. **Why use `sqlite:///mlflow.db` locally and PostgreSQL/RDS + S3 artifact root in production?**  
     SQLite supports single-node local Model Registry without external services; multi-worker SageMaker/ECS training requires RDS PostgreSQL backend store + S3 artifact store for concurrent ACID writes.

200. **What are Feature Stores (e.g., Feast / SageMaker Feature Store) and how does `features/` mirror one?**  
     Our `features/*.py` modules materialize versioned Parquet feature tables for offline training and share lookup/derivation logic with online serving; a Feature Store unifies offline S3 Parquet + online Redis/DynamoDB low-latency lookups.

201. **How do you ensure deterministic builds across local macOS and Docker Linux (`Dockerfile`)?**  
     Pin dependency bounds in `requirements.txt`, install system libs (`openjdk-17`, `libgomp1`), and run `pytest` + `ruff` + `black` in CI before building container images.

202. **Explain how `emit_cloudwatch_alert` (`monitoring/model_drift.py`) integrates with AWS CloudWatch.**  
     When AWS credentials are present, it pushes structured JSON alert events via `boto3.client("logs").put_log_events` to `/expedia/ad-intelligence`, enabling CloudWatch Metric Filters -> SNS PagerDuty/Slack alerts.

203. **Why separate unit tests (`test_regression_and_classification_metrics`) from integration tests (`test_pyspark_batch_etl_and_aggregations`)?**  
     Fast unit tests catch math/schema regressions in milliseconds; heavier PySpark/ML integration tests verify end-to-end data flow using scaled fixtures (`scale=0.02`).

204. **What is model governance & lineage?**  
     Tracing any production prediction back to `(git_commit, dataset_checksum_sha256, feature_version, mlflow_run_id, hyperparameters, validation_metrics)`—captured via `_ingestion_manifest.json`, `_etl_watermark.json`, and MLflow tags.

205. **How do you handle schema evolution in FastAPI (`api/schemas/models.py`) without breaking existing clients?**  
     Keep `/api/v1/` backward-compatible by adding optional fields with sensible defaults, or introduce `/api/v2/` for breaking contract changes.

206. **Why monitor `max_null_rate` and `schema_valid` BEFORE running model drift checks?**  
     An upstream ETL bug (all-NULL `budget` column) manifests immediately as a data quality failure; catching it early prevents silent `SimpleImputer` masking and garbage predictions.

207. **Explain why we enforce `ruff` and `black` in `Makefile` & `pyproject.toml`.**  
     Automated formatting and linting eliminate style bikeshedding, catch unused imports/mutable default bugs (`B008`), and keep PR diffs minimal and senior-reviewer ready.

208. **How do you manage secrets (DB passwords, AWS keys) safely?**  
     Never commit `.env` or credentials (`.gitignore` excludes `.env`); provide `.env.example` locally and use AWS Secrets Manager / IAM Roles for Service Accounts (IRSA) in production.

209. **What is the purpose of `/api/v1/health` reporting `"degraded"` if any `best_model.pkl` is missing?**  
     Load balancers / Kubernetes readiness probes stop routing inference traffic to pods lacking serialized models while still keeping liveness diagnostics inspectable.

210. **How do you benchmark FastAPI throughput and batch prediction sizing?**  
     Use Locust/k6 against `/api/v1/predict/campaign-performance` varying batch sizes (`1, 50, 200, 500`), tuning Uvicorn workers and `max_length=500` in Pydantic schemas.

211. **Why use `cache_ttl_seconds = 300` in Redis rather than infinite TTL?**  
     5-minute TTL balances high cache-hit efficiency for dashboard polling with bounded staleness after daily model retraining or inventory refreshes.

212. **How do you handle cold-start latency when FastAPI boots?**  
     Warm up `@lru_cache` model bundles during FastAPI lifespan startup so the very first user request does not pay pickle/XGBoost deserialization overhead.

213. **What is SageMaker Processing vs SageMaker Training vs SageMaker Endpoints?**  
     Processing runs feature prep/drift checks (`run_monitoring.py`); Training runs distributed model fitting with Spot instances & MLflow logging; Endpoints host autoscaling REST inference containers.

214. **How do you detect silent data pipeline failures when row count drops from 1M to 10K?**  
     Add volume anomaly assertions in `DataQualityReport` (`clean_count >= 0.8 * rolling_7d_avg_count`) before overwriting downstream Parquet/PostgreSQL tables.

215. **Why log both `evaluation_summary.json` and MLflow artifacts?**  
     MLflow provides centralized historical comparison UI & registry governance, while lightweight local `evaluation_summary.json` enables zero-dependency inspection in FastAPI `/api/v1/models` and Streamlit.

---

## Section 7: System Design (25 Questions)

216. **Design a real-time travel ad bidding & forecasting system handling 100K+ QPS.**  
     Edge Load Balancers -> Stateless Go/FastAPI Bidder Services -> In-memory/Redis Feature Lookup (populated by Kafka + Flink/Spark Structured Streaming) -> Pre-loaded LightGBM/Logistic pCTR+pCVR scorers (<10ms p99) -> Async Kafka impression/click logging -> S3 Parquet + EMR batch retraining.

217. **How does our Lambda Architecture (Batch + Streaming in `pipelines/` & `spark/streaming.py`) combine accuracy and freshness?**  
     Batch PySpark (`run_etl.py`) processes full 1M+ historical impressions accurately for daily model training and PostgreSQL KPIs, while Spark Structured Streaming computes 5-minute real-time CTR & anomaly signals with 10-minute watermarks.

218. **How would you scale `scripts/generate_synthetic_data.py` & `spark/run_etl.py` from 1M to 10B events/day?**  
     Replace single-node Pandas feature steps (`features/*.py`) with pure distributed PySpark SQL/DataFrames writing partitioned Iceberg tables on S3 via AWS EMR Auto-Scaling clusters, and train GBDTs via distributed XGBoost-on-Spark or Ray/SageMaker.

219. **Design the Redis caching strategy for Expedia advertiser dashboards.**  
     Cache expensive forecast queries keyed by `sha256(request_payload)` with 5-minute TTL, invalidate or version-prefix keys (`v{model_version}:...`) upon `@champion` promotion, and avoid caching high-cardinality arbitrary requests without LRU eviction (`maxmemory-policy allkeys-lru`).

220. **How do you prevent thundering-herd cache stampedes when a popular forecast key expires?**  
     Use probabilistic early expiration (jittered TTL) or singleflight distributed locking in Redis (`SETNX`) so only one worker recomputes while others wait briefly or serve stale-while-revalidate.

221. **How do you design an inventory pacing & frequency-capping system for advertisers?**  
     Maintain per-`(traveler_id, campaign_id)` sliding-window counters in Redis/Aerospike with TTL = cap window, paired with a PID pacing controller adjusting bid multipliers based on `actual_spend_rate` vs `forecasted_inventory`.

222. **Design multi-region active-active serving for US, EU, and APAC advertisers.**  
     Replicate S3 Parquet + MLflow model artifacts across `us-west-2`, `eu-west-1`, and `ap-northeast-1`; deploy regional FastAPI/SageMaker endpoints behind Route 53 latency-based routing with regional ElastiCache clusters.

223. **How do you decouple model training (`make train`) from API serving (`make api`)?**  
     Training writes immutable versioned artifacts to MLflow/S3 and updates the `@champion` alias; serving pods poll or restart cleanly via webhook/Lambda without sharing training compute or memory.

224. **How would you design a real-time feature store for `targeting_signals`?**  
     Kafka ingests clickstream/search events -> Spark Structured Streaming aggregates 5-min/1-hr/24-hr user intent features -> dual-writes to Redis (online serving <5ms) and S3 Iceberg (offline point-in-time joins for training).

225. **Why is point-in-time (as-of) join critical when joining `ad_impressions` with `targeting_signals` at scale?**  
     Joining with the *latest* signal table across all history leaks future searches into past impressions; an `ASOF` join matches each impression timestamp `t` strictly to the latest signal with `signal_timestamp <= t`.

226. **Design an automated A/B experimentation platform for Expedia travel ads.**  
     Deterministic hashing (`hash(traveler_id, experiment_salt) % 10000`) at ad-request time -> log exposure events to Kafka -> daily PySpark metric aggregation with CUPED + SRM checks + cluster-robust CR1 inference (`ab_testing.py`).

227. **How do you handle backpressure when Kafka lag spikes during Black Friday / Cyber Monday travel sales?**  
     Configure `maxOffsetsPerTrigger` in Spark Structured Streaming, autoscale Kafka partitions & EMR executors, and shed non-critical monitoring workloads while preserving core impression logging.

228. **Compare PostgreSQL/RDS vs ClickHouse/Snowflake for storing ad analytics aggregations (`agg_*`).**  
     PostgreSQL excels for transactional API lookups (`<10M` aggregated rows, point queries, constraints); ClickHouse/Redshift/Snowflake excels for ad-hoc OLAP slicing over billions of raw impression events.

229. **How do you design a budget optimization engine using Model 1 (Reach), Model 3 (Performance), and Uplift?**  
     Treat each `(country, placement, segment)` curve as a concave diminishing-returns function predicted by Reach/Performance/CATE models, then solve a constrained knapsack/convex optimization problem maximizing incremental conversions subject to `sum(spend) <= budget`.

230. **What SLAs (latency, availability, freshness) would you set for each endpoint in `api/routes/forecasting.py`?**  
     - `/health`: <10ms, 99.99%  
     - `/predict/campaign-performance` & `/forecast/audience-reach`: <100ms p95 (cached <5ms), 99.9%  
     - `/forecast/inventory`: <150ms p95 (30-step recursive rollout), 99.9%  
     - Batch data freshness: Daily <6:00 AM UTC; Streaming signal freshness: <5 minutes.

231. **How do you ensure GDPR/CCPA compliance in `travelers` and `ad_impressions` pipelines?**  
     Hash/tokenize PII, separate PII mapping tables from analytics Parquet, support right-to-be-forgotten (`DELETE` / Iceberg row-level deletes by `traveler_id`), and use coarse `age_group` / `city` buckets.

232. **Design a fallback hierarchy when the FastAPI ML inference container experiences high CPU or model load failure.**  
     1) Serve cached Redis forecast, 2) Fall back to precomputed PostgreSQL `pred_*` batch table, 3) Fall back to deterministic channel-level lookup heuristic so advertisers never see HTTP 500 during campaign setup.

233. **How do you prevent one noisy advertiser sending 500-item batch payloads from starving other API callers?**  
     Enforce Pydantic `max_length=500`, API Gateway rate limiting (token bucket per `advertiser_id`), request timeouts, and separate async worker pools for bulk batch jobs vs interactive UI calls.

234. **How would you architect multi-armed contextual bandits on top of our Uplift model?**  
     Use Thompson Sampling / LinUCB initialized with `CausalForestDML` priors to dynamically balance exploration vs exploitation across travel placements while logging propensity scores for unbiased offline policy evaluation (IPS/DR).

235. **Explain the cost-optimization tradeoffs of Spot instances on AWS EMR & SageMaker.**  
     Run stateless Spark workers & SageMaker Training on EC2 Spot (60–70% cost savings) with checkpointing to S3, while keeping the Spark Driver, RDS, ElastiCache, and FastAPI API pods on On-Demand/Reserved instances.

236. **How do you design dead-letter queues (DLQs) for malformed Kafka events?**  
     Wrap JSON schema parsing in Structured Streaming (`from_json`); route rows where `data IS NULL` to a `ad_impressions_dlq` Kafka topic / S3 quarantine path with error metadata for replay after schema fixes.

237. **How do you version API contracts, feature schemas, and model artifacts together?**  
     Tag releases semantically (`v1.0.0`), embed `model_version` in API responses, enforce contract tests (`tests/test_api_and_monitoring.py`), and store schema snapshots in AWS Glue.

238. **Design an anomaly detection alerting pipeline for sudden advertiser spend spikes or impression drops.**  
     Stream 5-min aggregations from `spark/streaming.py`, compare against rolling seasonal hourly baselines (STL residuals / z-score), and trigger AWS Lambda -> SNS/PagerDuty alerts on >3σ deviations.

239. **How would you support "what-if" budget simulation sliders in the Streamlit dashboard via FastAPI?**  
     Send a batch array (`items` list of 10 budget points from \$5K to \$100K) to `/api/v1/predict/campaign-performance` in one HTTP call to render smooth diminishing-returns ROAS curves in <50ms.

240. **Why choose Parquet + PostgreSQL + Redis together instead of a single database?**  
     Polyglot persistence matches each workload: S3 Parquet for cheap petabyte analytical scans/ML training, PostgreSQL for relational serving & joins, and Redis for sub-millisecond inference caching.

---

## Section 8: Production ML Troubleshooting Scenarios (20 Questions)

241. **Scenario: Campaign Performance Model 3 achieves R²=0.96 in offline validation, but R²=0.32 in the first week of production. What happened?**  
     Check for **target/post-launch leakage** (did a feature like `ad_spend` or `clicks` sneak into `X_train`?) or **chronological leakage** (random split instead of `time_based_split`, or `shift(1)` omitted in `features/campaign_features.py`).

242. **Scenario: Inventory Forecasting MAPE spikes from 4% to 38% exclusively for `IN` and `BR` in December.**  
     Missing regional holidays (Diwali/Carnival/local peak travel) or sudden regional supply expansion. Inspect residuals by `(country, month)`, add country-specific holiday calendars, and check `PSI` in `monitoring/data_drift.py`.

243. **Scenario: FastAPI `/forecast/inventory` p95 latency jumps from 35ms to 850ms after a deploy.**  
     Likely cause: `@lru_cache` removed or `latest_history` DataFrame is reloaded/filtered inefficiently on every request, or Redis went down causing cache misses + uncached 30-step Python loop. Profile middleware `X-Process-Time-Ms` and check `/api/v1/health` `cache_backend`.

244. **Scenario: A/B test shows ML targeting has +18% CVR lift (p=0.0001), yet SRM check fails (`srm_p_value = 1e-9`, 58% treatment vs 42% control). Do you ship?**  
     **No!** Never ship on a failed SRM. Investigate client SDK caching, bot filtering asymmetry, or dropped control tracking pixels before trusting any lift number.

245. **Scenario: LightGBM crashes in production Docker container with `libomp.dylib` / `libgomp.so.1` cannot be loaded.**  
     Missing OpenMP runtime shared library in slim base image (`python:3.11-slim`). Fix in `Dockerfile` by installing `libgomp1` (Linux) or linking `libomp` (macOS).

246. **Scenario: Spark Structured Streaming memory grows monotonically until OOMKill after 12 hours.**  
     Stateful aggregation (`groupBy(window, ...)`) without a valid `.withWatermark(...)` or `dropDuplicates` without watermark bounds retains infinite state in RocksDB/JVM heap. Verify watermark progression and event-timestamp parsing.

247. **Scenario: Audience Reach predictions return identical values (`1,420`) for every new campaign in a newly launched country `KR`.**  
     `OneHotEncoder(handle_unknown="ignore")` encodes `KR` as all zeros, and cold-start `target_audience_size` defaulted to `NaN` -> median imputation! Populate regional audience pool sizes dynamically from `travelers` before inference.

248. **Scenario: Propensity Score Matching drops 80% of treated campaigns during matching (`matched_pairs` collapses).**  
     Positivity/Common Support violation: treated campaigns have extreme budgets/intent with almost no comparable control units inside the `0.2 * SD` caliper. Relax overlap checks only after inspecting propensity histograms, or narrow inference to the overlap subpopulation (ATO).

249. **Scenario: Model monitoring triggers a critical `prediction_drift` alert (`PSI=0.34`), yet `model_drift` RMSE hasn't changed.**  
     Legitimate business mix shift (e.g., advertisers shifted budgets toward higher-value Q4 holiday campaigns, shifting $\hat{Y}$ upward while $P(Y|X)$ remains accurate). Confirm by slicing RMSE across budget quartiles before retraining unnecessarily.

250. **Scenario: XGBoost predicts negative `unique_reach` (-120) for a \$1,200 micro-campaign.**  
     Unconstrained tree/linear regression residuals can extrapolate below zero on small targets. Enforce non-negativity (`np.maximum(0.0, preds)`) or train on `log1p(unique_reach)` / `count:poisson` objective.

251. **Scenario: PySpark batch ETL duplicates yesterday's rows in PostgreSQL after an Airflow retry.**  
     Pipeline lacked idempotency: `if_exists="append"` ran twice without partition overwrite or primary-key upsert. Fix using `mode("overwrite")` on date partitions or SQL `ON CONFLICT DO UPDATE` + `_etl_watermark.json`.

252. **Scenario: EconML `CausalForestDML` predicts negative uplift (`Sleeping Dogs`) for Platinum loyalty users—is it a bug or real?**  
     Check decile observed lift (`decile_uplift` in `uplift_modeling.py`): Platinum users already book frequently organically ("Sure Things"), and aggressive ML retargeting can cause ad fatigue or wasted spend. Validate with a targeted holdout test!

253. **Scenario: Evidently HTML report generation fails after upgrading Pandas from 2.3 to 3.0.**  
     Breaking upstream Pandas internal API changes (`infer_dtype_from_object`). Pin validated versions (`pandas>=2.2.0,<2.4.0`) in `requirements.txt` and catch version regressions in CI `pytest`.

254. **Scenario: Difference-in-Differences shows +2.5% lift, but pre-trend placebo check 30 days earlier also shows +2.1% lift.**  
     Parallel trends assumption is violated! Treatment markets were already accelerating faster than comparison markets before ML rollout. Switch to Synthetic Control or Synthetic DiD matching pre-treatment trajectories.

255. **Scenario: Batch pipeline finishes, MLflow logs run, but FastAPI still serves yesterday's model predictions.**  
     Two causes: 1) `@lru_cache` inside `predict.py` kept old `best_model.pkl` in memory, and 2) Redis still holds cached responses for up to 300s. Add model reload endpoint / graceful rolling pod restart + version-prefixed Redis keys.

256. **Scenario: `campaign_ctr` is 0.045 overall, but when aggregated by `campaign_id` average CTR is 0.092.**  
     Simpson's / unweighted-average trap: tiny campaigns with 10 impressions and 1 click (10% CTR) skew unweighted means. Always compute impression-weighted CTR (`SUM(clicks) / SUM(impressions)`).

257. **Scenario: Spark job takes 45 minutes on `ad_impressions` because 1 partition handles `country='US'`.**  
     Classic key skew on `partitionBy`/`groupBy("country")`. Enable AQE skew join, include `campaign_id` or date salt in intermediate shuffles, and broadcast small tables.

258. **Scenario: Campaign Performance Classifier has 0.91 ROC-AUC, yet Precision at 0.5 threshold is only 0.42.**  
     Class imbalance or uncalibrated probabilities shifted optimal decision threshold. Inspect Precision-Recall curve, calibrate via isotonic regression, and tune threshold to business cost matrix.

259. **Scenario: Kafka producer drops messages silently during a brief broker restart.**  
     `AdEventKafkaProducer` needs `acks="all"`, `retries=5`, and `linger_ms` tuning, plus checking `future.get(timeout=10)` or routing failures to local retry buffer.

260. **Scenario: Inventory forecast 30-day curve flattens into a straight line after day 10.**  
     Recursive forecasting lost high-frequency signal because `booked_buf` wasn't updated dynamically (`booked_buf[-7]`), or tree depth wasn't capturing cyclic `sin_doy`/`sin_dow` strong enough. Verify cyclic features & direct horizon models (`target_30d_ahead`).

---

## Section 9: Business & Stakeholder Communication Scenarios (15 Questions)

261. **A Product VP asks: "Why does your Audience Reach model output a confidence interval `[1,850, 2,410]` instead of one exact number?"**  
     "Ad auctions and traveler browsing fluctuate daily. Giving advertisers a calibrated 95% interval lets them plan minimum guaranteed reach (`1,850`) for SLA commitments while sizing upside (`2,410`)—building trust rather than false precision."

262. **Marketing Leadership sees observational data where ML-targeted campaigns had 2× higher CVR and wants to claim +100% causal lift in a press release. How do you respond?**  
     Walk through `causal_impact_report.md`: show how naive comparison includes self-selection bias (higher-intent travelers & larger budgets opted into ML). Present the cluster-randomized A/B lift and PSM-matched ATT as the defensible causal truth so leadership cites accurate ROI.

263. **An enterprise hotel chain asks why their campaign performance forecast dropped when they increased budget from \$50K to \$250K in one week.**  
     Explain **audience saturation and frequency caps**: once budget exceeds the `target_audience_size` capacity in that country/placement, incremental impressions hit the same travelers repeatedly, driving diminishing marginal CTR and ROAS. Recommend expanding placements or target geos.

264. **Finance wants to know the dollar impact of deploying Causal Uplift targeting vs targeting everyone.**  
     Translate `uplift_segments.csv` into budget math: suppressing ads to *Lost Causes* and *Sleeping Dogs* while reallocating budget to *Persuadables* captures higher incremental conversions per \$1,000 spent—improving net incremental ROAS rather than paying for conversions that would happen organically (*Sure Things*).

265. **An Ad Ops Director asks which forecasting algorithm (Prophet vs XGBoost vs SARIMA) to trust for Q4 inventory planning.**  
     Show the side-by-side validation scorecard (`model_comparison.csv` in Dashboard Section 7): highlight WAPE & MAPE over the 30-day peak holdout, explain *why* the winning model won (e.g., capturing both weekly seasonality and advertiser demand pressure), and emphasize that MLflow tracks every contender objectively.

266. **Engineering leadership questions the cost of maintaining both a Batch PySpark pipeline and Kafka Structured Streaming.**  
     Frame the ROI: batch handles heavy daily historical joins, model retraining, and accounting-grade PostgreSQL KPIs cheaply on Spot EMR; streaming is scoped strictly to 5-minute real-time anomaly detection and fresh targeting signals where 24-hour latency would lose high-intent travelers.

267. **A stakeholder notices `model_drift` triggered a warning alert—should we immediately halt all automated bidding?**  
     "No—a warning (`PSI 0.10–0.25`) indicates moderate input shift, not broken predictions. We inspect whether business mix shifted (e.g., summer travel surge) and check RMSE/WAPE before deciding between scheduled retraining vs immediate rollback."

268. **How do you explain `WAPE = 4.2%` vs `R² = 0.89` to a non-technical Advertiser Account Manager?**  
     "WAPE of 4.2% means across every 100,000 forecasted impressions, our predictions are within ~4,200 impressions on average—accurate enough to commit inventory guarantees with a small safety buffer."

269. **Sales asks for a "guaranteed conversion count" before a brand-new advertiser launches their first campaign.**  
     Explain cold-start uncertainty: without advertiser history, Model 3 relies on channel/segment priors (`channel_hist_ctr`, `target_segment_size`) and provides an expected range plus `high_performance_probability`, recommending a 2-week test-and-learn phase first.

270. **A PM wants to run an A/B test on ML targeting for only 2 days to hit a Friday launch deadline.**  
     Explain travel booking cycles: traveler shopping differs sharply between weekdays and weekends (`is_weekend` effect) and conversion lag exists between ad click and hotel booking. Running <7–14 days risks novelty bias and underpowered conclusions; propose running at least one full weekly cycle with CUPED variance reduction.

271. **How do you justify investing in Evidently AI & Prometheus monitoring before adding Model 4?**  
     "Silent model decay in ad tech directly wastes advertiser budgets or misprices inventory. Observability cuts mean-time-to-detection from weeks of lost revenue to minutes, protecting all three production models simultaneously."

272. **An executive asks why `Random Forest` lost to `LightGBM` or `XGBoost` on Audience Reach.**  
     Explain concisely: LightGBM/XGBoost sequentially correct residual errors along reach saturation curves, achieving lower RMSE/WAPE and faster inference latency in FastAPI, while Random Forest serves as a valuable stability benchmark.

273. **How do you align Data Engineering, Applied Science, and Backend teams on ownership?**  
     Define crisp contracts: Data Engineering owns `extract.py`/`transform.py` SLAs and schemas; Applied Science owns `features/`, `models/`, `causal_inference/`, and MLflow quality gates; ML/Backend Engineering owns `api/`, Redis/Docker scaling, and Prometheus latency SLAs.

274. **Stakeholders ask why Platinum loyalty travelers have lower predicted incremental uplift than Silver/Gold travelers.**  
     Platinum members already book organically at high frequency (*Sure Things*) or experience diminishing returns from extra ads; Silver/Gold mobile shoppers with mid-to-high travel intent are the true *Persuadables* where ML targeting changes booking behavior.

275. **Summarize the entire platform's business value in a 60-second executive pitch.**  
     "Our platform transforms raw Expedia-style travel clickstream, auction, and inventory logs into four revenue-driving capabilities: (1) accurate pre-launch Reach and Campaign Performance forecasts so advertisers size budgets confidently, (2) 7/14/30-day Inventory Availability forecasts preventing overselling or wasted supply, (3) Causal Uplift targeting proven via cluster-robust A/B testing and EconML to allocate spend strictly toward persuadable travelers, and (4) automated MLflow + Evidently + FastAPI serving with real-time drift alerting."
