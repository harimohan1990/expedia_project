"""PySpark Transform layer: cleaning, deduplication, validation, normalization, joins, and window features."""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DoubleType, IntegerType, TimestampType

from configs.settings import get_logger

logger = get_logger(__name__)


@dataclass
class DataQualityReport:
    dataset: str
    raw_count: int
    clean_count: int
    duplicates_removed: int
    invalid_or_null_removed: int
    referential_orphans_removed: int = 0


def clean_travelers(df: DataFrame) -> tuple[DataFrame, DataQualityReport]:
    raw_count = df.count()
    deduped = df.dropDuplicates(["traveler_id"])
    after_dedup = deduped.count()

    cleaned = (
        deduped.filter(F.col("traveler_id").isNotNull())
        .withColumn("country", F.upper(F.trim(F.col("country"))))
        .withColumn("city", F.initcap(F.trim(F.col("city"))))
        .withColumn("device_type", F.lower(F.trim(F.col("device_type"))))
        .withColumn("travel_interest", F.lower(F.trim(F.col("travel_interest"))))
        .withColumn("loyalty_segment", F.initcap(F.trim(F.col("loyalty_segment"))))
        .withColumn(
            "booking_frequency",
            F.coalesce(F.col("booking_frequency").cast(IntegerType()), F.lit(1)),
        )
        .filter(F.col("booking_frequency") >= 0)
    )
    clean_count = cleaned.count()
    report = DataQualityReport(
        dataset="travelers",
        raw_count=raw_count,
        clean_count=clean_count,
        duplicates_removed=raw_count - after_dedup,
        invalid_or_null_removed=after_dedup - clean_count,
    )
    return cleaned, report


def clean_campaigns(df: DataFrame) -> tuple[DataFrame, DataQualityReport]:
    raw_count = df.count()
    deduped = df.dropDuplicates(["campaign_id"])
    after_dedup = deduped.count()

    cleaned = (
        deduped.filter(F.col("campaign_id").isNotNull())
        .withColumn("start_date", F.to_date(F.col("start_date")).cast(DateType()))
        .withColumn("end_date", F.to_date(F.col("end_date")).cast(DateType()))
        .withColumn("budget", F.col("budget").cast(DoubleType()))
        .withColumn("target_country", F.upper(F.trim(F.col("target_country"))))
        .withColumn("campaign_objective", F.lower(F.trim(F.col("campaign_objective"))))
        .withColumn("advertising_channel", F.lower(F.trim(F.col("advertising_channel"))))
        .filter(
            (F.col("budget") > 0)
            & F.col("start_date").isNotNull()
            & F.col("end_date").isNotNull()
            & (F.col("end_date") >= F.col("start_date"))
        )
        .withColumn(
            "campaign_duration_days", F.datediff(F.col("end_date"), F.col("start_date")) + 1
        )
    )
    clean_count = cleaned.count()
    report = DataQualityReport(
        dataset="campaigns",
        raw_count=raw_count,
        clean_count=clean_count,
        duplicates_removed=raw_count - after_dedup,
        invalid_or_null_removed=after_dedup - clean_count,
    )
    return cleaned, report


def clean_ad_inventory(df: DataFrame) -> tuple[DataFrame, DataQualityReport]:
    raw_count = df.count()
    deduped = df.dropDuplicates(["inventory_id"])
    after_dedup = deduped.count()

    cleaned = (
        deduped.filter(F.col("inventory_id").isNotNull())
        .withColumn("date", F.to_date(F.col("date")).cast(DateType()))
        .withColumn("placement", F.lower(F.trim(F.col("placement"))))
        .withColumn("country", F.upper(F.trim(F.col("country"))))
        .withColumn("available_impressions", F.col("available_impressions").cast(IntegerType()))
        .withColumn("booked_impressions", F.col("booked_impressions").cast(IntegerType()))
        .withColumn("remaining_impressions", F.col("remaining_impressions").cast(IntegerType()))
        .filter(
            F.col("date").isNotNull()
            & (F.col("available_impressions") >= 0)
            & (F.col("booked_impressions") >= 0)
            & (F.col("remaining_impressions") >= 0)
        )
        .withColumn("year", F.year("date"))
        .withColumn("month", F.month("date"))
        .withColumn("day_of_week", F.dayofweek("date"))
        .withColumn(
            "utilization_rate",
            F.when(
                F.col("available_impressions") > 0,
                F.col("booked_impressions") / F.col("available_impressions"),
            ).otherwise(0.0),
        )
    )

    # Window feature: 7-day rolling average of available and booked impressions
    inv_window = Window.partitionBy("country", "placement").orderBy("date").rowsBetween(-6, 0)
    cleaned = cleaned.withColumn(
        "rolling_7d_available", F.avg("available_impressions").over(inv_window)
    ).withColumn("rolling_7d_booked", F.avg("booked_impressions").over(inv_window))

    clean_count = cleaned.count()
    report = DataQualityReport(
        dataset="ad_inventory",
        raw_count=raw_count,
        clean_count=clean_count,
        duplicates_removed=raw_count - after_dedup,
        invalid_or_null_removed=after_dedup - clean_count,
    )
    return cleaned, report


def clean_bidding_data(
    df: DataFrame, campaigns_df: DataFrame
) -> tuple[DataFrame, DataQualityReport]:
    raw_count = df.count()
    deduped = df.dropDuplicates(["bid_id"])
    after_dedup = deduped.count()

    valid = (
        deduped.filter(F.col("bid_id").isNotNull())
        .withColumn("placement", F.lower(F.trim(F.col("placement"))))
        .withColumn("auction_result", F.lower(F.trim(F.col("auction_result"))))
        .withColumn("bid_amount", F.col("bid_amount").cast(DoubleType()))
        .withColumn(
            "auction_timestamp", F.to_timestamp(F.col("auction_timestamp")).cast(TimestampType())
        )
        .filter(
            (F.col("bid_amount") > 0)
            & F.col("auction_timestamp").isNotNull()
            & F.col("auction_result").isin("won", "lost", "floor_rejected")
        )
        .withColumn("auction_date", F.to_date("auction_timestamp"))
    )
    after_valid = valid.count()

    # Referential integrity against campaigns
    ref_checked = valid.join(
        campaigns_df.select("campaign_id").distinct(), on="campaign_id", how="left_semi"
    )
    clean_count = ref_checked.count()

    report = DataQualityReport(
        dataset="bidding_data",
        raw_count=raw_count,
        clean_count=clean_count,
        duplicates_removed=raw_count - after_dedup,
        invalid_or_null_removed=after_dedup - after_valid,
        referential_orphans_removed=after_valid - clean_count,
    )
    return ref_checked, report


def clean_targeting_signals(
    df: DataFrame, travelers_df: DataFrame
) -> tuple[DataFrame, DataQualityReport]:
    raw_count = df.count()
    typed = (
        df.filter(F.col("traveler_id").isNotNull())
        .withColumn("destination_interest", F.initcap(F.trim(F.col("destination_interest"))))
        .withColumn(
            "signal_timestamp", F.to_timestamp(F.col("signal_timestamp")).cast(TimestampType())
        )
        .withColumn("travel_intent", F.col("travel_intent").cast(DoubleType()))
        .withColumn("booking_probability", F.col("booking_probability").cast(DoubleType()))
        .withColumn("engagement_score", F.col("engagement_score").cast(DoubleType()))
        .withColumn("search_frequency", F.col("search_frequency").cast(IntegerType()))
        .filter(
            F.col("signal_timestamp").isNotNull()
            & (F.col("travel_intent").between(0.0, 1.0))
            & (F.col("booking_probability").between(0.0, 1.0))
        )
    )
    deduped = typed.dropDuplicates(["traveler_id", "signal_timestamp"])
    after_dedup = deduped.count()

    ref_checked = deduped.join(
        travelers_df.select("traveler_id").distinct(), on="traveler_id", how="left_semi"
    )
    clean_count = ref_checked.count()
    report = DataQualityReport(
        dataset="targeting_signals",
        raw_count=raw_count,
        clean_count=clean_count,
        duplicates_removed=len(typed.columns) * 0 + (raw_count - after_dedup),
        invalid_or_null_removed=0,
        referential_orphans_removed=after_dedup - clean_count,
    )
    return ref_checked, report


def clean_and_enrich_impressions(
    impressions_df: DataFrame,
    travelers_df: DataFrame,
    campaigns_df: DataFrame,
    signals_df: DataFrame,
) -> tuple[DataFrame, DataQualityReport]:
    raw_count = impressions_df.count()
    deduped = impressions_df.dropDuplicates(["impression_id"])
    after_dedup = deduped.count()

    cleaned = (
        deduped.filter(
            F.col("impression_id").isNotNull()
            & F.col("campaign_id").isNotNull()
            & F.col("traveler_id").isNotNull()
        )
        .withColumn("timestamp", F.to_timestamp(F.col("timestamp")).cast(TimestampType()))
        .withColumn("placement", F.lower(F.trim(F.col("placement"))))
        .withColumn("device_type", F.lower(F.trim(F.col("device_type"))))
        .withColumn("impression_cost", F.col("impression_cost").cast(DoubleType()))
        .withColumn("clicked", F.coalesce(F.col("clicked").cast(IntegerType()), F.lit(0)))
        .withColumn("converted", F.coalesce(F.col("converted").cast(IntegerType()), F.lit(0)))
        .filter(
            F.col("timestamp").isNotNull()
            & (F.col("impression_cost") >= 0.0)
            & F.col("clicked").isin(0, 1)
            & F.col("converted").isin(0, 1)
        )
        .withColumn("event_date", F.to_date("timestamp"))
        .withColumn("year", F.year("timestamp"))
        .withColumn("month", F.month("timestamp"))
        .withColumn("day_of_week", F.dayofweek("timestamp"))
        .withColumn("hour", F.hour("timestamp"))
        .withColumn("is_weekend", F.when(F.dayofweek("timestamp").isin(1, 7), 1).otherwise(0))
    )
    after_valid = cleaned.count()

    # Latest targeting signal per traveler using window row_number
    sig_win = Window.partitionBy("traveler_id").orderBy(F.col("signal_timestamp").desc())
    latest_signals = (
        signals_df.withColumn("rn", F.row_number().over(sig_win))
        .filter(F.col("rn") == 1)
        .drop("rn", "signal_timestamp")
    )

    # Referential integrity inner joins with travelers and campaigns
    enriched = (
        cleaned.join(
            campaigns_df.select(
                "campaign_id",
                "advertiser_id",
                "budget",
                "target_country",
                "target_age_group",
                "campaign_objective",
                "advertising_channel",
                "campaign_duration_days",
            ),
            on="campaign_id",
            how="inner",
        )
        .join(
            travelers_df.select(
                "traveler_id",
                "age_group",
                "country",
                "city",
                "travel_interest",
                "booking_frequency",
                "loyalty_segment",
            ),
            on="traveler_id",
            how="inner",
        )
        .join(latest_signals, on="traveler_id", how="left")
        .withColumn("travel_intent", F.coalesce(F.col("travel_intent"), F.lit(0.35)))
        .withColumn("booking_probability", F.coalesce(F.col("booking_probability"), F.lit(0.18)))
        .withColumn("engagement_score", F.coalesce(F.col("engagement_score"), F.lit(42.0)))
        .withColumn(
            "is_target_match",
            F.when(
                (F.col("country") == F.col("target_country"))
                & (F.col("age_group") == F.col("target_age_group")),
                1,
            ).otherwise(0),
        )
        # Modeled booking revenue per conversion ($145 average margin-adjusted travel booking value)
        .withColumn(
            "attributed_revenue",
            F.when(F.col("converted") == 1, F.lit(145.0)).otherwise(F.lit(0.0)),
        )
    )

    # Window function: cumulative spend per campaign ordered by timestamp
    camp_win = (
        Window.partitionBy("campaign_id")
        .orderBy("timestamp")
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    enriched = enriched.withColumn(
        "cumulative_campaign_spend", F.sum("impression_cost").over(camp_win)
    )

    clean_count = enriched.count()
    report = DataQualityReport(
        dataset="ad_impressions_enriched",
        raw_count=raw_count,
        clean_count=clean_count,
        duplicates_removed=raw_count - after_dedup,
        invalid_or_null_removed=after_dedup - after_valid,
        referential_orphans_removed=after_valid - clean_count,
    )
    return enriched, report


def transform_all(
    raw_datasets: dict[str, DataFrame],
) -> tuple[dict[str, DataFrame], list[DataQualityReport]]:
    """Run all cleaning, referential checks, enrichment, and window transformations."""
    reports: list[DataQualityReport] = []

    travelers_clean, r_trav = clean_travelers(raw_datasets["travelers"])
    reports.append(r_trav)

    campaigns_clean, r_camp = clean_campaigns(raw_datasets["campaigns"])
    reports.append(r_camp)

    inventory_clean, r_inv = clean_ad_inventory(raw_datasets["ad_inventory"])
    reports.append(r_inv)

    bidding_clean, r_bid = clean_bidding_data(raw_datasets["bidding_data"], campaigns_clean)
    reports.append(r_bid)

    signals_clean, r_sig = clean_targeting_signals(
        raw_datasets["targeting_signals"], travelers_clean
    )
    reports.append(r_sig)

    impressions_enriched, r_imp = clean_and_enrich_impressions(
        raw_datasets["ad_impressions"], travelers_clean, campaigns_clean, signals_clean
    )
    reports.append(r_imp)

    for rep in reports:
        logger.info(
            "DQ [%s]: raw=%d clean=%d dup_removed=%d invalid=%d orphans=%d",
            rep.dataset,
            rep.raw_count,
            rep.clean_count,
            rep.duplicates_removed,
            rep.invalid_or_null_removed,
            rep.referential_orphans_removed,
        )

    transformed = {
        "travelers": travelers_clean,
        "campaigns": campaigns_clean,
        "ad_inventory": inventory_clean,
        "bidding_data": bidding_clean,
        "targeting_signals": signals_clean,
        "ad_impressions_enriched": impressions_enriched,
    }
    return transformed, reports
