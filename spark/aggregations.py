"""PySpark Business Aggregations calculating all 9 core advertising intelligence KPIs."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def compute_business_aggregations(transformed: dict[str, DataFrame]) -> dict[str, DataFrame]:
    """Compute the 9 required business aggregation tables using PySpark DataFrames & SQL."""
    imp = transformed["ad_impressions_enriched"]
    inv = transformed["ad_inventory"]
    bid = transformed["bidding_data"]

    # 1. Daily impressions
    daily_impressions = (
        imp.groupBy("event_date")
        .agg(
            F.count("impression_id").alias("total_impressions"),
            F.countDistinct("traveler_id").alias("unique_travelers"),
            F.sum("clicked").alias("total_clicks"),
            F.sum("converted").alias("total_conversions"),
            F.round(F.sum("impression_cost"), 2).alias("total_ad_spend"),
            F.round(F.sum("attributed_revenue"), 2).alias("attributed_revenue"),
        )
        .orderBy("event_date")
    )

    # 2. Monthly impressions
    monthly_impressions = (
        imp.groupBy("year", "month")
        .agg(
            F.count("impression_id").alias("total_impressions"),
            F.countDistinct("traveler_id").alias("unique_travelers"),
            F.sum("clicked").alias("total_clicks"),
            F.sum("converted").alias("total_conversions"),
            F.round(F.sum("impression_cost"), 2).alias("total_ad_spend"),
            F.round(F.sum("attributed_revenue"), 2).alias("attributed_revenue"),
        )
        .orderBy("year", "month")
    )

    # 3. Campaign-wise revenue & performance (includes 6 & 7: CTR & CVR)
    campaign_metrics = (
        imp.groupBy(
            "campaign_id",
            "advertiser_id",
            "target_country",
            "campaign_objective",
            "advertising_channel",
            "budget",
            "campaign_duration_days",
        )
        .agg(
            F.count("impression_id").alias("impressions"),
            F.countDistinct("traveler_id").alias("unique_reach"),
            F.sum("clicked").alias("clicks"),
            F.sum("converted").alias("conversions"),
            F.round(F.sum("impression_cost"), 2).alias("ad_spend"),
            F.round(F.sum("attributed_revenue"), 2).alias("campaign_revenue"),
            F.round(F.avg("travel_intent"), 4).alias("avg_travel_intent"),
            F.round(F.avg("engagement_score"), 2).alias("avg_engagement_score"),
        )
        .withColumn(
            "ctr",
            F.round(
                F.when(F.col("impressions") > 0, F.col("clicks") / F.col("impressions")).otherwise(
                    0.0
                ),
                6,
            ),
        )
        .withColumn(
            "conversion_rate",
            F.round(
                F.when(
                    F.col("impressions") > 0, F.col("conversions") / F.col("impressions")
                ).otherwise(0.0),
                6,
            ),
        )
        .withColumn(
            "roas",
            F.round(
                F.when(
                    F.col("ad_spend") > 0, F.col("campaign_revenue") / F.col("ad_spend")
                ).otherwise(0.0),
                4,
            ),
        )
    )

    # 3b. Dedicated Campaign-wise revenue summary
    campaign_revenue = campaign_metrics.select(
        "campaign_id",
        "advertiser_id",
        "target_country",
        "advertising_channel",
        "budget",
        "ad_spend",
        "campaign_revenue",
        "roas",
    )

    # 4. Audience reach by country
    audience_reach_by_country = (
        imp.groupBy("country")
        .agg(
            F.countDistinct("traveler_id").alias("unique_reach"),
            F.count("impression_id").alias("total_impressions"),
            F.sum("clicked").alias("total_clicks"),
            F.sum("converted").alias("total_conversions"),
            F.round(F.sum("impression_cost"), 2).alias("total_spend"),
        )
        .withColumn(
            "avg_frequency",
            F.round(F.col("total_impressions") / F.greatest(F.col("unique_reach"), F.lit(1)), 2),
        )
        .orderBy(F.desc("unique_reach"))
    )

    # 5. Available inventory by placement
    inventory_by_placement = (
        inv.groupBy("placement", "country")
        .agg(
            F.sum("available_impressions").alias("total_available_impressions"),
            F.sum("booked_impressions").alias("total_booked_impressions"),
            F.sum("remaining_impressions").alias("total_remaining_impressions"),
            F.round(F.avg("utilization_rate"), 4).alias("avg_utilization_rate"),
        )
        .orderBy("country", "placement")
    )

    # 6. Campaign CTR
    campaign_ctr = campaign_metrics.select(
        "campaign_id", "advertising_channel", "impressions", "clicks", "ctr"
    )

    # 7. Campaign conversion rate
    campaign_cvr = campaign_metrics.select(
        "campaign_id",
        "campaign_objective",
        "impressions",
        "clicks",
        "conversions",
        "conversion_rate",
    )

    # 8. Average bidding price
    avg_bidding_price = (
        bid.groupBy("placement", "auction_result")
        .agg(
            F.count("bid_id").alias("bid_count"),
            F.round(F.avg("bid_amount"), 2).alias("avg_bid_amount"),
            F.round(F.expr("percentile_approx(bid_amount, 0.5)"), 2).alias("median_bid_amount"),
            F.round(F.max("bid_amount"), 2).alias("max_bid_amount"),
        )
        .orderBy("placement", "auction_result")
    )

    # 9. Audience engagement by segment
    audience_engagement_by_segment = (
        imp.groupBy("loyalty_segment", "travel_interest", "device_type")
        .agg(
            F.countDistinct("traveler_id").alias("unique_travelers"),
            F.count("impression_id").alias("impressions"),
            F.round(F.avg("engagement_score"), 2).alias("avg_engagement_score"),
            F.round(F.avg("travel_intent"), 4).alias("avg_travel_intent"),
            F.round(F.sum("clicked") / F.count("impression_id"), 6).alias("segment_ctr"),
            F.round(F.sum("converted") / F.count("impression_id"), 6).alias("segment_cvr"),
        )
        .orderBy(F.desc("avg_engagement_score"))
    )

    return {
        "daily_impressions": daily_impressions,
        "monthly_impressions": monthly_impressions,
        "campaign_metrics": campaign_metrics,
        "campaign_revenue": campaign_revenue,
        "audience_reach_by_country": audience_reach_by_country,
        "inventory_by_placement": inventory_by_placement,
        "campaign_ctr": campaign_ctr,
        "campaign_cvr": campaign_cvr,
        "avg_bidding_price": avg_bidding_price,
        "audience_engagement_by_segment": audience_engagement_by_segment,
    }
