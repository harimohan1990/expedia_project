"""Build all feature engineering tables."""

from __future__ import annotations

from configs.settings import PlatformConfig, get_config
from features.audience_features import build_audience_reach_features
from features.campaign_features import build_campaign_performance_features
from features.inventory_features import build_inventory_features


def build_all(config: PlatformConfig | None = None) -> dict[str, int]:
    cfg = config or get_config()
    reach_df = build_audience_reach_features(cfg)
    inv_df = build_inventory_features(cfg)
    camp_df = build_campaign_performance_features(cfg)
    return {
        "audience_reach_features": len(reach_df),
        "inventory_features": len(inv_df),
        "campaign_performance_features": len(camp_df),
    }


if __name__ == "__main__":
    build_all()
