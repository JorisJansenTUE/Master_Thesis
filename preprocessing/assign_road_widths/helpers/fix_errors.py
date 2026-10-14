import re
from dataclasses import replace
from typing import Dict, Optional

from preprocessing.assign_road_widths.helpers.width_estimation import WidthEstimator, WayWidthResult, WidthStats
from preprocessing.assign_road_widths.width_config import WidthEstimationConfig, FALLBACK_LANE_WIDTHS
class WidthResultFixer:
    """
    Applies rule-based corrections after cadastral width estimation.
    """
    def __init__(self, config: WidthEstimationConfig):
        self.config = config

    def process(self, result: WayWidthResult, tags: Dict[str, str]) -> WayWidthResult:
        name = (tags.get("name") or "").strip().casefold()
        highway = (tags.get("highway") or "").strip().lower()

        lanes = self._parse_lanes(tags)

        stats = result.carriageway

        if stats is None:
            raw_width = None
            n_samples = 0
            spread = None
        else:
            raw_width = stats.median
            n_samples = stats.n
            spread = stats.spread

        result = replace(
            result,
            final_width_m=raw_width,
            lanes_used=lanes,
        )

        # Always initialize these before any condition.
        final_width = raw_width
        method = "cadastral_estimate"

        # Exclude Piazza, Piazzale, Piazetta etc from optimization, as these are not regular roads
        if name.startswith("piaz"):
            return replace(
                result,
                optimizable=False,
                method="piazza_exclusion",
            )

        # largo are genuinely wide, thus we dont want to correct their estimate
        if name.startswith("largo"):
            return replace(
                result,
                optimizable=True,
                method="largo_overwrite",
            )
        

        # unreliable width estimates: too few samples or too much spread
        unreliable_measurement = (
            result.method=="cadastral_estimate"
            and (
                n_samples <= 1
                or (
                    n_samples <= self.config.max_unreliable_samples
                    and spread is not None
                    and spread >= self.config.min_unreliable_spread_m
                )
            )
        )

        if raw_width is None or unreliable_measurement:
            lane_width = FALLBACK_LANE_WIDTHS.get(
                highway,
                self.config.default_lane_width_m,
            )
            final_width = lanes * lane_width
            method = ("missing_width_fallback" if raw_width is None else "unreliability_fallback")

        # Via Stazione in Locarno/Minusio is a special case with a primary road through the city with many lanes. To avoid large overestimation manually correct it to match measurements well:
        if name == "via stazione" and highway == "primary":
            lane_width = self.config.default_lane_width_m
            final_width = lanes * lane_width
            method = "stazione_override"

        # Fix primary one-way roads, cadastral roadspace is often shared whilst roads are seperated in OSM
        if highway == "primary" and self._is_oneway(tags):
            lane_width = FALLBACK_LANE_WIDTHS.get(
                highway,
                self.config.default_lane_width_m,
            )
            final_width = lanes * lane_width

            method = "primary_oneway_fallback"

        return replace(
            result,
            final_width_m=final_width,
            method=method,
        )

    def _parse_lanes(self, tags: Dict[str, str]) -> int:
        lanes_tag = tags.get("lanes")
        if lanes_tag is None:
            return 1 if self._is_oneway(tags) else self.config.default_lanes

        try:
            lanes = int(lanes_tag)

            if lanes <= 0:
                return 1 if self._is_oneway(tags) else self.config.default_lanes
            return lanes
        except ValueError:
            return 1 if self._is_oneway(tags) else self.config.default_lanes

    @staticmethod
    def _is_oneway(tags: Dict[str, str]) -> bool:
        value = (tags.get("oneway") or "").strip().lower()

        return value in {
            "yes",
            "true",
            "1",
            "-1",
        }