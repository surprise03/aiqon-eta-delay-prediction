"""Checkpoint-aware feature engineering.

ETA is re-estimated as the shipment progresses. At each checkpoint only the
information available at that moment is used (no leakage from the future):

  C0 booking         static route / carrier / cargo / forecasts
  C1 departed        + departure delay
  C2 port arrival    + arrival delay, observed congestion, berth-window logic
  C3 customs release + port dwell, customs outcome, truck-appointment slack

ETA definition:
    ETA = scheduled_delivery + known_delay_so_far + predicted_remaining_delay
The model predicts *remaining* delay, so it must learn whether the delay
already incurred will be absorbed (buffers) or amplified (missed windows).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CARGO, DEST_PORTS, PLAN, ROUTES

CHECKPOINTS = ["C0_booking", "C1_departed", "C2_port_arrival", "C3_customs_release"]
CHECKPOINT_LABEL = {
    "C0_booking": "At booking", "C1_departed": "After departure",
    "C2_port_arrival": "After port arrival", "C3_customs_release": "After customs release",
}
H = pd.Timedelta(hours=1)

# Plain-language groups used to explain predictions
GROUPS = {
    "Upstream delay (propagation)": ["dep_delay_h", "arrival_delay_h", "release_delay_h", "early_arrival_h"],
    "Origin & carrier": ["carrier_reliability", "origin_congestion", "rolled_over_obs"],
    "Ocean transit & weather": ["weather_forecast", "suez_route", "panama_route", "transit_delay_h",
                                "transit_days"],
    "Destination port congestion": ["congestion_forecast", "congestion_at_arrival", "missed_berth_window",
                                    "late_x_congestion", "berth_wait_h", "discharge_h", "containers",
                                    "arrival_imputed", "port_events_missing"],
    "Customs & documentation": ["docs_complete", "inspection_risk", "customs_inspection", "customs_h",
                                "sched_release_weekend", "proj_release_weekend", "hours_to_weekend"],
    "Inland / truck appointment": ["peak_season", "appointment_slack_h", "missed_appointment", "inland_hours"],
}
FEATURE_GROUP = {f: g for g, fs in GROUPS.items() for f in fs}

# true stage increments that each explanation group corresponds to (for validation)
GROUP_TO_INCREMENTS = {
    "Destination port congestion": ["inc_berth", "inc_discharge"],
    "Customs & documentation": ["inc_customs"],
    "Inland / truck appointment": ["inc_inland"],
}


def _hours(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b) / H


def _hours_to_weekend(t: pd.Series) -> pd.Series:
    """Working hours left before Saturday 00:00 (0 if already weekend)."""
    sat = t.dt.normalize() + pd.to_timedelta((5 - t.dt.weekday).clip(lower=0), unit="D")
    return ((sat - t) / H).clip(lower=0)


def _static(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["carrier_reliability"] = df["carrier_reliability"]
    X["origin_congestion"] = df["origin_congestion"]
    X["weather_forecast"] = df["weather_forecast"]
    X["congestion_forecast"] = df["congestion_forecast"]
    X["suez_route"] = (df["chokepoint"] == "suez").astype(int)
    X["panama_route"] = (df["chokepoint"] == "panama").astype(int)
    X["transit_days"] = df["route"].map(lambda r: ROUTES[r]["transit_days"])
    X["inland_hours"] = df["route"].map(lambda r: ROUTES[r]["inland_hours"])
    X["containers"] = df["containers"]
    X["docs_complete"] = df["docs_complete"]
    X["peak_season"] = df["peak_season"]
    # interaction: cargo risk x documentation x port strictness
    base_p = df["dest_port"].map(lambda p: DEST_PORTS[p]["inspection_p"])
    X["inspection_risk"] = base_p * df["cargo_type"].map(CARGO) * np.where(df["docs_complete"] == 1, 1.0, 1.6)
    X["sched_release_weekend"] = (df["sched_customs_release"].dt.weekday >= 5).astype(int)
    return X


def build_features(df: pd.DataFrame, checkpoint: str):
    """Return (X, known_delay_h) for a checkpoint using only information available then."""
    k = CHECKPOINTS.index(checkpoint)
    X = _static(df)
    known = pd.Series(0.0, index=df.index)

    if k >= 1:  # ---- departed
        dep = _hours(df["act_departure"], df["sched_departure"])
        X["dep_delay_h"] = dep
        X["rolled_over_obs"] = (dep >= 120).astype(int)
        known = dep.copy()

    if k >= 2:  # ---- arrived at destination port
        arr = _hours(df["act_port_arrival"], df["sched_port_arrival"])
        # missing-data handling: reconstruct arrival from berth time if the AIS event is missing
        imputed = arr.isna() & df["act_berth"].notna()
        typical_wait = 3 + 50 * df["congestion_at_arrival"] ** 3
        recon = _hours(df["act_berth"] - typical_wait * H, df["sched_port_arrival"])
        arr = arr.where(~imputed, recon)
        X["arrival_imputed"] = imputed.astype(int)
        X["arrival_delay_h"] = arr
        X["transit_delay_h"] = arr - X["dep_delay_h"]
        X["early_arrival_h"] = arr.clip(upper=0)
        X["congestion_at_arrival"] = df["congestion_at_arrival"]
        X["missed_berth_window"] = (arr > 24).astype(float).where(arr.notna())
        X["late_x_congestion"] = arr.clip(lower=0) * df["congestion_at_arrival"]
        # projected customs start -> will clearance run into a weekend?
        act_arr = df["sched_port_arrival"] + arr.fillna(X["dep_delay_h"]) * H
        proj_dis = act_arr + (typical_wait + PLAN["discharge"]) * H
        X["hours_to_weekend"] = _hours_to_weekend(proj_dis)
        cust_h = df["dest_port"].map(lambda p: DEST_PORTS[p]["customs_hours"])
        X["proj_release_weekend"] = (X["hours_to_weekend"] < cust_h).astype(int)
        known = arr.fillna(known)

    if k >= 3:  # ---- customs released
        rel = _hours(df["act_customs_release"], df["sched_customs_release"])
        X["release_delay_h"] = rel
        X["berth_wait_h"] = _hours(df["act_berth"], df["act_port_arrival"])
        X["discharge_h"] = _hours(df["act_discharge"], df["act_berth"])
        X["customs_h"] = _hours(df["act_customs_release"], df["act_discharge"])
        X["port_events_missing"] = df[["act_berth", "act_discharge"]].isna().sum(axis=1)
        X["customs_inspection"] = df["customs_inspection"]
        slack = _hours(df["truck_appointment"], df["act_customs_release"])
        X["appointment_slack_h"] = slack
        X["missed_appointment"] = (slack < 0).astype(float).where(slack.notna())
        known = rel.fillna(known)

    return X.astype(float), known.astype(float)


def group_of(feature: str) -> str:
    return FEATURE_GROUP.get(feature, "Other")
