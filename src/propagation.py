"""Delay-propagation analysis: how much of an upstream delay reaches the customer?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_features

STAGES = ["inc_origin", "inc_transit", "inc_berth", "inc_discharge", "inc_customs", "inc_inland"]
STAGE_LABEL = {"inc_origin": "Origin departure", "inc_transit": "Ocean transit", "inc_berth": "Berth wait",
               "inc_discharge": "Discharge", "inc_customs": "Customs", "inc_inland": "Inland / trucking"}


def congestion_band(c: pd.Series) -> pd.Series:
    return pd.cut(c, [0, 0.5, 0.7, 1.0], labels=["low (<0.5)", "medium (0.5-0.7)", "high (>0.7)"])


def propagation_factors(obs: pd.DataFrame) -> pd.DataFrame:
    """How much delay is ADDED after the vessel reaches port, by arrival lateness x congestion.

    downstream_added = final delay - arrival delay. Negative means downstream buffers
    absorbed part of the upstream delay; large positive means it was amplified
    (missed berth window, queueing, weekend customs, missed truck slot).
    Medians are used because n per cell is small and delays are heavy-tailed.
    """
    X, _ = build_features(obs, "C2_port_arrival")
    d = pd.DataFrame({"arr": X["arrival_delay_h"], "final": obs["final_delay_h"],
                      "band": congestion_band(obs["congestion_at_arrival"])}).dropna()
    d["lateness"] = pd.cut(d.arr, [-1e9, 0, 24, 1e9], labels=["early/on time", "late 0-24h", "late >24h"])
    rows = []
    for (band, late), g in d.groupby(["band", "lateness"], observed=True):
        added = g.final - g.arr
        rows.append(dict(congestion=str(band), arrival=str(late), n=len(g),
                         median_arrival_delay_h=round(float(g.arr.median()), 1),
                         median_final_delay_h=round(float(g.final.median()), 1),
                         median_downstream_added_h=round(float(added.median()), 1),
                         share_absorbed=round(float((added < 0).mean()), 2)))
    return pd.DataFrame(rows)


def model_implied_curve(model, obs: pd.DataFrame, congestion: float, grid: np.ndarray) -> np.ndarray:
    """Predicted final delay as a function of arrival delay, at a fixed congestion level
    (all other features held at their observed values, then averaged - a partial-dependence curve)."""
    X, _ = build_features(obs, "C2_port_arrival")
    out = []
    for a in grid:
        Z = X.copy()
        Z["arrival_delay_h"] = a
        Z["transit_delay_h"] = a - Z["dep_delay_h"]
        Z["early_arrival_h"] = min(a, 0)
        Z["congestion_at_arrival"] = congestion
        Z["missed_berth_window"] = float(a > 24)
        Z["late_x_congestion"] = max(a, 0) * congestion
        out.append(float(np.mean(a + model.reg.predict(Z))))
    return np.array(out)


def stage_contribution(gt: pd.DataFrame, by: str = "route") -> pd.DataFrame:
    """Mean delay increment per stage, delayed shipments only (where do delays come from?)."""
    d = gt[gt["is_delayed"] == 1]
    return d.groupby(by)[STAGES].mean().rename(columns=STAGE_LABEL)
