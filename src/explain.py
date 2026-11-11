"""Explanations.

Two complementary views per shipment:
 1. Delay so far  - an exact, assumption-free decomposition of the delay already
    observed into stage increments (milestone-to-milestone). Missing events are
    handled by merging the adjacent stages ("berth -> customs release").
 2. Delay to come - SHAP values of the remaining-delay model, aggregated into
    plain-language cause groups (port congestion, customs, ...).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .features import CHECKPOINTS, GROUP_TO_INCREMENTS, build_features, group_of

H = pd.Timedelta(hours=1)
MILESTONES = [("departure", "sched_departure", "act_departure"),
              ("port arrival", "sched_port_arrival", "act_port_arrival"),
              ("berth", "sched_berth", "act_berth"),
              ("discharge", "sched_discharge", "act_discharge"),
              ("customs release", "sched_customs_release", "act_customs_release")]
LAST_MILESTONE = {"C0_booking": -1, "C1_departed": 0, "C2_port_arrival": 1, "C3_customs_release": 4}

FEATURE_TEXT = {
    "dep_delay_h": "departure delay", "arrival_delay_h": "vessel arrival delay",
    "release_delay_h": "customs release delay", "early_arrival_h": "early arrival (absorbed at anchor)",
    "carrier_reliability": "carrier schedule reliability", "origin_congestion": "origin port congestion",
    "rolled_over_obs": "cargo rolled to next sailing", "weather_forecast": "weather on route",
    "suez_route": "Suez-dependent route", "panama_route": "Panama-dependent route",
    "transit_delay_h": "delay accrued at sea", "transit_days": "voyage length",
    "congestion_forecast": "forecast congestion at destination", "congestion_at_arrival": "port congestion on arrival",
    "missed_berth_window": "missed berth window", "late_x_congestion": "lateness x congestion score",
    "berth_wait_h": "anchorage/berth wait", "discharge_h": "discharge time", "containers": "number of containers",
    "arrival_imputed": "arrival time reconstructed (missing AIS)", "port_events_missing": "missing port events",
    "docs_complete": "documentation completeness", "inspection_risk": "customs inspection risk (cargo x docs)",
    "customs_inspection": "physical customs inspection", "customs_h": "customs processing time",
    "sched_release_weekend": "scheduled release falls on weekend",
    "proj_release_weekend": "clearance projected to hit weekend", "hours_to_weekend": "working hours before weekend",
    "peak_season": "peak season (truck scarcity)", "appointment_slack_h": "slack before truck appointment",
    "missed_appointment": "missed truck appointment", "inland_hours": "inland leg length",
}


def shap_frame(model, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    explainer = shap.TreeExplainer(model.reg)
    sv = explainer.shap_values(X)
    return pd.DataFrame(sv, index=X.index, columns=X.columns), float(np.ravel(explainer.expected_value)[0])


def group_shap(sv: pd.DataFrame) -> pd.DataFrame:
    return sv.T.groupby(lambda f: group_of(f)).sum().T


def delay_so_far(row: pd.Series, checkpoint: str) -> list[tuple[str, float]]:
    """Exact stage-by-stage breakdown of observed delay up to the checkpoint."""
    last = LAST_MILESTONE[checkpoint]
    out, prev_name, prev_delay = [], "schedule", 0.0
    for name, s, a in MILESTONES[: last + 1]:
        if pd.isna(row[a]):
            continue  # missing event: its delay is merged into the next observed span
        d = (row[a] - row[s]) / H
        span = f"origin (until {name})" if prev_name == "schedule" else f"{prev_name} -> {name}"
        out.append((span, d - prev_delay))
        prev_name, prev_delay = name, d
    return out


def fmt_h(h: float) -> str:
    sign = "+" if h >= 0 else "-"
    h = abs(h)
    return f"{sign}{h:.0f}h" if h < 48 else f"{sign}{h / 24:.1f}d"


def explain_shipment(model, df: pd.DataFrame, shipment_id: str, top_k: int = 3) -> str:
    row_df = df[df["shipment_id"] == shipment_id]
    row = row_df.iloc[0]
    pred = model.predict(row_df).iloc[0]
    X, _ = build_features(row_df, model.cp)
    sv, base = shap_frame(model, X)
    sv = sv.iloc[0]
    groups = group_shap(sv.to_frame().T).iloc[0].sort_values(key=np.abs, ascending=False)

    lines = [f"Shipment {shipment_id}  |  {row['origin_port']} -> {row['dest_port']}  |  carrier {row['carrier']}"
             f"  |  cargo {row['cargo_type']}  |  checkpoint: {model.cp}",
             f"  Scheduled delivery : {row['sched_delivery']:%Y-%m-%d %H:%M}",
             f"  Predicted ETA      : {pred['eta']:%Y-%m-%d %H:%M}   (delay {fmt_h(pred['pred_delay_h'])}, "
             f"80% band {fmt_h(pred['p10_h'])} .. {fmt_h(pred['p90_h'])})",
             f"  P(delay > 24h)     : {pred['p_late']:.0%}", ""]

    so_far = delay_so_far(row, model.cp)
    if so_far:
        lines.append(f"  Delay already incurred: {fmt_h(pred['known_delay_h'])}")
        for span, d in so_far:
            lines.append(f"     {span:<38s}{fmt_h(d):>8s}")
        lines.append("")
    lines.append(f"  Expected further delay: {fmt_h(pred['remaining_h'])}  "
                 f"(typical shipment at this stage: {fmt_h(base)})")
    lines.append("  Main drivers vs typical shipment:")
    for g, v in groups.head(top_k).items():
        feats = sv[[f for f in sv.index if group_of(f) == g]].sort_values(key=np.abs, ascending=False)
        detail = ", ".join(f"{FEATURE_TEXT.get(f, f)}={_val(X.iloc[0][f], f)}" for f in feats.index[:2])
        lines.append(f"     {g:<32s}{fmt_h(v):>8s}   [{detail}]")
    return "\n".join(lines)


def _val(v, f):
    if pd.isna(v):
        return "missing"
    if f.endswith("_h"):
        return fmt_h(v).lstrip("+")
    if f in ("missed_berth_window", "missed_appointment", "customs_inspection", "docs_complete", "peak_season",
             "rolled_over_obs", "suez_route", "panama_route", "proj_release_weekend", "sched_release_weekend",
             "arrival_imputed"):
        return "yes" if v >= 0.5 else "no"
    if float(v).is_integer():
        return f"{v:.0f}"
    return f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"


def validate_attributions(model, obs: pd.DataFrame, gt: pd.DataFrame, min_remaining_h: float = 24) -> dict:
    """Does the top explained cause match the stage that truly generated the remaining delay?

    Only possible with synthetic data: compare the SHAP group with the largest positive
    contribution against the ground-truth stage increment that was largest, among
    downstream stages (port / customs / inland), for shipments with meaningful delay ahead.
    """
    X, known = build_features(obs, model.cp)
    sv, _ = shap_frame(model, X)
    g = group_shap(sv)[list(GROUP_TO_INCREMENTS)]
    true = pd.DataFrame({grp: gt[incs].sum(axis=1) for grp, incs in GROUP_TO_INCREMENTS.items()})
    true = true - true.mean()  # SHAP is relative to the average shipment, so compare deviations
    remaining = gt["final_delay_h"] - known
    mask = remaining > min_remaining_h
    pred_top, true_top = g[mask].idxmax(axis=1), true[mask].idxmax(axis=1)
    top2 = g[mask].apply(lambda r: set(r.nlargest(2).index), axis=1)
    return dict(n=int(mask.sum()), top1_agreement=float((pred_top == true_top).mean()),
                top2_agreement=float(np.mean([t in s for t, s in zip(true_top, top2)])),
                chance_top1=float(1 / len(GROUP_TO_INCREMENTS)))
