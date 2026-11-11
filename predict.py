"""Demo CLI: explain the ETA of a shipment.

Examples
  python predict.py --list                         # show some shipment ids
  python predict.py --id SHP0244                   # ETA timeline across all checkpoints
  python predict.py --id SHP0244 --checkpoint C2_port_arrival   # full explanation at one checkpoint
  python predict.py --csv my_shipments.csv --checkpoint C2_port_arrival  # score new shipments
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import pandas as pd

from src.explain import explain_shipment, fmt_h
from src.features import CHECKPOINT_LABEL, CHECKPOINTS

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
DATE_COLS = ["sched_departure", "sched_port_arrival", "sched_berth", "sched_discharge", "sched_customs_release",
             "truck_appointment", "sched_delivery", "act_departure", "act_port_arrival", "act_berth",
             "act_discharge", "act_customs_release", "act_delivery"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in DATE_COLS:
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id")
    ap.add_argument("--checkpoint", choices=CHECKPOINTS)
    ap.add_argument("--csv", type=Path, help="score shipments from a CSV with the same schema as data/shipments.csv")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    models_path = ROOT / "outputs" / "models.pkl"
    if not models_path.exists():
        raise SystemExit("Models not found - run `python run_pipeline.py` first.")
    models = pickle.loads(models_path.read_bytes())
    df = load(a.csv or ROOT / "data" / "shipments_live.csv")  # unseen shipments by default
    gt_path = ROOT / "data" / "shipments_live_ground_truth.csv"
    gt = load(gt_path) if gt_path.exists() and not a.csv else None

    if a.list:
        cols = ["shipment_id", "route", "carrier", "cargo_type", "sched_delivery"]
        print(df[cols].sample(15, random_state=1).to_string(index=False))
        return

    if a.csv and not a.id:
        cp = a.checkpoint or "C0_booking"
        out = models[cp].predict(df)
        out["eta"] = out["eta"].dt.round("min")
        print(pd.concat([df[["shipment_id", "sched_delivery"]], out[["eta", "pred_delay_h", "p10_h", "p90_h", "p_late"]]],
                        axis=1).round(1).to_string(index=False))
        return

    if not a.id:
        ap.error("--id, --list or --csv is required")
    if a.checkpoint:
        print(explain_shipment(models[a.checkpoint], df, a.id))
    else:  # ETA timeline
        row = df[df.shipment_id == a.id]
        if row.empty:
            raise SystemExit(f"Unknown shipment {a.id}")
        print(f"ETA timeline for {a.id} ({row.origin_port.iloc[0]} -> {row.dest_port.iloc[0]}), "
              f"scheduled {row.sched_delivery.iloc[0]:%Y-%m-%d %H:%M}\n")
        print(f"  {'checkpoint':<24s}{'ETA':<18s}{'delay':>8s}   {'80% band':<18s}{'P(late)':>8s}")
        for cp in CHECKPOINTS:
            p = models[cp].predict(row).iloc[0]
            print(f"  {CHECKPOINT_LABEL[cp]:<24s}{p.eta:%Y-%m-%d %H:%M}  {fmt_h(p.pred_delay_h):>8s}   "
                  f"{fmt_h(p.p10_h) + ' .. ' + fmt_h(p.p90_h):<18s}{p.p_late:>8.0%}")
        if gt is not None:
            act = gt[gt.shipment_id == a.id].iloc[0]
            print(f"\n  {'ACTUAL':<24s}{act.act_delivery:%Y-%m-%d %H:%M}  {fmt_h(act.final_delay_h):>8s}")
        print("\nUse --checkpoint <name> for the full explanation at a stage.")


if __name__ == "__main__":
    main()
