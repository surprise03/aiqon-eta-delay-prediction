# Shipment ETA & Delay Propagation

Predicts the final ETA of container shipments, the probability and size of a delay, and **explains which stage
and cause drive it**. It also models how delays are absorbed or amplified as they move through the shipment's stages.

Aiqon AI Internship take-home assignment. Full write-up: [`report/Technical_Report.pdf`](report/Technical_Report.pdf).

## Key ideas

| Idea | Why it matters |
|---|---|
| **Causal simulator** with realistic mechanisms (rollovers, Suez/Panama disruption, weekly port congestion, berth windows, weekend customs, truck appointments) | Delays *propagate* through the stages; they aren't noise added to the final date. Ground truth for each stage is known, so the explanations can be checked. |
| **ETA re-estimated at 4 checkpoints** (booking → departure → port arrival → customs release) | Matches how ETA is actually updated in operations. Each checkpoint uses only the information available at that moment, so nothing leaks from the future. |
| **Predict *remaining* delay**: `ETA = scheduled + delay so far + predicted remaining` | The model has to learn when delay already incurred is absorbed and when it is amplified. |
| **Interaction features**: lateness × congestion, missed berth window, clearance hitting a weekend, truck-appointment slack | Captures how stages interact instead of treating each stage on its own. |
| **Two-layer explanations**: an exact breakdown of the delay so far, plus SHAP drivers of the delay to come | Shows what has happened and why more delay is expected. |
| **Explanations validated against ground truth** | The top explained cause matches the true cause 71% of the time, against 33% by chance. |
| **Conformal 80% ETA bands** and P(late > 24h) | Honest uncertainty, especially at booking. |

## Results (300 shipments, 5-fold CV ×3; MAE of final delay in hours)

| Checkpoint | Schedule | Carry-forward | Ridge | **GBM (ours)** | Hold-out (60 unseen) | AUC |
|---|---|---|---|---|---|---|
| Booking | 63.0 | 63.0 | 64.3 | **56.9** | 51.1 | 0.74 |
| Departed | 63.0 | 48.4 | 52.0 | **44.2** | 44.2 | 0.78 |
| Port arrival | 63.0 | 44.6 | 31.8 | **29.2** | 29.8 | 0.85 |
| Customs released | 63.0 | 10.7 | 8.8 | **5.6** | 5.9 | 0.99 |

**Propagation finding:** a vessel arriving more than 24h late adds a median **+108h** of further delay at a
highly congested port, compared with **+34h** at a lightly congested one. Small delays (0–24h) at low
congestion are mostly absorbed.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python run_pipeline.py                 # simulate -> evaluate -> train -> explain -> figures (~2 min; --fast for ~40s)
python report/build_report.py --author "Your Name"   # rebuild the PDF report from the outputs
```

### Demo (CLI)

```bash
python predict.py --list                                        # some unseen "live" shipment ids
python predict.py --id SHP0323                                  # ETA timeline across all checkpoints
python predict.py --id SHP0323 --checkpoint C2_port_arrival     # full explanation at one stage
python predict.py --csv my_shipments.csv --checkpoint C0_booking  # score new shipments (same schema)
```

Example output (unseen shipment, after port arrival):

```
Shipment SHP0323  |  Singapore -> Hamburg  |  carrier Meridian  |  cargo perishable  |  checkpoint: C2_port_arrival
  Scheduled delivery : 2025-10-21 11:12
  Predicted ETA      : 2025-11-04 11:54   (delay +14.0d, 80% band +10.2d .. +17.5d)
  P(delay > 24h)     : 99%

  Delay already incurred: +10.3d
     origin (until departure)                   +4h
     departure -> port arrival               +10.1d

  Expected further delay: +3.8d  (typical shipment at this stage: +33h)
  Main drivers vs typical shipment:
     Destination port congestion         +36h   [lateness x congestion score=193, port congestion on arrival=0.78]
     Upstream delay (propagation)        +12h   [vessel arrival delay=10.3d, early arrival (absorbed at anchor)=0h]
     Customs & documentation             +11h   [documentation completeness=no, working hours before weekend=62]

  ACTUAL: delivered 2025-11-08 18:24, delay 439h
```

## Repository layout

```
src/
  config.py        lanes, ports, carriers, cargo types, planned stage durations (all assumptions in one place)
  simulate.py      causal simulator + observed view with missing events
  features.py      checkpoint-aware features (no future leakage), cause groups
  model.py         GBM / ridge / baselines, conformal intervals, delay classifier, repeated-CV evaluation
  explain.py       exact delay-so-far decomposition, grouped SHAP, explanation validation
  propagation.py   absorption/amplification analysis, model partial-dependence curves
run_pipeline.py    end-to-end run, writes data/ and outputs/
predict.py         demo CLI
report/            build_report.py -> Technical_Report.pdf
data/              shipments.csv (observed, with gaps), *_ground_truth.csv (true timestamps + causes),
                   shipments_live*.csv (60 hold-out shipments)
outputs/           metrics CSVs, summary.json, example_explanations.txt, figures/
```

## Data dictionary (`data/shipments.csv`)

| Column(s) | Meaning |
|---|---|
| `route`, `origin_port`, `dest_port`, `chokepoint` | Lane (6 real-world lanes) and canal exposure |
| `carrier`, `carrier_reliability` | Carrier and its published schedule reliability |
| `cargo_type`, `docs_complete`, `containers` | Cargo risk class, documentation completeness, volume |
| `origin_congestion`, `congestion_forecast`, `weather_forecast` | Information known at booking (forecasts are noisy) |
| `congestion_at_arrival` | Destination port congestion in the week the vessel actually arrives |
| `sched_*` / `act_*` | Scheduled / actual timestamps for each milestone (`act_*` may be missing) |
| `truck_appointment` | Booked truck pickup slot |
| `customs_inspection` | Physical inspection (known after customs) |
| `final_delay_h`, `is_delayed` | Target: delivery delay in hours; delayed if > 24h |

The ground-truth file also contains the hidden causes (`rolled_over`, `chokepoint_disruption`,
`missed_berth_window`, `truck_rebooked`) and the exact stage increments `inc_*`, which sum to `final_delay_h`.

## Assumptions

- The schedule is fixed at booking. Planned durations come from `src/config.py`, and ocean schedules include about 4% padding.
- Congestion is a weekly per-port index (an AR(1) process plus a peak-season bump) that a port or AIS feed could supply.
- Customs works on weekdays only. Each shipment has one truck appointment, 12h after the planned customs release.
- Forecasts available at booking are the true values plus noise. Hidden causes are never used as model inputs.
- The data is synthetic, so the absolute errors illustrate the method rather than claiming real-world accuracy.
