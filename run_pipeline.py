"""End-to-end pipeline: simulate -> evaluate -> train -> explain -> analyse -> figures.

Usage:  python run_pipeline.py [--n 300] [--seed 42] [--fast]
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.explain import explain_shipment, group_shap, shap_frame, validate_attributions  # noqa: E402
from src.features import CHECKPOINT_LABEL, CHECKPOINTS, GROUPS, build_features  # noqa: E402
from src.model import CheckpointModel, evaluate  # noqa: E402
from src.propagation import model_implied_curve, propagation_factors, stage_contribution  # noqa: E402
from src.simulate import observed_view, simulate  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
DATA, OUT, FIG = ROOT / "data", ROOT / "outputs", ROOT / "outputs" / "figures"
N_LIVE = 60
COLORS = {"schedule": "#9aa0a6", "carry_forward": "#f2a65a", "carry_plus_route_mean": "#c77dff",
          "ridge": "#4ea8de", "gbm": "#1b4965"}
NAMES = {"schedule": "Schedule (no delay)", "carry_forward": "Carry-forward current delay",
         "carry_plus_route_mean": "Carry-forward + route median", "ridge": "Ridge (linear)",
         "gbm": "Gradient boosting (ours)"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def main(n: int, seed: int, fast: bool):
    for d in (DATA, OUT, FIG):
        d.mkdir(parents=True, exist_ok=True)

    # 1. data -----------------------------------------------------------------
    # one simulated world; the last N_LIVE shipments are never used for training and act
    # as unseen "live" shipments for the hold-out test and the demo
    world = simulate(n=n + N_LIVE, seed=seed)
    world_obs = observed_view(world)
    gt, obs = world.iloc[:n].reset_index(drop=True), world_obs.iloc[:n].reset_index(drop=True)
    gt_live, live = world.iloc[n:].reset_index(drop=True), world_obs.iloc[n:].reset_index(drop=True)
    gt.to_csv(DATA / "shipments_ground_truth.csv", index=False)
    obs.to_csv(DATA / "shipments.csv", index=False)
    gt_live.to_csv(DATA / "shipments_live_ground_truth.csv", index=False)
    live.to_csv(DATA / "shipments_live.csv", index=False)
    print(f"[data] {n} shipments | delayed(>24h): {gt.is_delayed.mean():.0%} | "
          f"median delay {gt.final_delay_h.median():.1f}h")

    # 2. evaluation -------------------------------------------------------------
    metrics, oof = evaluate(obs, n_repeats=1 if fast else 3)
    metrics.to_csv(OUT / "metrics_by_checkpoint.csv", index=False)
    print("[eval]\n" + metrics[["checkpoint", "mae_schedule", "mae_carry_forward", "mae_carry_plus_route_mean",
                                "mae_ridge", "mae_gbm", "auc", "p10_p90_coverage"]].round(2).to_string(index=False))

    # 3. final models -------------------------------------------------------------
    models = {cp: CheckpointModel(cp).fit(obs) for cp in CHECKPOINTS}
    with open(OUT / "models.pkl", "wb") as f:
        pickle.dump(models, f)
    holdout = []
    for cp, m in models.items():
        p = m.predict(live)
        y = gt_live["final_delay_h"]
        holdout.append(dict(checkpoint=cp, mae_gbm=float((p.pred_delay_h - y).abs().mean()),
                            mae_carry_forward=float((p.known_delay_h - y).abs().mean()),
                            coverage=float(((y >= p.p10_h) & (y <= p.p90_h)).mean())))
    holdout = pd.DataFrame(holdout)
    holdout.to_csv(OUT / "holdout_metrics.csv", index=False)
    print(f"[holdout] {N_LIVE} unseen shipments\n" + holdout.round(2).to_string(index=False))

    # 4. explanations -------------------------------------------------------------
    importance = {}
    for cp, m in models.items():
        X, _ = build_features(obs, cp)
        sv, _ = shap_frame(m, X)
        importance[cp] = group_shap(sv).abs().mean()
    imp = pd.DataFrame(importance).reindex(list(GROUPS)).fillna(0)
    imp.to_csv(OUT / "driver_importance_by_checkpoint.csv")
    attribution = validate_attributions(models["C2_port_arrival"], obs, gt)
    print(f"[explain] attribution check @C2: {attribution}")

    # 5. propagation ----------------------------------------------------------------
    prop = propagation_factors(obs)
    prop.to_csv(OUT / "propagation_factors.csv", index=False)
    print("[propagation]\n" + prop.to_string(index=False))
    contrib = stage_contribution(gt)
    contrib.to_csv(OUT / "stage_contribution_by_route.csv")

    # 6. robustness to missing events ---------------------------------------------------
    robust = []
    for scale in [0.0, 1.0, 2.0, 3.0]:
        o = observed_view(gt, scale=scale)
        r, _ = evaluate(o, n_repeats=1, checkpoints=["C2_port_arrival", "C3_customs_release"])
        for _, row in r.iterrows():
            robust.append(dict(missing_scale=scale, checkpoint=row.checkpoint, mae_gbm=row.mae_gbm,
                               mae_carry_forward=row.mae_carry_forward))
    robust = pd.DataFrame(robust)
    robust.to_csv(OUT / "missing_data_robustness.csv", index=False)
    print("[robustness]\n" + robust.round(2).to_string(index=False))

    # 7. example explanations ------------------------------------------------------------
    ex_ids = pick_examples(gt_live)  # unseen shipments
    with open(OUT / "example_explanations.txt", "w", encoding="utf-8") as f:
        for sid in ex_ids:
            for cp in ["C0_booking", "C2_port_arrival", "C3_customs_release"]:
                f.write(explain_shipment(models[cp], live, sid) + "\n\n")
            act = gt_live.loc[gt_live.shipment_id == sid].iloc[0]
            f.write(f"  ACTUAL: delivered {act.act_delivery:%Y-%m-%d %H:%M}, delay {act.final_delay_h:.0f}h\n")
            f.write("=" * 100 + "\n\n")

    # 8. figures -----------------------------------------------------------------------
    fig_mae(metrics)
    fig_propagation(obs, models["C2_port_arrival"])
    fig_contrib(contrib)
    fig_drivers(imp)
    fig_eta_evolution(live, gt_live, models, ex_ids[0])

    summary = dict(n=n, seed=seed, delayed_share=float(gt.is_delayed.mean()),
                   median_delay_h=float(gt.final_delay_h.median()),
                   metrics=metrics.round(3).to_dict(orient="records"), attribution_check=attribution,
                   holdout=holdout.round(3).to_dict(orient="records"),
                   propagation=prop.to_dict(orient="records"), examples=ex_ids)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[done] outputs written to {OUT}")


def pick_examples(gt: pd.DataFrame) -> list[str]:
    """One amplified-at-port case, one customs-inspection case, one on-time case."""
    a = gt[(gt.missed_berth_window == 1) & (gt.congestion_at_arrival > 0.6) & (gt.rolled_over == 0)]
    b = gt[(gt.customs_inspection == 1) & (gt.missed_berth_window == 0)]
    c = gt[gt.final_delay_h.abs() < 12]
    return [x.sort_values("final_delay_h").iloc[len(x) // 2].shipment_id for x in (a, b, c) if len(x)]


def fig_mae(m: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(len(m))
    for j, k in enumerate(["schedule", "carry_forward", "carry_plus_route_mean", "ridge", "gbm"]):
        ax.errorbar(x + (j - 2) * 0.04,m[f"mae_{k}"], yerr=m[f"mae_{k}_sd"], marker="o", ms=4, capsize=2, lw=2.2 if k == "gbm" else 1.2,
                    color=COLORS[k], label=NAMES[k])
    ax.set_xticks(x, [CHECKPOINT_LABEL[c] for c in m.checkpoint])
    ax.set_ylabel("MAE of final delay (hours)")
    ax.set_title("ETA error as the shipment progresses (5-fold CV x3)")
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(FIG / "fig1_mae_by_checkpoint.png")
    plt.close(fig)


def fig_propagation(obs, model):
    X, _ = build_features(obs, "C2_port_arrival")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    c = obs["congestion_at_arrival"]
    sc = axes[0].scatter(X["arrival_delay_h"].clip(-40, 250), obs["final_delay_h"].clip(-40, 400), c=c, cmap="viridis",
                         s=12, alpha=0.8)
    axes[0].plot([-40, 250], [-40, 250], ls="--", c="grey", lw=1)
    axes[0].text(170, 60, "1:1 line", fontsize=7, color="grey")
    axes[0].set_xlabel("Vessel arrival delay (h)")
    axes[0].set_ylabel("Final delivery delay (h)")
    axes[0].set_title("Observed: arrival vs final delay")
    fig.colorbar(sc, ax=axes[0], label="port congestion")
    grid = np.linspace(-24, 120, 37)
    for cong, col in [(0.35, "#4ea8de"), (0.6, "#f2a65a"), (0.85, "#d62828")]:
        axes[1].plot(grid, model_implied_curve(model, obs, cong, grid), c=col, lw=2, label=f"congestion {cong}")
    axes[1].plot(grid, grid, ls="--", c="grey", lw=1)
    axes[1].axvline(24, c="k", lw=0.6, ls=":")
    axes[1].text(26, axes[1].get_ylim()[0] + 5, "berth window\nmissed", fontsize=7)
    axes[1].set_xlabel("Vessel arrival delay (h)")
    axes[1].set_ylabel("Predicted final delay (h)")
    axes[1].set_title("Learned propagation (model, partial dependence)")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_propagation.png")
    plt.close(fig)


def fig_contrib(contrib):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    cols = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#8d99ae"]
    left_pos = np.zeros(len(contrib))
    left_neg = np.zeros(len(contrib))
    for col, color in zip(contrib.columns, cols):
        v = contrib[col].to_numpy()
        left = np.where(v >= 0, left_pos, left_neg)
        ax.barh(contrib.index, v, left=left, color=color, label=col)
        left_pos += np.where(v >= 0, v, 0)
        left_neg += np.where(v < 0, v, 0)
    ax.axvline(0, c="k", lw=0.6)
    ax.set_xlabel("Mean delay contributed (h), delayed shipments only")
    ax.set_title("Where delays originate, by route (ground-truth stage increments)")
    ax.legend(frameon=False, fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(FIG / "fig3_stage_contribution.png")
    plt.close(fig)


def fig_drivers(imp):
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    im = ax.imshow(imp.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(imp.shape[1]), [CHECKPOINT_LABEL[c].replace(" ", "\n", 1) for c in imp.columns])
    ax.set_yticks(range(imp.shape[0]), imp.index)
    for i in range(imp.shape[0]):
        for j in range(imp.shape[1]):
            v = imp.iat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > imp.to_numpy().max() * 0.6 else "black")
    ax.set_title("Mean |SHAP| (hours of remaining delay) per cause group")
    fig.colorbar(im, ax=ax, label="hours")
    fig.savefig(FIG / "fig4_drivers_by_checkpoint.png")
    plt.close(fig)


def fig_eta_evolution(obs, gt, models, sid):
    row = obs[obs.shipment_id == sid]
    preds = pd.concat([models[cp].predict(row).assign(cp=cp) for cp in CHECKPOINTS])
    actual = gt.loc[gt.shipment_id == sid, "final_delay_h"].iloc[0]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    x = np.arange(len(preds))
    ax.fill_between(x, preds.p10_h, preds.p90_h, color="#4ea8de", alpha=0.25, label="80% band")
    ax.plot(x, preds.pred_delay_h, marker="o", c="#1b4965", label="predicted delay")
    ax.plot(x, preds.known_delay_h, marker="s", ls=":", c="#f2a65a", label="delay known so far")
    ax.axhline(actual, c="#d62828", ls="--", label=f"actual ({actual:.0f}h)")
    for i, p in enumerate(preds.p_late):
        ax.text(i, preds.pred_delay_h.iloc[i] + 4, f"P(late)={p:.0%}", ha="center", fontsize=7)
    ax.set_xticks(x, [CHECKPOINT_LABEL[c] for c in CHECKPOINTS])
    ax.set_ylabel("Delay vs schedule (h)")
    ax.set_title(f"ETA updates for {sid} ({row.origin_port.iloc[0]} -> {row.dest_port.iloc[0]})")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIG / "fig5_eta_evolution.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="1 CV repeat instead of 3")
    a = ap.parse_args()
    main(a.n, a.seed, a.fast)
