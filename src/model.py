"""Models, baselines and cross-validated evaluation per checkpoint."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, f1_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import LATE_THRESHOLD_H
from .features import CHECKPOINTS, build_features

GB = dict(max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=12, l2_regularization=1.0,
          random_state=0)


def make_regressor(loss: str = "absolute_error", quantile: float | None = None):
    if quantile is not None:
        return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **GB)
    return HistGradientBoostingRegressor(loss=loss, **GB)


class ConformalInterval:
    """Conformalised quantile regression (CQR): P10/P90 models fitted on 75% of the
    training data, then widened on the remaining 25% so the band reaches ~80% coverage."""

    def __init__(self, alpha: float = 0.2, seed: int = 0):
        self.alpha, self.seed = alpha, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(y))
        cut = int(0.75 * len(y))
        fit_i, cal_i = idx[:cut], idx[cut:]
        self.lo = make_regressor(quantile=self.alpha / 2).fit(X.iloc[fit_i], y[fit_i])
        self.hi = make_regressor(quantile=1 - self.alpha / 2).fit(X.iloc[fit_i], y[fit_i])
        Xc, yc = X.iloc[cal_i], y[cal_i]
        scores = np.maximum(self.lo.predict(Xc) - yc, yc - self.hi.predict(Xc))
        n = len(cal_i)
        self.q = float(np.quantile(scores, min(1.0, (1 - self.alpha) * (n + 1) / n)))
        return self

    def predict(self, X):
        return self.lo.predict(X) - self.q, self.hi.predict(X) + self.q


def make_classifier():
    return HistGradientBoostingClassifier(**GB)


def make_ridge():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=3.0))


def evaluate(df: pd.DataFrame, n_splits: int = 5, n_repeats: int = 3, seed: int = 0,
             checkpoints: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Repeated K-fold CV. Returns a metrics table and out-of-fold predictions (first repeat)."""
    y_final = df["final_delay_h"].to_numpy()
    y_late = (y_final > LATE_THRESHOLD_H).astype(int)
    rows, oof = [], {}

    for cp in checkpoints or CHECKPOINTS:
        X, known = build_features(df, cp)
        known = known.to_numpy()
        y_rem = y_final - known  # remaining delay still to come
        acc = {m: [] for m in ["schedule", "carry_forward", "carry_plus_route_mean", "ridge", "gbm"]}
        cls_auc, cls_brier, cls_f1, cover, width = [], [], [], [], []
        oof_pred = np.full(len(df), np.nan)
        oof_lo, oof_hi, oof_p = (np.full(len(df), np.nan) for _ in range(3))

        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        for fold, (tr, te) in enumerate(rkf.split(X)):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            # baselines
            acc["schedule"].append(mean_absolute_error(y_final[te], np.zeros(len(te))))
            acc["carry_forward"].append(mean_absolute_error(y_final[te], known[te]))
            route_mean = pd.Series(y_rem[tr]).groupby(df["route"].iloc[tr].to_numpy()).median()
            rm = df["route"].iloc[te].map(route_mean).fillna(np.median(y_rem[tr])).to_numpy()
            acc["carry_plus_route_mean"].append(mean_absolute_error(y_final[te], known[te] + rm))
            # models (predict remaining delay)
            ridge = make_ridge().fit(Xtr, y_rem[tr])
            acc["ridge"].append(mean_absolute_error(y_final[te], known[te] + ridge.predict(Xte)))
            gbm = make_regressor().fit(Xtr, y_rem[tr])
            pred = known[te] + gbm.predict(Xte)
            acc["gbm"].append(mean_absolute_error(y_final[te], pred))
            # uncertainty: P10 / P90 quantile models
            ci_lo, ci_hi = ConformalInterval(seed=fold).fit(Xtr, y_rem[tr]).predict(Xte)
            lo, hi = known[te] + ci_lo, known[te] + ci_hi
            lo, hi = np.minimum(lo, pred), np.maximum(hi, pred)
            cover.append(np.mean((y_final[te] >= lo) & (y_final[te] <= hi)))
            width.append(np.mean(hi - lo))
            # delay probability
            clf = make_classifier().fit(Xtr, y_late[tr])
            p = clf.predict_proba(Xte)[:, 1]
            cls_auc.append(roc_auc_score(y_late[te], p))
            cls_brier.append(brier_score_loss(y_late[te], p))
            cls_f1.append(f1_score(y_late[te], p > 0.5))
            if fold < n_splits:  # first repeat -> one clean OOF prediction per shipment
                oof_pred[te], oof_lo[te], oof_hi[te], oof_p[te] = pred, lo, hi, p

        row = {"checkpoint": cp}
        for m, v in acc.items():
            row[f"mae_{m}"] = float(np.mean(v))
            row[f"mae_{m}_sd"] = float(np.std(v))
        row.update(auc=float(np.mean(cls_auc)), brier=float(np.mean(cls_brier)), f1=float(np.mean(cls_f1)),
                   p10_p90_coverage=float(np.mean(cover)), interval_width_h=float(np.mean(width)))
        rows.append(row)
        oof[cp] = pd.DataFrame({"pred": oof_pred, "lo": oof_lo, "hi": oof_hi, "p_late": oof_p,
                                "known": known, "actual": y_final}, index=df.index)
    return pd.DataFrame(rows), oof


class CheckpointModel:
    """Final models trained on all data for one checkpoint (used for explanations / demo)."""

    def __init__(self, checkpoint: str):
        self.cp = checkpoint

    def fit(self, df: pd.DataFrame):
        X, known = build_features(df, self.cp)
        y_final = df["final_delay_h"].to_numpy()
        y_rem = y_final - known.to_numpy()
        self.columns = list(X.columns)
        self.reg = make_regressor().fit(X, y_rem)
        self.interval = ConformalInterval().fit(X, y_rem)
        self.clf = make_classifier().fit(X, (y_final > LATE_THRESHOLD_H).astype(int))
        self.X_background = X
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X, known = build_features(df, self.cp)
        rem = self.reg.predict(X)
        pred = known + rem
        ci_lo, ci_hi = self.interval.predict(X)
        lo = np.minimum(known + ci_lo, pred)
        hi = np.maximum(known + ci_hi, pred)
        out = pd.DataFrame({"known_delay_h": known, "remaining_h": rem, "pred_delay_h": pred,
                            "p10_h": lo, "p90_h": hi, "p_late": self.clf.predict_proba(X)[:, 1]}, index=df.index)
        out["eta"] = df["sched_delivery"] + pd.to_timedelta(out["pred_delay_h"], unit="h")
        return out
