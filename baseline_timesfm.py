"""TimesFM baseline for next-recurring-merchant-family prediction.

Pipeline
1. Tag subscription-like card payments with a merchant family (keyword rules + MCC).
2. Build a weekly transaction-count series per (client, family).
3. Forecast the 13 weeks after the cutoff with Google's TimesFM (zero-shot):
   - 2.5: univariate, one series per (client, family)
   - 3.0: univariate, or multivariate with the 7 family series of a client forecast jointly
4a. "rule": pick the family with the most forecast mass; `none` below a threshold tuned on train.
4b. "gbm": TimesFM forecasts + simple recurrence stats -> gradient-boosted classifier.

Every run is logged to TensorBoard under runs/ (view with: .venv/bin/tensorboard --logdir runs).

Usage:
  .venv/bin/python baseline_timesfm.py --model 2.5
  .venv/bin/python baseline_timesfm.py --model 3.0 --mode multi
"""

import argparse
import io
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
RUNS = ROOT / "runs"
CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
N_WEEKS = 60
HORIZON_WEEKS = 13  # ~90 days
FAMILIES = ["cloud", "gym", "insurance", "mobile", "music", "software", "streaming"]
LABELS = FAMILIES + ["none"]

MCC_FAMILY = {"4814": "mobile", "5732": "cloud", "5734": "software", "6300": "insurance", "7997": "gym"}
KEYWORDS = [  # first match wins
    (r"audio|member pass|music", "music"),
    (r"media stream|video", "streaming"),
    (r"phone|service bill", "mobile"),
    (r"cloud|storage|service plan", "cloud"),
    (r"saas|software|productivity|prod suite", "software"),
    (r"cover|policy|insurance", "insurance"),
    (r"gym|fit", "gym"),
]
# Generic descriptions that appear mostly as one-off decoys (see EDA).
GENERIC = re.compile(r"^(member plan|monthly plan|digital service|subscription charge|merchant charge|"
                     r"service payment|card purchase|digital order)$")


def load(split):
    tx = pd.read_json(DATA / f"{split}_transactions.jsonl", lines=True, dtype={"mcc": str})
    tx["timestamp"] = pd.to_datetime(tx["timestamp"], utc=True)
    return tx


def tag_family(tx):
    """Assign a family to subscription-like payments; ambiguous 5812 ones resolved by amount."""
    tx = tx[(tx["type"] == "card_payment") & tx["direction"].eq("out")].copy()
    desc = tx["description"].str.lower()
    fam = pd.Series(None, index=tx.index, dtype=object)
    for pat, f in KEYWORDS:
        fam = fam.where(fam.notna(), np.where(desc.str.contains(pat), f, None))
    fam = fam.where(fam.notna(), tx["mcc"].map(MCC_FAMILY))
    subscription_mcc = tx["mcc"].isin(list(MCC_FAMILY) + ["5812"])
    ambiguous = fam.isna() & subscription_mcc & desc.str.contains("digital plus|premium plan")
    fam[ambiguous] = "amb"
    tx["family"] = fam
    tx["generic"] = desc.str.match(GENERIC)
    tx = tx[tx["family"].notna() & (subscription_mcc | tx["family"].isin(FAMILIES))]

    # Resolve music-vs-streaming ambiguity: nearest-amount known stream of the same client.
    known = tx[tx["family"].isin(["music", "streaming"])].groupby(["client_id", "family"])["amount"].median()
    def resolve(row):
        cands = known.get(row.client_id)
        if cands is None or len(cands) == 0:
            return "streaming"
        return (cands - row.amount).abs().idxmin()
    amb = tx["family"] == "amb"
    tx.loc[amb, "family"] = tx[amb].apply(resolve, axis=1) if amb.any() else []
    return tx


def weekly_series(sub, clients):
    """Array [n_clients, n_families, N_WEEKS] of weekly counts ending at the cutoff (non-generic txns only)."""
    start = CUTOFF - pd.Timedelta(weeks=N_WEEKS)
    s = sub[~sub["generic"] & (sub["timestamp"] >= start)].copy()
    s["week"] = ((s["timestamp"] - start).dt.days // 7).clip(0, N_WEEKS - 1)
    c_idx = pd.Index(clients).get_indexer(s["client_id"])
    f_idx = pd.Index(FAMILIES).get_indexer(s["family"])
    arr = np.zeros((len(clients), len(FAMILIES), N_WEEKS), dtype=np.float32)
    np.add.at(arr, (c_idx, f_idx, s["week"].to_numpy()), 1)
    return arr


class Forecaster:
    """Returns (mean [C,F,H], q90 [C,F,H]) for a [C,F,T] array, logging progress to TensorBoard."""

    def __init__(self, model, mode, device, writer):
        self.model_name, self.mode, self.writer = model, mode, writer
        self.step = 0
        if model == "2.5":
            import timesfm
            torch.set_float32_matmul_precision("high")
            self.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
            self.m.compile(timesfm.ForecastConfig(
                max_context=64, max_horizon=16, normalize_inputs=True, use_continuous_quantile_head=True,
                infer_is_positive=True, fix_quantile_crossing=True, per_core_batch_size=256))
        else:
            from timesfm3 import TimesFM3Forecaster
            self.m = TimesFM3Forecaster.from_pretrained(device=device, per_core_batch_size=128)

    def _run_chunk(self, chunk):
        """chunk: list of arrays, each [T] (univariate) or [F, T] (multivariate)."""
        if self.model_name == "2.5":
            point, quant = self.m.forecast(horizon=HORIZON_WEEKS, inputs=chunk)
            return np.clip(point, 0, None), np.clip(quant[..., -1], 0, None)
        outs = list(self.m.predict_batch(chunk, horizon=HORIZON_WEEKS, return_quantiles=True, make_positive=True))
        q = np.stack([o.quantiles for o in outs])  # [..., H, Q]
        return np.clip(q.mean(-1), 0, None), np.clip(q[..., -1], 0, None)

    def __call__(self, arr, split, chunk_size=1024):
        C, F, T = arr.shape
        items = list(arr) if self.mode == "multi" else list(arr.reshape(C * F, T))
        if self.mode == "multi":
            chunk_size = max(1, chunk_size // F)
        means, q90s, t0 = [], [], time.time()
        for i in range(0, len(items), chunk_size):
            m, q = self._run_chunk(items[i:i + chunk_size])
            means.append(m); q90s.append(q)
            done = min(i + chunk_size, len(items))
            self.step += 1
            self.writer.add_scalar(f"progress/{split}_frac_done", done / len(items), self.step)
            self.writer.add_scalar("progress/series_per_sec", done / (time.time() - t0), self.step)
            print(f"  [{split}] {done}/{len(items)} ({done / (time.time() - t0):.0f}/s)", flush=True)
        mean, q90 = np.concatenate(means), np.concatenate(q90s)
        return mean.reshape(C, F, HORIZON_WEEKS), q90.reshape(C, F, HORIZON_WEEKS)


def features(sub, clients, mean, q90):
    rows = []
    by_client = {c: g for c, g in sub[~sub["generic"]].groupby("client_id")}
    empty = sub.iloc[:0]
    for ci, c in enumerate(clients):
        s_c = by_client.get(c, empty)
        row = {"client_id": c}
        for fi, f in enumerate(FAMILIES):
            s = s_c[s_c["family"] == f].sort_values("timestamp")
            p = mean[ci, fi]
            row[f"{f}_fc_sum"] = p.sum()
            row[f"{f}_fc_4w"] = p[:4].sum()
            row[f"{f}_fc_8w"] = p[:8].sum()
            row[f"{f}_fc_q90_4w"] = q90[ci, fi, :4].sum()
            row[f"{f}_fc_first_week"] = np.argmax(np.cumsum(p) >= 0.5) if p.sum() >= 0.5 else HORIZON_WEEKS
            n = len(s)
            row[f"{f}_n"] = n
            row[f"{f}_n_90d"] = (s["timestamp"] >= CUTOFF - pd.Timedelta(days=90)).sum()
            gaps = s["timestamp"].diff().dt.days.dropna()
            days_since = (CUTOFF - s["timestamp"].iloc[-1]).days if n else np.nan
            med_gap = gaps.median() if len(gaps) else np.nan
            row[f"{f}_days_since"] = days_since
            row[f"{f}_med_gap"] = med_gap
            row[f"{f}_gap_std"] = gaps.std() if len(gaps) > 1 else np.nan
            row[f"{f}_next_in"] = med_gap - days_since
            row[f"{f}_amt_cv"] = s["amount"].std() / s["amount"].mean() if n > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("client_id")


def rule_predict(X, col, thr):
    scores = X[[f"{f}_{col}" for f in FAMILIES]].to_numpy()
    return np.where(scores.max(1) >= thr, np.array(FAMILIES)[scores.argmax(1)], "none")


def macro_f1(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


# ---------- TensorBoard helpers ----------

def fig_to_tb(writer, tag, fig, step=0):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = plt.imread(buf)[..., :3]
    writer.add_image(tag, img, step, dataformats="HWC")


def log_eval(writer, name, y, p):
    f1 = macro_f1(y, p)
    writer.add_scalar(f"valid_macro_f1/{name}", f1, 0)
    prec, rec, f1s, _ = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    for lab, a, b, c in zip(LABELS, prec, rec, f1s):
        writer.add_scalar(f"per_class_f1_{name}/{lab}", c, 0)
        writer.add_scalar(f"per_class_recall_{name}/{lab}", b, 0)
        writer.add_scalar(f"per_class_precision_{name}/{lab}", a, 0)
    writer.add_text(f"report/{name}", "```\n" + classification_report(y, p, labels=LABELS, zero_division=0) + "\n```")
    cm = confusion_matrix(y, p, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm / cm.sum(1, keepdims=True).clip(1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right"); ax.set_yticks(range(len(LABELS)), LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="black" if cm[i, j] < cm.max() / 2 else "white")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{name} — valid macro-F1 {f1:.4f}")
    fig_to_tb(writer, f"confusion/{name}", fig)
    return f1


def log_forecast_examples(writer, series, mean, q90, clients, y, n=6):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(clients), n, replace=False)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.1 * n), sharex=True)
    hist_x, fut_x = np.arange(-N_WEEKS, 0), np.arange(HORIZON_WEEKS)
    colors = plt.cm.tab10(np.arange(len(FAMILIES)))
    for ax, ci in zip(axes, idx):
        for fi, f in enumerate(FAMILIES):
            if series[ci, fi].sum() == 0:
                continue
            ax.plot(hist_x, series[ci, fi], color=colors[fi], lw=1, label=f)
            ax.plot(fut_x, mean[ci, fi], color=colors[fi], lw=2, ls="--")
            ax.fill_between(fut_x, 0, q90[ci, fi], color=colors[fi], alpha=0.12)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_title(f"{clients[ci]} — true label: {y.iloc[ci]}", fontsize=9, loc="left")
        ax.legend(fontsize=7, ncol=7, loc="upper left")
    axes[-1].set_xlabel("weeks relative to cutoff (dashed = forecast mean, band = q90)")
    fig_to_tb(writer, "forecast_examples/valid", fig)


def gbm_curve(writer, clf, X, y):
    """Valid macro-F1 per boosting iteration (a 'training curve' for the classifier)."""
    for it, proba in enumerate(clf.staged_predict_proba(X)):
        if it % 10 == 0 or it == clf.n_iter_ - 1:
            writer.add_scalar("gbm_curve/valid_macro_f1", macro_f1(y, clf.classes_[proba.argmax(1)]), it + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["2.5", "3.0"], default="3.0")
    ap.add_argument("--mode", choices=["uni", "multi"], default=None, help="default: multi for 3.0, uni for 2.5")
    ap.add_argument("--device", default="cpu", help="3.0 only: cpu or mps")
    args = ap.parse_args()
    mode = args.mode or ("multi" if args.model == "3.0" else "uni")
    if args.model == "2.5" and mode == "multi":
        ap.error("TimesFM 2.5 is univariate only")
    tag = f"timesfm{args.model}_{mode}"
    OUT.mkdir(exist_ok=True)
    writer = SummaryWriter(RUNS / f"{datetime.now():%m%d-%H%M%S}_{tag}")
    writer.add_text("config", f"model={args.model} mode={mode} device={args.device} horizon={HORIZON_WEEKS}w context={N_WEEKS}w")

    splits = {s: load(s) for s in ["train", "valid", "test"]}
    labels = {s: pd.read_csv(DATA / f"{s}_labels.csv").set_index("client_id")["target_next_recurring_merchant"]
              for s in ["train", "valid"]}
    clients = {"train": labels["train"].index, "valid": labels["valid"].index,
               "test": pd.Index(pd.read_csv(DATA / "sample_submission.csv")["client_id"])}

    forecaster = None
    X, cache = {}, {}
    for s, tx in splits.items():
        sub = tag_family(tx)
        series = weekly_series(sub, clients[s])
        cache_path = OUT / f"forecast_{tag}_{s}.npz"
        if cache_path.exists():
            z = np.load(cache_path)
            mean, q90 = z["mean"], z["q90"]
            print(f"[{s}] loaded cached forecasts {cache_path.name}")
        else:
            forecaster = forecaster or Forecaster(args.model, mode, args.device, writer)
            print(f"[{s}] forecasting {series.shape[0]}x{series.shape[1]} series with TimesFM {args.model} ({mode}) ...", flush=True)
            t0 = time.time()
            mean, q90 = forecaster(series, s)
            writer.add_scalar(f"timing/forecast_seconds_{s}", time.time() - t0, 0)
            np.savez(cache_path, mean=mean, q90=q90)
        cache[s] = (series, mean, q90)
        X[s] = features(sub, clients[s], mean, q90).loc[clients[s]]
        X[s].to_csv(OUT / f"features_{tag}_{s}.csv")

    ytr, yva = labels["train"].loc[X["train"].index], labels["valid"].loc[X["valid"].index]
    log_forecast_examples(writer, *cache["valid"], clients["valid"], yva)

    # 4a. Pure TimesFM rule: tune window + threshold on train.
    best = max((macro_f1(ytr, rule_predict(X["train"], col, thr)), col, thr)
               for col in ["fc_4w", "fc_8w", "fc_sum"] for thr in np.arange(0.02, 3.0, 0.02))
    _, col, thr = best
    f1_rule = log_eval(writer, "rule", yva, rule_predict(X["valid"], col, thr))
    print(f"\n== TimesFM {args.model} {mode} rule ({col} >= {thr:.2f}) | train F1 {best[0]:.4f} | valid macro-F1 {f1_rule:.4f}")

    # 4b. TimesFM features + recurrence stats -> GBM.
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         class_weight="balanced", random_state=0)
    clf.fit(X["train"], ytr)
    gbm_curve(writer, clf, X["valid"], yva)
    p_gbm = clf.predict(X["valid"])
    f1_gbm = log_eval(writer, "gbm", yva, p_gbm)
    print(f"== TimesFM {args.model} {mode} + GBM | valid macro-F1 {f1_gbm:.4f}")
    print(classification_report(yva, p_gbm, labels=LABELS, zero_division=0))

    writer.add_hparams({"model": args.model, "mode": mode, "rule_window": col, "rule_thr": float(thr)},
                       {"hparam/valid_f1_rule": f1_rule, "hparam/valid_f1_gbm": f1_gbm}, run_name=".")

    # Refit on train+valid, predict test (files written locally only; nothing is submitted).
    clf.fit(pd.concat([X["train"], X["valid"]]), pd.concat([ytr, yva]))
    test_ids = clients["test"]
    for name, pred in [("gbm", clf.predict(X["test"].loc[test_ids])), ("rule", rule_predict(X["test"].loc[test_ids], col, thr))]:
        out = pd.DataFrame({"client_id": test_ids, "predicted_next_recurring_merchant": pred})
        assert out["predicted_next_recurring_merchant"].isin(LABELS).all() and len(out) == len(test_ids)
        out.to_csv(OUT / f"submission_{tag}_{name}.csv", index=False)
    writer.close()
    print(f"Wrote outputs/submission_{tag}_*.csv and TensorBoard run to {writer.log_dir}")


if __name__ == "__main__":
    main()
