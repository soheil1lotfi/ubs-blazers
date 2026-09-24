"""Stream-level model for next-recurring-merchant-family prediction.

1. Streams: per client, cluster subscription-MCC card payments by amount (single-linkage on
   log-amount, optional same currency). Descriptions are NOT used to build streams, because
   in valid/test real payments often carry generic descriptions.
2. Candidates: one row per (stream, candidate family). Family evidence = MCC votes (5812 is split
   between music/streaming) + description-keyword votes.
3. Model: regularized XGBoost binary classifier "is this the client's label family?".
   Client scores = max row score per family; `none` when the best score is below a threshold.
   Optional dedicated `none` model on client-level aggregates (--none_model).
   Threshold + per-class biases are tuned for macro-F1 on out-of-fold predictions over TRAIN clients.
4. Clean protocol (per README): train on train (+ self-labels from train/unlabeled histories, see
   pseudo_labels.py); valid is scored once and never trained on; test is only predicted.

Usage: .venv/bin/python stream_model.py [--pseudo off|train|unlabeled|both] [--none_model]
"""

import argparse
import io
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# xgboost and torch ship different OpenMP runtimes that segfault when both do heavy compute in one process.
# xgboost must load before torch here; seq_model.py (torch compute) sets STREAM_NO_XGB=1 to skip it entirely.
if os.environ.get("STREAM_NO_XGB") != "1":
    import xgboost as xgb
if os.environ.get("STREAM_PRELOAD_LGBM") == "1":  # same OpenMP issue: lightgbm must also load before torch
    import lightgbm  # noqa: F401
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).parent
DATA, OUT, RUNS = ROOT / "data", ROOT / "outputs", ROOT / "runs"
CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
FAMILIES = ["cloud", "gym", "insurance", "mobile", "music", "software", "streaming"]
LABELS = FAMILIES + ["none"]
SUB_MCC = ["4814", "5732", "5734", "6300", "7997", "5812"]
MCC_VOTES = {  # mcc -> {family: vote}
    "4814": {"mobile": 1.0}, "5732": {"cloud": 1.0}, "5734": {"software": 1.0},
    "6300": {"insurance": 1.0}, "7997": {"gym": 1.0}, "5812": {"music": 0.5, "streaming": 0.5},
}
KEYWORDS = [
    (r"audio|member pass|music", "music"),
    (r"media stream|video|stream", "streaming"),
    (r"phone|service bill|mobile", "mobile"),
    (r"cloud|storage", "cloud"),
    (r"saas|software|productivity|prod suite", "software"),
    (r"cover|policy|insurance", "insurance"),
    (r"gym|fit", "gym"),
]
GENERIC = (r"^(member plan|monthly plan|digital service|subscription charge|merchant charge|"
           r"service payment|card purchase|digital order)$")
# Canonical (un-scrambled) subscription names. Train analysis: the label stream uses them more often
# than the client's other streams (within-client AUC ~0.60).
CANONICAL = {"cloud access", "cloud backup", "storage plan", "service plan", "urban gym", "gym membership", "fit club",
             "fitness monthly", "cover plan", "insurance monthly", "policy premium", "safe cover", "phone contract",
             "service bill", "digital plus", "saas billing", "productivity suite", "software access", "premium plan",
             "media streaming", "video access", "audio streaming", "member pass"}


def load(split):
    tx = pd.read_json(DATA / f"{split}_transactions.jsonl", lines=True, dtype={"mcc": str})
    tx["timestamp"] = pd.to_datetime(tx["timestamp"], utc=True)
    return tx


GENERIC_PHRASES = ["member plan", "monthly plan", "digital service", "subscription charge", "merchant charge",
                   "service payment", "card purchase", "digital order"]


def parse_aug(spec):
    """'0.3' -> legacy (generic 0.3, mcc 0.1, drop 0.1); 'g0.5m0.15d0' -> explicit rates."""
    spec = str(spec)
    if spec[0].isdigit():
        lv = float(spec)
        return lv, lv / 3, lv / 3
    import re
    v = dict(re.findall(r"([gmd])([0-9.]+)", spec))
    return float(v.get("g", 0)), float(v.get("m", 0)), float(v.get("d", 0))


def augment(tx, level, seed=0):
    """Corrupt subscription-like payments to mimic the noisier valid/test histories: generic descriptions,
    wrong subscription MCC, dropped payments. `level` is a spec understood by parse_aug.
    (Measured on amount-based streams, valid/test differ from train only in descriptions and MCC noise,
    not in missing payments, so new specs use d0.)"""
    p_gen, p_mcc, p_drop = parse_aug(level)
    rng = np.random.default_rng(seed)
    tx = tx.copy()
    sub = ((tx["type"] == "card_payment") & tx["mcc"].isin(SUB_MCC)).to_numpy()
    idx = np.flatnonzero(sub)
    gen = idx[rng.random(len(idx)) < p_gen]
    tx.iloc[gen, tx.columns.get_loc("description")] = rng.choice(GENERIC_PHRASES, len(gen))
    swp = idx[rng.random(len(idx)) < p_mcc]
    tx.iloc[swp, tx.columns.get_loc("mcc")] = rng.choice(SUB_MCC, len(swp))
    drop = idx[rng.random(len(idx)) < p_drop]
    return tx.drop(tx.index[drop])


_PRICE = None


def price_features(c):
    """Price-cluster prior learned on unlabeled streams (price_prior.py): P(candidate family | amount)."""
    global _PRICE
    if _PRICE is None:
        f = OUT / "price_prior.npz"
        _PRICE = np.load(f) if f.exists() else False
    if _PRICE is False:
        return c
    edges, prob = _PRICE["edges"], _PRICE["prob"]
    b = np.clip(np.digitize(np.log(c["amount"].to_numpy()), edges) - 1, 0, len(edges) - 2)
    P = prob[b]                                              # [rows, 7]
    fam = c["fam_id"].to_numpy().astype(int)
    c["price_prob"] = P[np.arange(len(c)), fam]
    c["price_rank"] = (P > c["price_prob"].to_numpy()[:, None]).sum(1)
    mi, si = FAMILIES.index("music"), FAMILIES.index("streaming")
    c["price_music_vs_streaming"] = np.log(P[:, mi] / P[:, si])
    return c


def keyword_family(desc):
    fam = pd.Series(None, index=desc.index, dtype=object)
    for pat, f in KEYWORDS:
        fam = fam.where(fam.notna(), np.where(desc.str.contains(pat), f, None))
    return fam


def circ_std_dom(days):
    ang = 2 * np.pi * (days - 1) / 30.4
    R = np.hypot(np.cos(ang).mean(), np.sin(ang).mean())
    return np.sqrt(-2 * np.log(max(R, 1e-9))) * 30.4 / (2 * np.pi)


# keep single-payment "streams" whose payment is this recent (new subscriptions; +0.005 on valid); 0 = off
SINGLETON_DAYS = float(os.environ.get("STREAM_SINGLETON_DAYS", "45"))


def build_candidates(tx, cutoff=CUTOFF, tol=0.03, by_currency=False, min_n=2):
    """Return one row per (client, stream, candidate family) with stream + client features."""
    tx = tx[tx["timestamp"] < cutoff]
    pay = tx[(tx["type"] == "card_payment") & tx["mcc"].isin(SUB_MCC)].copy()
    ref = tx[(tx["type"] == "refund") & tx["mcc"].isin(SUB_MCC)]
    pay["la"] = np.log(pay["amount"])
    key = ["client_id", "currency"] if by_currency else ["client_id"]
    pay = pay.sort_values(key + ["la"])
    new = (pay[key] != pay[key].shift()).any(axis=1) | (pay["la"].diff() > tol)
    pay["sid"] = new.cumsum()
    size = pay.groupby("sid")["sid"].transform("size")
    recent = (cutoff - pay["timestamp"]).dt.days <= SINGLETON_DAYS
    pay = pay[(size >= min_n) | ((size == 1) & recent)].sort_values(["sid", "timestamp"])

    desc = pay["description"].str.lower()
    pay["generic"] = desc.str.match(GENERIC)
    pay["kw"] = keyword_family(desc).where(~pay["generic"])
    pay["gap"] = pay.groupby("sid")["timestamp"].diff().dt.total_seconds() / 86400
    pay["dom"] = pay["timestamp"].dt.day

    g = pay.groupby("sid")
    st = pd.DataFrame({
        "client_id": g["client_id"].first(),
        "n": g.size(),
        "amount": g["amount"].median(),
        "amt_cv": g["amount"].std() / g["amount"].mean(),
        "first": g["timestamp"].min(), "last": g["timestamp"].max(),
        "gap_med": g["gap"].median(), "gap_std": g["gap"].std(),
        "gap_mad": g["gap"].agg(lambda x: (x - x.median()).abs().median()),
        "gap_monthly_frac": g["gap"].agg(lambda x: x.between(24, 37).mean() if len(x.dropna()) else np.nan),
        "gap_short_frac": g["gap"].agg(lambda x: (x < 20).mean() if len(x.dropna()) else np.nan),
        "dom_std": g["dom"].agg(circ_std_dom),
        "cur_purity": g["currency"].agg(lambda x: x.value_counts(normalize=True).iloc[0]),
        "mcc_purity": g["mcc"].agg(lambda x: x.value_counts(normalize=True).iloc[0]),
        "generic_frac": g["generic"].mean(),
        "kw_frac": g["kw"].agg(lambda x: x.notna().mean()),
        "hour_std": g["timestamp"].agg(lambda t: t.dt.hour.std()),
        "weekday_std": g["timestamp"].agg(lambda t: t.dt.weekday.std()),
        "canon_frac": g["description"].agg(lambda d: d.str.lower().isin(CANONICAL).mean()),
        "n_desc_variants": g["description"].nunique(),
        "dom_med": g["dom"].median(),
    })
    st["last_dom"] = g["timestamp"].max().dt.day
    days = lambda s: (cutoff - s).dt.total_seconds() / 86400
    st["days_since"] = days(st["last"])
    st["age_days"] = days(st["first"])
    st["span_days"] = st["age_days"] - st["days_since"]
    gm = st["gap_med"].fillna(30.4).clip(7, 120)
    st["due_in"] = gm - st["days_since"]                       # days from cutoff to expected next payment
    st["overdue_ratio"] = st["days_since"] / gm
    st["missed_cycles"] = np.floor(st["days_since"] / gm)
    st["expected_n"] = st["span_days"] / gm + 1
    st["fill_ratio"] = st["n"] / st["expected_n"]              # <1 -> missing months
    n90 = pay[pay["timestamp"] >= cutoff - pd.Timedelta(days=90)].groupby("sid").size()
    st["n_90d"] = n90.reindex(st.index).fillna(0)
    st["due_in_pos"] = st["due_in"].clip(lower=0)
    # --- extra time representation (stream level) ---
    ts_days = (cutoff - pay["timestamp"]).dt.total_seconds() / 86400
    pay["decay"] = np.exp(-ts_days / 45)
    st["decay_n"] = pay.groupby("sid")["decay"].sum()
    gl = pay.dropna(subset=["gap"]).groupby("sid")["gap"]
    st["gap_last"] = gl.last()
    pg_ = pay.dropna(subset=["gap"])
    st["gap_prev"] = pg_[pg_.groupby("sid").cumcount(ascending=False) == 1].set_index("sid")["gap"]
    st["gap_trend"] = (st["gap_last"] - st["gap_med"]) / st["gap_med"].clip(lower=7)
    # lag features: the last few gaps / payment days as a sequence (most recent first)
    rev = pg_.groupby("sid").cumcount(ascending=False)
    for k in (2, 3):
        st[f"gap_l{k + 1}"] = pg_[rev == k].set_index("sid")["gap"]
    prev = pay.groupby("sid").cumcount(ascending=False)
    st["dom_l1"] = pay[prev == 0].set_index("sid")["dom"]
    st["dom_l2"] = pay[prev == 1].set_index("sid")["dom"]
    st["dom_shift"] = ((st["dom_l1"] - st["dom_l2"] + 15) % 31) - 15
    st["days_since_2nd"] = (cutoff - pay[prev == 1].set_index("sid")["timestamp"]).dt.total_seconds() / 86400
    for w in (30, 60, 180):
        st[f"n_{w}d"] = (ts_days <= w).groupby(pay["sid"]).sum()
    st["long_period"] = (st["gap_med"] > 80).astype(float)
    st["amt_last_ratio"] = g["amount"].last() / st["amount"] - 1
    sal = tx[(tx["type"] == "topup") & tx["description"].str.contains("salary")][["client_id", "timestamp"]].sort_values("timestamp")
    ps = pd.merge_asof(pay[["sid", "client_id", "timestamp"]].sort_values("timestamp"), sal.rename(columns={"timestamp": "sal_ts"}),
                       left_on="timestamp", right_on="sal_ts", by="client_id", direction="backward")
    ps["lag"] = (ps["timestamp"] - ps["sal_ts"]).dt.total_seconds() / 86400
    st["payday_lag_med"] = ps.groupby("sid")["lag"].median()
    st["payday_lag_std"] = ps.groupby("sid")["lag"].std()
    # projected next cycle on/after the cutoff: skips missed (dropped) payments instead of calling them overdue
    k = np.ceil(st["days_since"] / gm).clip(lower=1)
    st["proj_due"] = k * gm - st["days_since"]
    st["alive"] = (st["missed_cycles"] <= 1).astype(float)
    # expected payments inside the 90-day horizon (uses the real gap, not the 120-day cap)
    g_true = st["gap_med"].fillna(30.4).clip(lower=7)
    st["exp_pay_h"] = np.where(st["proj_due"] < 90, np.floor((90 - st["proj_due"]) / g_true) + 1, 0)
    st["due_true"] = np.ceil(st["days_since"] / g_true).clip(lower=1) * g_true - st["days_since"]
    # due date from the stream's day-of-month phase: first date >= cutoff on its usual day of the month,
    # at least 15 days after its last payment
    month0 = cutoff.normalize() - pd.Timedelta(days=cutoff.day - 1)
    off = pd.to_timedelta(st["dom_med"].round().clip(1, 28) - 1, unit="D")
    due_dom = pd.Series(np.nan, index=st.index)
    for k in (2, 1, 0):  # earliest valid month wins
        cand = month0 + pd.DateOffset(months=k) + off
        ok = (cand >= cutoff) & ((cand - st["last"]).dt.days >= 15)
        due_dom = due_dom.mask(ok, (cand - cutoff).dt.days)
    st["due_dom"] = due_dom.fillna(90)

    # refunds matching the stream amount
    r = ref.merge(st[["client_id", "amount", "last"]].reset_index(), on="client_id", suffixes=("", "_s"))
    r = r[(np.log(r["amount"]) - np.log(r["amount_s"])).abs() <= tol]
    rg = r.groupby("sid")
    st["n_refunds"] = rg.size().reindex(st.index).fillna(0)
    st["refund_days_since"] = days(rg["timestamp"].max()).reindex(st.index)
    st["refund_after_last"] = (rg["timestamp"].max() >= st["last"].reindex(rg.size().index)).reindex(st.index).fillna(False).astype(float)

    # family votes: MCC votes + keyword votes (keywords count double: they're rarer but specific)
    votes = np.zeros((len(pay), len(FAMILIES)))
    fi = {f: i for i, f in enumerate(FAMILIES)}
    for mcc, vv in MCC_VOTES.items():
        m = (pay["mcc"] == mcc).to_numpy()
        for f, v in vv.items():
            votes[m, fi[f]] += v
    kwm = pay["kw"].notna().to_numpy()
    votes[kwm, pay.loc[kwm, "kw"].map(fi).to_numpy()] += 2.0
    V = pd.DataFrame(votes, index=pay["sid"], columns=FAMILIES).groupby(level=0).sum()
    V = V.div(V.sum(axis=1), axis=0).loc[st.index]

    rows = []
    for f in FAMILIES:  # candidate families: vote share >= 0.25
        m = V[f] >= 0.25
        d = st[m].copy()
        d["family"] = f
        d["fam_vote"] = V.loc[m, f]
        d["fam_vote_rank"] = V[m].rank(axis=1, ascending=False)[f]
        rows.append(d)
    c = pd.concat(rows).reset_index().rename(columns={"index": "sid"})
    c["amb_5812"] = ((c["family"].isin(["music", "streaming"])) & (c["fam_vote"] < 0.75)).astype(float)

    # client-relative features (within client, over "established" candidates)
    est = (c["n"] >= 3).astype(float)
    grp = c.groupby("client_id")
    c["cl_n_streams"] = c.groupby("client_id")["sid"].transform("nunique")
    c["cl_n_established"] = est.groupby(c["client_id"]).transform("sum")
    c["rank_due"] = grp["due_in_pos"].rank(method="min")
    c["rank_n"] = grp["n"].rank(method="min", ascending=False)
    c["rank_since"] = grp["days_since"].rank(method="min")
    c["rank_age"] = grp["age_days"].rank(method="min")
    c["due_minus_best"] = c["due_in_pos"] - grp["due_in_pos"].transform("min")
    c["n_share"] = c["n"] / grp["n"].transform("sum")
    c["fam_n_streams"] = c.groupby(["client_id", "family"])["sid"].transform("nunique")
    c["fam_id"] = c["family"].map(fi)
    # timing relative to the client's ALIVE streams only (dead streams used to pollute "which is due first")
    pa = c["proj_due"].where(c["alive"] == 1)
    ga = pa.groupby(c["client_id"])
    c["n_alive_streams"] = c["sid"].where(c["alive"] == 1).groupby(c["client_id"]).transform("nunique")
    c["rank_proj_alive"] = ga.rank(method="min")
    c["proj_minus_best_alive"] = pa - ga.transform("min")
    c["is_first_alive"] = (c["proj_minus_best_alive"] == 0).astype(float)
    second = ga.transform(lambda x: x.nsmallest(2).iloc[-1] if x.notna().sum() >= 2 else np.nan)
    c["lead_over_next_alive"] = np.where(c["is_first_alive"] == 1, second - pa, np.nan)
    c = price_features(c)
    # --- extra time representation (client level) ---
    dd = (cutoff - tx["timestamp"]).dt.total_seconds() / 86400
    t2 = pd.DataFrame({"client_id": tx["client_id"], "dd": dd, "w30": dd <= 30, "w120": (dd > 30) & (dd <= 120),
                       "dec": np.exp(-dd / 43)}).groupby("client_id")
    cl = pd.DataFrame({"cl_tx_30d": t2["w30"].sum(), "cl_tx_prev90d": t2["w120"].sum(),
                       "cl_days_since_any": t2["dd"].min(), "cl_decay_act": t2["dec"].sum()})
    cl["cl_activity_ratio"] = cl["cl_tx_30d"] / (cl["cl_tx_prev90d"] / 3 + 1)
    sd = pd.DataFrame({"client_id": sal["client_id"].to_numpy(),
                       "dd": ((cutoff - sal["timestamp"]).dt.total_seconds() / 86400).to_numpy()})
    sd["gap"] = sd.groupby("client_id")["dd"].diff()          # sal is time-sorted -> dd decreasing
    sg = sd.groupby("client_id")
    cl["cl_salary_days_since"] = sg["dd"].min()
    cl["cl_salary_gap_med"] = (-sg["gap"].median()).where(lambda x: x > 0)
    cl["cl_salary_overdue"] = cl["cl_salary_days_since"] / cl["cl_salary_gap_med"].clip(lower=7)
    cl["cl_salary_n90"] = sd.assign(r=sd["dd"] <= 90).groupby("client_id")["r"].sum()
    pdd = (cutoff - pay["timestamp"]).dt.total_seconds() / 86400
    pg = pd.DataFrame({"client_id": pay["client_id"], "a": pdd <= 60, "b": (pdd > 60) & (pdd <= 180)}).groupby("client_id")
    cl["cl_sub_trend"] = pg["a"].sum() / (pg["b"].sum() / 2 + 1)
    c = c.join(cl, on="client_id")
    n_tx = tx.groupby("client_id").size()
    refunds = tx[tx["type"] == "refund"]
    c["cl_refund_share"] = c["client_id"].map(refunds.groupby("client_id").size() / n_tx).fillna(0)
    c["cl_sub_refunds"] = c["client_id"].map(ref.groupby("client_id").size()).fillna(0)
    c["cl_sub_refunds_90d"] = c["client_id"].map(
        ref[ref["timestamp"] >= cutoff - pd.Timedelta(days=90)].groupby("client_id").size()).fillna(0)
    return c.drop(columns=["first", "last"])


def feature_cols(c):
    return [x for x in c.columns if x not in ("client_id", "sid", "family", "y")]


def attach_labels(c, labels):
    c = c.copy()
    c["y"] = (c["family"] == c["client_id"].map(labels)).astype(int)
    return c


ALGO = "xgb"


class _SkTrees:
    """RandomForest / ExtraTrees with NaN filled (bagged deep trees: a different inductive bias from boosting)."""
    def __init__(self, kind):
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        cls = RandomForestClassifier if kind == "rf" else ExtraTreesClassifier
        self.m = cls(n_estimators=400, min_samples_leaf=20, max_features=0.4, n_jobs=8, random_state=0)

    def fit(self, X, y, sample_weight=None):
        self.m.fit(X.fillna(-999), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X.fillna(-999))


def make_model(n_estimators=600):
    if ALGO in ("rf", "et"):
        return _SkTrees(ALGO)
    if ALGO == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, n_jobs=8,
                                  random_state=0, verbose=-1)
    return xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=0.03, max_depth=4, min_child_weight=5,
                             subsample=0.8, colsample_bytree=0.7, reg_lambda=5, n_jobs=8, random_state=0,
                             eval_metric="logloss")


def client_scores(c, p, clients):
    """[n_clients, 7] matrix: max candidate probability per family (0 when no candidate)."""
    d = pd.DataFrame({"client_id": c["client_id"].to_numpy(), "family": c["family"].to_numpy(), "p": p})
    S = d.pivot_table(index="client_id", columns="family", values="p", aggfunc="max")
    return S.reindex(index=clients, columns=FAMILIES).fillna(0.0).to_numpy()


def macro_f1(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


# ---------- TensorBoard helpers ----------

def fig_to_tb(writer, tag, fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    writer.add_image(tag, plt.imread(buf)[..., :3], 0, dataformats="HWC")


def log_eval(writer, name, y, p):
    f1 = macro_f1(y, p)
    writer.add_scalar(f"valid_macro_f1/{name}", f1, 0)
    pr, rc, f1s, _ = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    for lab, a, b, c in zip(LABELS, pr, rc, f1s):
        writer.add_scalar(f"per_class_f1_{name}/{lab}", c, 0)
        writer.add_scalar(f"per_class_recall_{name}/{lab}", b, 0)
        writer.add_scalar(f"per_class_precision_{name}/{lab}", a, 0)
    cm = confusion_matrix(y, p, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm / cm.sum(1, keepdims=True).clip(1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right"); ax.set_yticks(range(len(LABELS)), LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{name} — valid macro-F1 {f1:.4f}")
    fig_to_tb(writer, f"confusion/{name}", fig)
    return f1


def log_importance(writer, model, cols):
    if hasattr(model, "get_booster"):
        imp = pd.Series(model.get_booster().get_score(importance_type="gain")).reindex(cols).fillna(0).sort_values()
    elif hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=cols).sort_values()
    else:
        imp = pd.Series(model.m.feature_importances_, index=cols).sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(imp) + 1))
    ax.barh(imp.index, imp.values, color="#4C72B0")
    ax.set_title("XGBoost feature importance (gain)")
    fig_to_tb(writer, "feature_importance", fig)
    return imp


def client_features(c, clients):
    """Client-level aggregates of candidate streams, for the dedicated `none` model."""
    alive = c[(c["n"] >= 3) & (c["overdue_ratio"] < 1.5)]
    g = c.groupby("client_id")
    f = pd.DataFrame({
        "n_streams": g["sid"].nunique(), "max_n": g["n"].max(), "sum_n": g["n"].sum(),
        "min_due": g["due_in_pos"].min(), "min_missed": g["missed_cycles"].min(),
        "min_overdue": g["overdue_ratio"].min(), "max_n90": g["n_90d"].max(),
        "min_days_since": g["days_since"].min(), "max_fam_vote": g["fam_vote"].max(),
        "max_fill": g["fill_ratio"].max(), "min_gap_std": g["gap_std"].min(),
        "max_monthly": g["gap_monthly_frac"].max(), "mean_generic": g["generic_frac"].mean(),
        "sum_refunds": g["n_refunds"].sum(), "any_refund_after_last": g["refund_after_last"].max(),
        "min_refund_days": g["refund_days_since"].min(),
        "n_alive": alive.groupby("client_id")["sid"].nunique(),
    }).reindex(clients)
    f[["n_streams", "sum_n", "n_alive"]] = f[["n_streams", "sum_n", "n_alive"]].fillna(0)
    return f


def decide(S, thr, bias, P_none=None):
    """Family = argmax(S * bias). `none` if the best score < thr, or (with a none model) P_none >= thr."""
    adj = S * bias
    fam = np.array(FAMILIES)[adj.argmax(1)]
    is_none = (P_none >= thr) | (S.max(1) == 0) if P_none is not None else adj.max(1) < thr
    return np.where(is_none, "none", fam)


def tune_decision(S, y, P_none=None, rounds=3):
    """Coordinate ascent on the none-threshold + per-family multiplicative bias, maximizing macro-F1."""
    bias, thr = np.ones(len(FAMILIES)), 0.3
    grid_t, grid_b = np.arange(0.02, 0.98, 0.02), np.arange(0.5, 2.01, 0.05)
    score = lambda t, b: macro_f1(y, decide(S, t, b, P_none))
    for _ in range(rounds):
        thr = max(grid_t, key=lambda t: score(t, bias))
        for i in range(len(FAMILIES)):
            bias[i] = max(grid_b, key=lambda v: score(thr, np.where(np.arange(len(bias)) == i, v, bias)))
    return thr, bias


def owner(ids):
    """Real client behind a (pseudo-)client id: 'C000001@2025-07' -> 'C000001'."""
    return pd.Series(ids).str.split(r"[@#]", regex=True).str[0].to_numpy()


def fit_models(Crows, cols, Yclients, w_rows, w_clients, use_none):
    m = make_model().fit(Crows[cols], Crows["y"], sample_weight=w_rows)
    mn = None
    if use_none:
        F = client_features(Crows, Yclients.index)
        mn = make_model(400).fit(F, (Yclients == "none").astype(int), sample_weight=w_clients)
    return m, mn


def predict(m, mn, Crows, cols, clients):
    S = client_scores(Crows, m.predict_proba(Crows[cols])[:, 1], clients)
    Pn = mn.predict_proba(client_features(Crows, clients))[:, 1] if mn is not None else None
    return S, Pn


def main():
    ap = argparse.ArgumentParser(description="Clean protocol: train on train (+ self-labels from train/unlabeled "
                                             "histories); tune on train OOF; score valid once; predict test.")
    ap.add_argument("--tol", type=float, default=0.03, help="amount tolerance (log) for stream clustering")
    ap.add_argument("--min_n", type=int, default=2)
    ap.add_argument("--pseudo", choices=["off", "train", "unlabeled", "both"], default="off")
    ap.add_argument("--pseudo_weight", type=float, default=0.5)
    ap.add_argument("--none_model", action="store_true")
    ap.add_argument("--augment", default="", help="noise spec for augmented copies of train (+ pseudo), e.g. 0.3 or g0.5m0.15d0")
    ap.add_argument("--testlike", default="g0.27m0.04d0",
                    help="extra noise added to valid to reach test's measured description/MCC noise (test-like valid)")
    # Default drop list: features that scored well on train but lowered valid macro-F1 (ablation, see README):
    # description-based ones break under test-level noise; the extra timing ones added nothing.
    ap.add_argument("--drop", default="canon_frac,n_desc_variants,due_dom,dom_med,last_dom,weekday_std",
                    help="comma-separated feature columns to exclude")
    ap.add_argument("--dump_cache", action="store_true", help="save the prepared training/eval data for tune.py")
    ap.add_argument("--race", action="store_true", help="add cross-fitted competing-risks race features (race.py)")
    ap.add_argument("--algo", choices=["xgb", "lgbm", "rf", "et"], default="xgb")
    ap.add_argument("--use_valid", action="store_true",
                    help="also train on valid labels (+ valid replays); estimate via 5-fold CV over valid clients")
    ap.add_argument("--valid_weight", type=float, default=2.0, help="weight of valid (test-like, noisy) clients")
    ap.add_argument("--name", default="clean")
    args = ap.parse_args()
    global ALGO
    ALGO = args.algo
    if args.algo != "xgb":
        args.name = f"{args.name}_{args.algo}"
    if args.race:
        args.name = f"{args.name}_race"
    if args.use_valid and args.name == "clean":
        args.name = "withvalid"
    tag = f"{args.name}_pseudo-{args.pseudo}{args.pseudo_weight if args.pseudo != 'off' else ''}{'_nonemodel' if args.none_model else ''}{f'_aug{args.augment}' if args.augment else ''}"
    writer = SummaryWriter(RUNS / f"{datetime.now():%m%d-%H%M%S}_{tag}")
    writer.add_text("config", json.dumps(vars(args)))

    lab = {s: pd.read_csv(DATA / f"{s}_labels.csv", index_col=0)["target_next_recurring_merchant"] for s in ["train", "valid"]}
    test_ids = pd.Index(pd.read_csv(DATA / "sample_submission.csv")["client_id"])
    C = {s: build_candidates(load(s), tol=args.tol, min_n=args.min_n) for s in ["train", "valid", "test"]}
    C["train"] = attach_labels(C["train"], lab["train"])
    cols = [x for x in feature_cols(C["train"]) if x not in set(filter(None, args.drop.split(",")))]

    # Training pool = real train clients (+ pseudo-clients from allowed histories)
    rows, ys = [C["train"]], [lab["train"]]
    sources = {"off": [], "train": ["train"], "unlabeled": ["unlabeled_pretrain"], "both": ["train", "unlabeled_pretrain"]}[args.pseudo]
    for s in sources:
        pc, py = pd.read_pickle(OUT / f"pseudo_{s}.pkl")
        rows.append(pc); ys.append(py)
    if args.augment:
        ca = attach_labels(build_candidates(augment(load("train"), args.augment), tol=args.tol, min_n=args.min_n), lab["train"])
        ca["client_id"] = ca["client_id"] + "#aug"
        ya = lab["train"].copy(); ya.index = ya.index + "#aug"
        rows.append(ca); ys.append(ya)
        for s in sources:
            pc, py = pd.read_pickle(OUT / f"pseudo_{s}_aug{args.augment}.pkl")
            rows.append(pc); ys.append(py)
    P_rows, P_y = pd.concat(rows, ignore_index=True), pd.concat(ys)
    is_real_row = pd.Index(owner(P_rows["client_id"])).isin(lab["train"].index) & ~P_rows["client_id"].str.contains("@").to_numpy()
    is_real_cl = pd.Index(owner(P_y.index)).isin(lab["train"].index) & ~P_y.index.str.contains("@")
    w_rows = np.where(is_real_row, 1.0, args.pseudo_weight)
    w_cl = np.where(is_real_cl, 1.0, args.pseudo_weight)
    print(f"training pool: {is_real_cl.sum()} real + {(~is_real_cl).sum()} pseudo clients, {len(P_rows)} rows")
    if args.race:
        import race as racemod
        RM = racemod.fit_or_load()
        P_rows = racemod.race_features(P_rows, RM, crossfit=True)      # training rows: owner-disjoint half-models
        C["train"] = racemod.race_features(C["train"], RM, crossfit=True)
        C["valid"] = racemod.race_features(C["valid"], RM, crossfit=False)
        C["test"] = racemod.race_features(C["test"], RM, crossfit=False)
        cols = cols + racemod.RACE_COLS
        print("added race features:", racemod.RACE_COLS, flush=True)
    if args.dump_cache:
        C_tl = build_candidates(augment(load("valid"), args.testlike, seed=7), tol=args.tol, min_n=args.min_n)
        if args.race:
            C_tl = racemod.race_features(C_tl, RM, crossfit=False)
        pd.to_pickle(dict(lab=lab, C=C, C_tl=C_tl, cols=cols, P_rows=P_rows, P_y=P_y, w_rows=w_rows, w_cl=w_cl,
                          test_ids=test_ids), OUT / f"cache_{tag}.pkl")
        print(f"cached prepared data -> outputs/cache_{tag}.pkl", flush=True)
    if args.use_valid:
        return run_with_valid(args, tag, writer, lab, C, cols, P_rows, P_y, w_rows, w_cl, test_ids)

    # 1) Out-of-fold scores on real train clients (grouped by owner: a held-out client's pseudo copies are excluded)
    ytr = lab["train"]
    S_oof = np.zeros((len(ytr), len(FAMILIES)))
    Pn_oof = np.zeros(len(ytr)) if args.none_model else None
    row_owner, cl_owner = owner(P_rows["client_id"]), owner(P_y.index)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(ytr.index, ytr):
        held = set(ytr.index[va])
        rm, cm = ~np.isin(row_owner, list(held)), ~np.isin(cl_owner, list(held))
        m, mn = fit_models(P_rows[rm], cols, P_y[cm], w_rows[rm], w_cl[cm], args.none_model)
        S, Pn = predict(m, mn, C["train"][C["train"]["client_id"].isin(held)], cols, ytr.index[va])
        S_oof[va] = S
        if Pn is not None:
            Pn_oof[va] = Pn
    thr, bias = tune_decision(S_oof, ytr, Pn_oof)
    f1_oof = macro_f1(ytr, decide(S_oof, thr, bias, Pn_oof))
    writer.add_scalar("train_oof_macro_f1", f1_oof, 0)
    print(f"train OOF macro-F1 {f1_oof:.4f} | decision: thr {thr:.2f}, bias {dict(zip(FAMILIES, bias.round(2)))}")

    # 2) Fit on the full training pool, score valid ONCE
    m, mn = fit_models(P_rows, cols, P_y, w_rows, w_cl, args.none_model)
    S_va, Pn_va = predict(m, mn, C["valid"], cols, lab["valid"].index)
    pred_va = decide(S_va, thr, bias, Pn_va)
    f1_va = log_eval(writer, "valid", lab["valid"], pred_va)
    pd.DataFrame(S_va, index=lab["valid"].index, columns=FAMILIES).assign(
        predicted=pred_va, truth=lab["valid"]).to_csv(OUT / f"valid_scores_{tag}.csv")
    print(f"VALID macro-F1 {f1_va:.4f}")
    if args.testlike:
        C_tl = build_candidates(augment(load("valid"), args.testlike, seed=7), tol=args.tol, min_n=args.min_n)
        if args.race:
            C_tl = racemod.race_features(C_tl, RM, crossfit=False)
        S_tl, Pn_tl = predict(m, mn, C_tl, cols, lab["valid"].index)
        f1_tl = log_eval(writer, "valid_testlike", lab["valid"], decide(S_tl, thr, bias, Pn_tl))
        print(f"TEST-LIKE VALID macro-F1 {f1_tl:.4f} (valid + {args.testlike} noise)")
    imp = log_importance(writer, m, cols)
    print("top features:", imp.sort_values(ascending=False).head(10).round(1).to_dict())

    # 3) Test predictions from the same model (trained without any valid labels)
    S_te, Pn_te = predict(m, mn, C["test"], cols, test_ids)
    np.savez(OUT / f"xgb_scores_{tag}.npz", oof=S_oof, oof_ids=np.array(ytr.index), va=S_va, te=S_te,
             tl=S_tl if args.testlike else S_va, thr=thr, bias=bias)
    sub = pd.DataFrame({"client_id": test_ids, "predicted_next_recurring_merchant": decide(S_te, thr, bias, Pn_te)})
    assert sub["predicted_next_recurring_merchant"].isin(LABELS).all() and len(sub) == len(test_ids)
    OUT.mkdir(exist_ok=True)
    sub.to_csv(OUT / f"submission_{tag}.csv", index=False)
    print("test prediction mix:", sub["predicted_next_recurring_merchant"].value_counts().to_dict())
    writer.add_hparams({"pseudo": args.pseudo, "pseudo_weight": args.pseudo_weight, "none_model": args.none_model, "augment": args.augment,
                        "tol": args.tol, "min_n": args.min_n, "drop": args.drop},
                       {"hparam/valid_macro_f1": f1_va, "hparam/train_oof_macro_f1": f1_oof,
                        "hparam/valid_testlike_macro_f1": f1_tl if args.testlike else float("nan")}, run_name=".")
    writer.close()
    print(f"wrote outputs/submission_{tag}.csv (not submitted)")


def run_with_valid(args, tag, writer, lab, C, cols, P_rows, P_y, w_rows, w_cl, test_ids):
    """Train on train + valid. Honest estimate: 5-fold CV over valid clients; every fold trains on the full
    train pool + the other 4/5 of valid (labels + their replays); decision tuned out-of-fold (nested)."""
    yv = lab["valid"]
    V_rows = [attach_labels(C["valid"], yv)]
    V_y = [yv]
    if args.pseudo != "off":
        for f in ["pseudo_valid.pkl"] + ([f"pseudo_valid_aug{args.augment}.pkl"] if args.augment else []):
            pc, py = pd.read_pickle(OUT / f)
            V_rows.append(pc); V_y.append(py)
    V_rows, V_y = pd.concat(V_rows, ignore_index=True), pd.concat(V_y)
    v_real_row = ~V_rows["client_id"].str.contains("@").to_numpy()
    v_real_cl = ~V_y.index.str.contains("@")
    vw_rows = np.where(v_real_row, args.valid_weight, args.pseudo_weight * args.valid_weight)
    vw_cl = np.where(v_real_cl, args.valid_weight, args.pseudo_weight * args.valid_weight)
    v_row_owner, v_cl_owner = owner(V_rows["client_id"]), owner(V_y.index)
    print(f"valid added to training: {v_real_cl.sum()} real + {(~v_real_cl).sum()} replay clients")

    S_oof = np.zeros((len(yv), len(FAMILIES)))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(yv.index, yv):
        held = list(yv.index[va])
        rm, cm = ~np.isin(v_row_owner, held), ~np.isin(v_cl_owner, held)
        X = pd.concat([P_rows, V_rows[rm]], ignore_index=True)
        Y = pd.concat([P_y, V_y[cm]])
        m, _ = fit_models(X, cols, Y, np.r_[w_rows, vw_rows[rm]], np.r_[w_cl, vw_cl[cm]], False)
        S_oof[va], _ = predict(m, None, C["valid"][C["valid"]["client_id"].isin(set(held))], cols, yv.index[va])
    pred = np.empty(len(yv), dtype=object)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=1).split(yv.index, yv):
        t, b = tune_decision(S_oof[tr], yv.iloc[tr])
        pred[va] = decide(S_oof[va], t, b)
    f1_cv = log_eval(writer, "valid_cv_with_valid", yv, pred)
    thr, bias = tune_decision(S_oof, yv)
    print(f"CV-on-valid macro-F1 {f1_cv:.4f} (train + other valid folds; decision tuned out-of-fold)")
    print(f"decision: thr {thr:.2f}, bias {dict(zip(FAMILIES, bias.round(2)))}")

    X = pd.concat([P_rows, V_rows], ignore_index=True)
    Y = pd.concat([P_y, V_y])
    m, _ = fit_models(X, cols, Y, np.r_[w_rows, vw_rows], np.r_[w_cl, vw_cl], False)
    S_te, _ = predict(m, None, C["test"], cols, test_ids)
    sub = pd.DataFrame({"client_id": test_ids, "predicted_next_recurring_merchant": decide(S_te, thr, bias)})
    assert sub["predicted_next_recurring_merchant"].isin(LABELS).all() and len(sub) == len(test_ids)
    sub.to_csv(OUT / f"submission_{tag}.csv", index=False)
    print("test prediction mix:", sub["predicted_next_recurring_merchant"].value_counts().to_dict())
    writer.add_hparams({"pseudo": args.pseudo, "augment": args.augment, "use_valid": True, "valid_weight": args.valid_weight},
                       {"hparam/valid_cv_with_valid": f1_cv}, run_name=".")
    writer.close()
    print(f"wrote outputs/submission_{tag}.csv (not submitted)")


if __name__ == "__main__":
    main()
