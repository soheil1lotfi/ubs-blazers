"""Competing-risks race features for the stream model (stacking the survival model into XGBoost).

For every candidate row: P(its stream pays first after the cutoff), P(its family pays first), P(no stream pays
within 90 days), the stream's continuation probability, and the family's race rank within the client.

Leakage control (cross-fitting): the continuation model is fit on replay streams in two owner-disjoint halves.
Rows that XGBoost trains on get race features from the half-model that never saw their owner; valid/test rows
use the model fit on all replays. The timing-residual distribution is label-free and pooled.

Usage: imported by stream_model.py (--race). Running this file (re)builds the cached replay streams + models.
"""

import pickle

import numpy as np
import pandas as pd

from stream_model import (FAMILIES, OUT, SUB_MCC, augment, build_candidates, feature_cols, load, make_model, owner)

CUTS = [pd.Timestamp(d, tz="UTC") for d in ["2025-07-01", "2025-08-01", "2025-09-01", "2025-10-01"]]
H, TOL, DMIN, DMAX = 90, 0.03, -30, 150
DROP = {"canon_frac", "n_desc_variants", "due_dom", "dom_med", "last_dom", "weekday_std"}
RACE_COLS = ["race_c", "race_p_stream", "race_p_fam", "race_p_none", "race_rank"]


def _first_payment_after(tx, st, T):
    fut = tx[(tx["timestamp"] >= T) & (tx["timestamp"] < T + pd.Timedelta(days=DMAX)) & (tx["type"] == "card_payment")
             & tx["mcc"].isin(SUB_MCC)]
    m = fut.merge(st[["client_id", "amount"]].reset_index(), on="client_id", suffixes=("", "_s"))
    m = m[(np.log(m["amount"]) - np.log(m["amount_s"])).abs() <= TOL]
    return ((m.groupby("sid")["timestamp"].min() - T).dt.total_seconds() / 86400).reindex(st.index)


def replay_streams():
    f = OUT / "race_replays.pkl"
    if f.exists():
        return pd.read_pickle(f)
    rows = []
    for split in ["train", "unlabeled_pretrain"]:
        tx0 = load(split)
        for aug in [False, True]:
            tx = augment(tx0, "0.3", seed=1) if aug else tx0
            for T in CUTS:
                c = build_candidates(tx, cutoff=T)
                st = c.sort_values("fam_vote", ascending=False).drop_duplicates("sid").set_index("sid")
                first = _first_payment_after(tx, st, T)
                rows.append(st.assign(cont=first.notna().astype(int), delta=first - st["proj_due"]).reset_index(drop=True))
            print(f"race replays: {split} aug={aug}", flush=True)
    R = pd.concat(rows, ignore_index=True)
    R.to_pickle(f)
    return R


def fold_of(ids):
    return (pd.util.hash_pandas_object(pd.Series(owner(ids)), index=False).to_numpy() % 2).astype(int)


def fit_or_load():
    f = OUT / "race_models.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    R = replay_streams()
    cols = [c for c in feature_cols(R) if c not in DROP | {"cont", "delta"}]
    fold = fold_of(R["client_id"])
    models = {"full": make_model().fit(R[cols], R["cont"])}
    for k in (0, 1):  # model k never saw owners of fold k -> used to score fold-k rows
        models[k] = make_model().fit(R.loc[fold != k, cols], R.loc[fold != k, "cont"])
    d = np.clip(np.round(R.loc[R["cont"] == 1, "delta"]), DMIN, DMAX - 1).astype(int) - DMIN
    pmf = np.convolve(np.bincount(d, minlength=DMAX - DMIN).astype(float), [0.25, 0.5, 0.25], mode="same") + 1e-3
    race = {"models": models, "cols": cols, "pmf": pmf / pmf.sum()}
    f.write_bytes(pickle.dumps(race))
    print(f"race models fit on {len(R)} replay streams", flush=True)
    return race


def race_features(rows, race, crossfit):
    """Return `rows` with RACE_COLS added (same row order)."""
    key = ["client_id", "sid"]
    st = rows.sort_values("fam_vote", ascending=False).drop_duplicates(key).reset_index(drop=True)
    X = st[race["cols"]]
    if crossfit:
        fold = fold_of(st["client_id"])
        c = np.zeros(len(st))
        for k in (0, 1):
            if (fold == k).any():
                c[fold == k] = race["models"][k].predict_proba(X[fold == k])[:, 1]
    else:
        c = race["models"]["full"].predict_proba(X)[:, 1]
    st["race_c"] = c
    fam = rows.pivot_table(index=key, columns="family", values="fam_vote", aggfunc="max").reindex(columns=FAMILIES).fillna(0)
    fam = fam.div(fam.sum(axis=1), axis=0).reindex(pd.MultiIndex.from_frame(st[key])).to_numpy()

    pmf, t = race["pmf"], np.arange(H)
    p_stream = np.zeros(len(st)); p_none = np.zeros(len(st)); fam_prob = np.zeros((len(st), len(FAMILIES)))
    for _, idx in st.groupby("client_id").indices.items():
        g = st.iloc[idx]
        d = np.clip(t[None, :] - np.round(g["proj_due"].to_numpy())[:, None], DMIN, DMAX - 1).astype(int) - DMIN
        p = g["race_c"].to_numpy()[:, None] * pmf[d]
        F = np.cumsum(p, 1)
        surv = np.clip(1 - np.concatenate([np.zeros((len(p), 1)), F[:, :-1]], 1), 1e-9, 1)
        first = (p * np.exp(np.log(surv).sum(0, keepdims=True) - np.log(surv))).sum(1)
        p_stream[idx] = first
        p_none[idx] = np.prod(np.clip(1 - F[:, -1], 0, 1))
        fam_prob[idx] = first @ fam[idx]
    st["race_p_stream"], st["race_p_none"] = p_stream, p_none
    famp = pd.DataFrame(fam_prob, columns=FAMILIES).groupby(st["client_id"].to_numpy()).max()  # client x family
    out = rows.merge(st[key + ["race_c", "race_p_stream", "race_p_none"]], on=key, how="left")
    out["race_p_fam"] = famp.stack().reindex(pd.MultiIndex.from_arrays([out["client_id"], out["family"]])).to_numpy()
    out["race_rank"] = out.groupby("client_id")["race_p_fam"].rank(ascending=False, method="min")
    out.index = rows.index
    return out


if __name__ == "__main__":
    fit_or_load()
