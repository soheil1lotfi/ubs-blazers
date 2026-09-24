"""Competing-risks expert: which subscription pays first after the cutoff, or none within 90 days?

Survival-analysis view (competing risks, cf. Fine & Gray 1999): every stream is a "risk" racing to its next
payment; `none` means no stream fires inside the 90-day horizon.

  1. Continuation c_i = P(stream i pays again), from an XGBoost classifier trained on REPLAYS: at earlier
     cutoffs we observe directly whether each stream paid again (label-free, allowed data: train + unlabeled
     histories, clean and noise-augmented).
  2. Timing f(delta) = distribution of (actual next payment - projected due date), measured on the same
     replays (captures jitter and skipped months).
  3. Exact race on a daily grid t = 0..89:
        P(i first) = sum_t c_i f(t - due_i) * prod_{j != i} (1 - F_j(t-1)),   P(none) = prod_j (1 - F_j(89))
     Family probability = sum over streams of P(i first) x the stream's family vote share.

Output: an 8-way distribution per client (train at the real cutoff for tuning, valid, test-like valid, test),
saved in the same format as the other experts for blending.

Usage: .venv/bin/python risk_model.py
"""

import numpy as np
import pandas as pd

from stream_model import (CUTOFF, DATA, FAMILIES, LABELS, OUT, SUB_MCC, augment, build_candidates, feature_cols, load,
                          macro_f1, make_model)

CUTS = [pd.Timestamp(d, tz="UTC") for d in ["2025-07-01", "2025-08-01", "2025-09-01", "2025-10-01"]]
H, TOL, DMIN, DMAX = 90, 0.03, -30, 150
DROP = {"canon_frac", "n_desc_variants", "due_dom", "dom_med", "last_dom", "weekday_std"}


def streams(tx, T):
    """One row per stream (its most-voted candidate row) + its family vote distribution."""
    c = build_candidates(tx, cutoff=T)
    fam = c.pivot_table(index="sid", columns="family", values="fam_vote", aggfunc="max").reindex(columns=FAMILIES).fillna(0)
    fam = fam.div(fam.sum(1), axis=0)
    st = c.sort_values("fam_vote", ascending=False).drop_duplicates("sid").set_index("sid")
    return st, fam.loc[st.index]


def first_payment_after(tx, st, T):
    fut = tx[(tx["timestamp"] >= T) & (tx["timestamp"] < T + pd.Timedelta(days=DMAX)) & (tx["type"] == "card_payment")
             & tx["mcc"].isin(SUB_MCC)]
    m = fut.merge(st[["client_id", "amount"]].reset_index(), on="client_id", suffixes=("", "_s"))
    m = m[(np.log(m["amount"]) - np.log(m["amount_s"])).abs() <= TOL]
    return ((m.groupby("sid")["timestamp"].min() - T).dt.total_seconds() / 86400).reindex(st.index)


def race(st, fam, c_prob, pmf, clients):
    """8-way distribution per client from the exact discrete-time race."""
    out = np.zeros((len(clients), len(LABELS)))
    out[:, -1] = 1.0                                   # no streams -> none
    t = np.arange(H)
    st = st.assign(c=c_prob, i=np.arange(len(st)))
    fam = fam.to_numpy()
    pos = {cl: k for k, cl in enumerate(clients)}
    for cl, g in st.groupby("client_id"):
        if cl not in pos:
            continue
        d = np.clip(t[None, :] - np.round(g["proj_due"].to_numpy())[:, None], DMIN, DMAX - 1).astype(int) - DMIN
        p = g["c"].to_numpy()[:, None] * pmf[d]                     # [k, H] prob. stream fires on day t
        F = np.cumsum(p, 1)
        Fprev = np.concatenate([np.zeros((len(p), 1)), F[:, :-1]], 1)
        surv = np.clip(1 - Fprev, 1e-9, 1)
        others = np.exp(np.log(surv).sum(0, keepdims=True) - np.log(surv))   # prod over j != i
        first = (p * others).sum(1)
        none = np.prod(np.clip(1 - F[:, -1], 0, 1))
        dist = np.zeros(len(LABELS))
        dist[:-1] = first @ fam[g["i"].to_numpy()]
        dist[-1] = none
        out[pos[cl]] = dist / dist.sum()
    return out


def tune_bias(P, y, rounds=2):
    b = np.ones(P.shape[1])
    grid = np.arange(0.3, 3.01, 0.1)
    for _ in range(rounds):
        for j in range(len(b)):
            def f(v):
                bb = b.copy(); bb[j] = v
                return macro_f1(y, np.array(LABELS)[(P * bb).argmax(1)])
            b[j] = max(grid, key=f)
    return b


def main():
    # 1-2) replays: continuation labels + timing residuals
    rows = []
    for split in ["train", "unlabeled_pretrain"]:
        tx0 = load(split)
        for aug in [False, True]:
            tx = augment(tx0, "0.3", seed=1) if aug else tx0
            for T in CUTS:
                st, _ = streams(tx, T)
                first = first_payment_after(tx, st, T)
                st = st.assign(cont=first.notna().astype(int), delta=first - st["proj_due"])
                rows.append(st)
            print(f"replays built: {split} aug={aug}", flush=True)
    R = pd.concat(rows, ignore_index=True)
    cols = [c for c in feature_cols(R) if c not in DROP | {"cont", "delta", "family"}]
    print(f"{len(R)} replay streams; continuation rate {R['cont'].mean():.2f}", flush=True)
    cont_model = make_model().fit(R[cols], R["cont"])

    d = np.clip(np.round(R.loc[R["cont"] == 1, "delta"]), DMIN, DMAX - 1).astype(int) - DMIN
    pmf = np.bincount(d, minlength=DMAX - DMIN).astype(float)
    pmf = np.convolve(pmf, [0.25, 0.5, 0.25], mode="same") + 1e-3
    pmf /= pmf.sum()
    q = np.cumsum(pmf)
    print("timing residual (days) quantiles 10/50/90%:",
          [int(np.searchsorted(q, x)) + DMIN for x in (0.1, 0.5, 0.9)], flush=True)

    # 3) race for real clients at the real cutoff
    ytr = pd.read_csv(DATA / "train_labels.csv", index_col=0)["target_next_recurring_merchant"]
    yva = pd.read_csv(DATA / "valid_labels.csv", index_col=0)["target_next_recurring_merchant"]
    test_ids = pd.Index(pd.read_csv(DATA / "sample_submission.csv")["client_id"])
    P = {}
    for key, tx, ids in [("oof", load("train"), ytr.index), ("va", load("valid"), yva.index),
                         ("tl", augment(load("valid"), "g0.27m0.04d0", seed=7), yva.index), ("te", load("test"), test_ids)]:
        st, fam = streams(tx, CUTOFF)
        P[key] = race(st, fam, cont_model.predict_proba(st[cols])[:, 1], pmf, ids)
    b = tune_bias(P["oof"], ytr)
    pred = lambda M: np.array(LABELS)[(M * b).argmax(1)]
    print(f"competing-risks expert | train {macro_f1(ytr, pred(P['oof'])):.4f} (bias tuned here) | "
          f"VALID {macro_f1(yva, pred(P['va'])):.4f} | TEST-LIKE {macro_f1(yva, pred(P['tl'])):.4f}")
    print(f"  untuned argmax: VALID {macro_f1(yva, np.array(LABELS)[P['va'].argmax(1)]):.4f}")
    np.savez(OUT / "risk_scores.npz", oof=P["oof"], oof_ids=np.array(ytr.index), va=P["va"], tl=P["tl"], te=P["te"], bias=b)
    print("saved outputs/risk_scores.npz")


if __name__ == "__main__":
    main()
