"""Mixture of experts: blend the XGBoost stream model with a sequence model (Transformer/GRU).

Experts
  - XGBoost: 7 family scores S (max candidate probability) + its tuned none-threshold thr.
    Turned into an 8-way distribution: [S_1..S_7, thr], normalised.
  - Sequence model: 8-way softmax.
Gate
  - "global": one weight alpha for all clients.
  - "segment": a separate alpha per client type (<=1 alive subscription vs >=2), i.e. a hand-built gate.
Weights and per-class biases are tuned on held-out TRAIN clients (the sequence model's 20% hold-out, where both
experts made honest out-of-sample predictions). Valid and test-like valid are scored once.

Usage: .venv/bin/python blend.py --xgb outputs/xgb_scores_<tag>.npz --seq outputs/seq_scores_<tag>.npz
"""

import argparse

import numpy as np
import pandas as pd

from stream_model import CUTOFF, DATA, LABELS, OUT, augment, build_candidates, load, macro_f1

ALPHAS = np.round(np.arange(0, 1.01, 0.1), 2)


def xgb_dist(S, thr):
    P = np.concatenate([S, np.full((len(S), 1), thr)], 1)
    return P / P.sum(1, keepdims=True)


def tune_bias(P, y, rounds=2):
    b = np.ones(P.shape[1])
    grid = np.arange(0.4, 2.51, 0.1)
    for _ in range(rounds):
        for j in range(len(b)):
            def f(v):
                bb = b.copy(); bb[j] = v
                return macro_f1(y, np.array(LABELS)[(P * bb).argmax(1)])
            b[j] = max(grid, key=f)
    return b


def segments(tx, clients, cutoff=CUTOFF):
    c = build_candidates(tx, cutoff=cutoff)
    n_alive = c[c["alive"] == 1].groupby("client_id")["sid"].nunique().reindex(clients).fillna(0)
    return (n_alive >= 2).astype(int).to_numpy()          # 0: <=1 alive subscription, 1: several


def blend(Px, Ps, alpha_by_seg, seg):
    a = np.array([alpha_by_seg[s] for s in seg])[:, None]
    return a * Px + (1 - a) * Ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xgb", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--testlike", default="g0.27m0.04d0")
    args = ap.parse_args()
    X, Q = np.load(args.xgb, allow_pickle=True), np.load(args.seq, allow_pickle=True)
    ytr = pd.read_csv(DATA / "train_labels.csv", index_col=0)["target_next_recurring_merchant"]
    yva = pd.read_csv(DATA / "valid_labels.csv", index_col=0)["target_next_recurring_merchant"]
    test_ids = pd.read_csv(DATA / "sample_submission.csv")["client_id"]

    hold = pd.Index(Q["hold_ids"])
    oof = pd.DataFrame(X["oof"], index=X["oof_ids"]).loc[hold].to_numpy()
    yh = ytr.loc[hold]
    Px = {"hold": xgb_dist(oof, X["thr"]), "va": xgb_dist(X["va"], X["thr"]), "tl": xgb_dist(X["tl"], X["thr"]),
          "te": xgb_dist(X["te"], X["thr"])}
    Ps = {"hold": Q["hold"], "va": Q["va"], "tl": Q["tl"], "te": Q["te"]}
    seg = {"hold": segments(load("train"), hold), "va": segments(load("valid"), yva.index),
           "tl": segments(augment(load("valid"), args.testlike, seed=7), yva.index), "te": segments(load("test"), test_ids)}
    print(f"held-out tuning clients: {len(hold)} (segment sizes {np.bincount(seg['hold']).tolist()})")

    def fit(gates):
        best = (-1, None, None)
        for a in gates:
            P = blend(Px["hold"], Ps["hold"], a, seg["hold"])
            b = tune_bias(P, yh)
            f = macro_f1(yh, np.array(LABELS)[(P * b).argmax(1)])
            if f > best[0]:
                best = (f, a, b)
        return best

    def evaluate(name, a, b):
        out = {}
        for k, y in [("va", yva), ("tl", yva)]:
            P = blend(Px[k], Ps[k], a, seg[k])
            out[k] = macro_f1(y, np.array(LABELS)[(P * b).argmax(1)])
        print(f"{name:34s} gate={a}  VALID {out['va']:.4f} | TEST-LIKE {out['tl']:.4f}")
        return out

    results = {}
    for name, gates in [("XGBoost only", [{0: 1.0, 1: 1.0}]), ("sequence model only", [{0: 0.0, 1: 0.0}]),
                        ("global blend", [{0: a, 1: a} for a in ALPHAS]),
                        ("segment-gated blend (MoE)", [{0: a0, 1: a1} for a0 in ALPHAS for a1 in ALPHAS])]:
        f, a, b = fit(gates)
        results[name] = (a, b, evaluate(name, a, b))

    name = max(["global blend", "segment-gated blend (MoE)"], key=lambda n: results[n][2]["tl"])
    a, b, _ = results[name]
    pred = np.array(LABELS)[(blend(Px["te"], Ps["te"], a, seg["te"]) * b).argmax(1)]
    sub = pd.DataFrame({"client_id": test_ids, "predicted_next_recurring_merchant": pred})
    assert sub["predicted_next_recurring_merchant"].isin(LABELS).all() and len(sub) == len(test_ids)
    sub.to_csv(OUT / "submission_blend.csv", index=False)
    print(f"wrote outputs/submission_blend.csv using '{name}' (not submitted); test mix:", sub.iloc[:, 1].value_counts().to_dict())


if __name__ == "__main__":
    main()
