"""Mixture of experts by greedy ensemble selection (Caruana et al., 2004), then a sequence-model stage.

Stage 1 - experts with out-of-fold predictions on all 2,000 train clients (XGBoost, LightGBM, RandomForest,
          ExtraTrees stream models; competing-risks model). Greedy forward selection WITH replacement: repeatedly
          add the expert that most improves train macro-F1 (per-class bias re-tuned each time); the multiset of
          picks is the weighting.
Stage 2 - blend the stage-1 ensemble with the Transformer (only 400 held-out train clients) using one weight.
Valid and test-like valid are scored once for every expert and both stages. Nothing trains on valid.

Usage: .venv/bin/python ensemble.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from stream_model import DATA, LABELS, OUT

_L = np.array(LABELS)


def macro_f1(y, p):
    """Fast macro-F1 over the 8 fixed labels (numpy confusion counts; same value as sklearn's)."""
    yi = pd.Index(LABELS).get_indexer(np.asarray(y)); pi = pd.Index(LABELS).get_indexer(np.asarray(p))
    cm = np.bincount(yi * len(LABELS) + pi, minlength=len(LABELS) ** 2).reshape(len(LABELS), -1)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    return float(np.mean(np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0)))

TREES = {"xgboost": "xgb_scores_v6_pseudo-both0.5_aug0.3.npz", "xgboost + race": "xgb_scores_v7_race_pseudo-both0.5_aug0.3.npz", "xgboost + race + time": "xgb_scores_v8_race_pseudo-both0.5_aug0.3.npz", "xgboost tuned (d6)": "xgb_scores_tuned_d6.npz", "lightgbm": "xgb_scores_v6_lgbm_pseudo-both0.5_aug0.3.npz",
         "random forest": "xgb_scores_v6_rf_pseudo-both0.5_aug0.3.npz", "extra trees": "xgb_scores_v6_et_pseudo-both0.5_aug0.3.npz"}
RISK = "risk_scores.npz"
SEQ = "seq_scores_seq_transformer_aug0.3_s0_nodesc.npz"


def to_dist(z, key):
    """Tree experts store 7 family scores + a none threshold; the others store 8-way distributions."""
    M = z[key]
    if M.shape[1] == len(LABELS):
        return M
    P = np.concatenate([M, np.full((len(M), 1), float(z["thr"]))], 1)
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


def score(P, b, y):
    return macro_f1(y, np.array(LABELS)[(P * b).argmax(1)])


def main():
    ytr = pd.read_csv(DATA / "train_labels.csv", index_col=0)["target_next_recurring_merchant"]
    yva = pd.read_csv(DATA / "valid_labels.csv", index_col=0)["target_next_recurring_merchant"]
    test_ids = pd.read_csv(DATA / "sample_submission.csv")["client_id"]

    E = {}
    for name, f in list(TREES.items()) + [("competing risks", RISK)]:
        if not (OUT / f).exists():
            print(f"(skipping {name}: {f} not found)")
            continue
        z = np.load(OUT / f, allow_pickle=True)
        oof = pd.DataFrame(to_dist(z, "oof"), index=z["oof_ids"]).loc[ytr.index].to_numpy()
        E[name] = {"oof": oof, "va": to_dist(z, "va"), "tl": to_dist(z, "tl"), "te": to_dist(z, "te")}

    print(f"{'expert':28s} {'train':>7s} {'VALID':>7s} {'TEST-LIKE':>9s}")
    for name, e in E.items():
        b = tune_bias(e["oof"], ytr)
        print(f"{name:28s} {score(e['oof'], b, ytr):7.4f} {score(e['va'], b, yva):7.4f} {score(e['tl'], b, yva):9.4f}")

    # Stage 1: greedy selection with replacement on train OOF
    picks, cur, best = [], None, (-1, None, None)
    for step in range(12):
        cand = []
        for name, e in E.items():
            P = e["oof"] if cur is None else (cur * len(picks) + e["oof"]) / (len(picks) + 1)
            b = tune_bias(P, ytr)
            cand.append((score(P, b, ytr), name, P, b))
        f, name, P, b = max(cand, key=lambda x: x[0])
        picks.append(name); cur = P
        if f > best[0]:
            best = (f, list(picks), b)
        print(f"  step {step + 1:2d}: add {name:15s} train macro-F1 {f:.4f}")
    f1_tr, sel, b1 = best
    w = pd.Series(sel).value_counts() / len(sel)
    mix = lambda k: sum(w[n] * E[n][k] for n in w.index)
    print(f"stage 1 weights: {w.round(2).to_dict()}")
    print(f"stage 1 ensemble             {f1_tr:7.4f} {score(mix('va'), b1, yva):7.4f} {score(mix('tl'), b1, yva):9.4f}")

    # Stage 2: + Transformer, weight tuned on its 400 held-out train clients
    final = {"va": mix("va"), "tl": mix("tl"), "te": mix("te")}
    if (OUT / SEQ).exists():
        q = np.load(OUT / SEQ, allow_pickle=True)
        hold = pd.Index(q["hold_ids"])
        h1 = pd.DataFrame(mix("oof"), index=ytr.index).loc[hold].to_numpy()
        yh = ytr.loc[hold]
        cand = []
        for a in np.round(np.arange(0, 1.01, 0.1), 2):
            P = a * h1 + (1 - a) * q["hold"]
            b = tune_bias(P, yh)
            cand.append((score(P, b, yh), a, b))
        _, a, b2 = max(cand, key=lambda x: x[0])
        final = {k: a * mix(k) + (1 - a) * q[k] for k in ["va", "tl", "te"]}
        print(f"stage 2 (+Transformer, trees/risk weight {a})   -  {score(final['va'], b2, yva):7.4f} {score(final['tl'], b2, yva):9.4f}")
    else:
        b2 = b1

    pred = np.array(LABELS)[(final["te"] * b2).argmax(1)]
    sub = pd.DataFrame({"client_id": test_ids, "predicted_next_recurring_merchant": pred})
    assert sub["predicted_next_recurring_merchant"].isin(LABELS).all() and len(sub) == len(test_ids)
    sub.to_csv(OUT / "submission_ensemble.csv", index=False)
    print("wrote outputs/submission_ensemble.csv (not submitted); test mix:", sub.iloc[:, 1].value_counts().to_dict())


if __name__ == "__main__":
    main()
