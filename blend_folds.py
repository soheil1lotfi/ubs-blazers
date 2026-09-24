"""Blend tree/race experts with the 5-fold Transformer, tuned on all 2,000 train clients.

The 5 fold models give out-of-fold Transformer probabilities for every train client (and an average of 5 models
for valid/test). Every choice - which tree/race experts to average, the Transformer weight alpha, the per-class
bias - is made on train out-of-fold macro-F1 only. Valid and test-like valid are then reported for transparency.

Usage: .venv/bin/python blend_folds.py
"""

import itertools

import numpy as np
import pandas as pd

from ensemble import DATA, LABELS, OUT, score, to_dist, tune_bias

EXPERTS = {"xgb tuned d6": "xgb_scores_tuned_d6.npz", "xgb v8 race+time": "xgb_scores_v8_race_pseudo-both0.5_aug0.3.npz",
           "lightgbm": "xgb_scores_v6_lgbm_pseudo-both0.5_aug0.3.npz", "competing risks": "risk_scores.npz"}
FOLDS = "seq_scores_seq_transformer_aug0.3_s0_nodesc_f{}.npz"


def main():
    ytr = pd.read_csv(DATA / "train_labels.csv", index_col=0)["target_next_recurring_merchant"]
    yva = pd.read_csv(DATA / "valid_labels.csv", index_col=0)["target_next_recurring_merchant"]
    test_ids = pd.read_csv(DATA / "sample_submission.csv")["client_id"]

    fz = [np.load(OUT / FOLDS.format(k), allow_pickle=True) for k in range(5)]
    T = {"oof": pd.concat([pd.DataFrame(z["hold"], index=z["hold_ids"]) for z in fz]).loc[ytr.index].to_numpy(),
         **{s: np.mean([z[s] for z in fz], 0) for s in ["va", "tl", "te"]}}
    b = tune_bias(T["oof"], ytr)
    print(f"Transformer 5-fold: train OOF {score(T['oof'], b, ytr):.4f} | VALID {score(T['va'], b, yva):.4f} | "
          f"TEST-LIKE {score(T['tl'], b, yva):.4f}")

    E = {}
    for name, f in EXPERTS.items():
        z = np.load(OUT / f, allow_pickle=True)
        E[name] = {"oof": pd.DataFrame(to_dist(z, "oof"), index=z["oof_ids"]).loc[ytr.index].to_numpy(),
                   **{s: to_dist(z, s) for s in ["va", "tl", "te"]}}

    results = []
    for r in range(1, len(E) + 1):
        for combo in itertools.combinations(E, r):
            trees = {s: np.mean([E[n][s] for n in combo], 0) for s in ["oof", "va", "tl", "te"]}
            for a in np.round(np.arange(0.4, 1.01, 0.1), 2):
                P = {s: a * trees[s] + (1 - a) * T[s] for s in ["oof", "va", "tl", "te"]}
                bb = tune_bias(P["oof"], ytr)
                results.append((score(P["oof"], bb, ytr), combo, a, bb, P))
    results.sort(key=lambda x: -x[0])
    print(f"\n{'experts (+Transformer)':58s} {'alpha':>5s} {'trainOOF':>8s} {'VALID':>7s} {'TEST-LIKE':>9s}")
    for f, combo, a, bb, P in results[:8]:
        print(f"{' + '.join(combo):58s} {a:5.1f} {f:8.4f} {score(P['va'], bb, yva):7.4f} {score(P['tl'], bb, yva):9.4f}")
    f, combo, a, bb, P = results[0]
    pred = np.array(LABELS)[(P["te"] * bb).argmax(1)]
    sub = pd.DataFrame({"client_id": test_ids, "predicted_next_recurring_merchant": pred})
    assert sub["predicted_next_recurring_merchant"].isin(LABELS).all() and len(sub) == len(test_ids)
    sub.to_csv(OUT / "submission_blend_folds.csv", index=False)
    print(f"\nchosen on train OOF: {' + '.join(combo)} + Transformer, alpha {a} -> VALID {score(P['va'], bb, yva):.4f} "
          f"TEST-LIKE {score(P['tl'], bb, yva):.4f}; wrote outputs/submission_blend_folds.csv (not submitted); test mix:",
          sub.iloc[:, 1].value_counts().to_dict())


if __name__ == "__main__":
    main()
