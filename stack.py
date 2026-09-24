"""Stacking meta-model (Instacart-style None handling + learned mixture of experts).

Level 1: experts' out-of-fold 8-way distributions on the 2,000 train clients (tree/race stream models, competing
risks, and the 5-fold Transformer when available) plus their valid / test-like / test predictions.
Level 2: a small model learns the final 8-way answer from
  - every expert's distribution,
  - summary statistics of the expert consensus (max, 2nd max, margin, entropy, mean P(none), disagreement),
  - a few robust client time features (activity trend, salary rhythm, number of alive streams, race P(none)).
It is fit with 5-fold CV on train OOF (so its own decision bias is tuned on out-of-fold meta predictions), then
applied to valid, test-like valid and test. Nothing is fit on valid.

Usage: .venv/bin/python stack.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ensemble import DATA, LABELS, OUT, score, to_dist, tune_bias
from stream_model import augment, build_candidates, load

EXPERTS = {"xgb_d6": "xgb_scores_tuned_d6.npz", "xgb_v8": "xgb_scores_v8_race_pseudo-both0.5_aug0.3.npz",
           "lgbm": "xgb_scores_v6_lgbm_pseudo-both0.5_aug0.3.npz", "risk": "risk_scores.npz"}
FOLDS = "seq_scores_seq_transformer_aug0.3_s0_nodesc_f{}.npz"
CLIENT_COLS = ["cl_activity_ratio", "cl_days_since_any", "cl_decay_act", "cl_salary_days_since", "cl_salary_overdue",
               "cl_salary_n90", "cl_sub_trend", "cl_refund_share", "cl_sub_refunds_90d"]


def client_table(tx, ids):
    c = build_candidates(tx)
    f = c.groupby("client_id")[CLIENT_COLS].first()
    f["n_alive"] = c[c["alive"] == 1].groupby("client_id")["sid"].nunique()
    f["n_streams"] = c.groupby("client_id")["sid"].nunique()
    f["best_proj_due"] = c[c["alive"] == 1].groupby("client_id")["proj_due"].min()
    return f.reindex(ids).fillna({"n_alive": 0, "n_streams": 0}).fillna(-1)


def meta_features(dists, client):
    X = [np.concatenate(dists, 1)]
    avg = np.mean(dists, 0)
    srt = np.sort(avg[:, :-1], 1)
    ent = -(avg * np.log(avg + 1e-9)).sum(1, keepdims=True)
    agree = np.mean([d.argmax(1) == avg.argmax(1) for d in dists], 0)[:, None]
    X += [srt[:, -1:], srt[:, -2:-1], srt[:, -1:] - srt[:, -2:-1], avg[:, -1:], ent, agree,
          np.std([d[:, -1] for d in dists], 0)[:, None], client.to_numpy()]
    return np.concatenate(X, 1)


def main():
    ytr = pd.read_csv(DATA / "train_labels.csv", index_col=0)["target_next_recurring_merchant"]
    yva = pd.read_csv(DATA / "valid_labels.csv", index_col=0)["target_next_recurring_merchant"]
    test_ids = pd.read_csv(DATA / "sample_submission.csv")["client_id"]
    D = {s: [] for s in ["oof", "va", "tl", "te"]}
    names = []
    for name, f in EXPERTS.items():
        z = np.load(OUT / f, allow_pickle=True)
        D["oof"].append(pd.DataFrame(to_dist(z, "oof"), index=z["oof_ids"]).loc[ytr.index].to_numpy())
        for s in ["va", "tl", "te"]:
            D[s].append(to_dist(z, s))
        names.append(name)
    if all((OUT / FOLDS.format(k)).exists() for k in range(5)):
        fz = [np.load(OUT / FOLDS.format(k), allow_pickle=True) for k in range(5)]
        D["oof"].append(pd.concat([pd.DataFrame(z["hold"], index=z["hold_ids"]) for z in fz]).loc[ytr.index].to_numpy())
        for s in ["va", "tl", "te"]:
            D[s].append(np.mean([z[s] for z in fz], 0))
        names.append("transformer5f")
    print("experts:", names)

    cl = {"oof": client_table(load("train"), ytr.index), "va": client_table(load("valid"), yva.index),
          "tl": client_table(augment(load("valid"), "g0.27m0.04d0", seed=7), yva.index),
          "te": client_table(load("test"), test_ids)}
    X = {s: meta_features(D[s], cl[s]) for s in D}
    y = pd.Index(LABELS).get_indexer(ytr)
    sc = StandardScaler().fit(X["oof"])
    Xs = {s: sc.transform(X[s]) for s in X}

    best = None
    for C in [0.03, 0.1, 0.3, 1.0]:
        oof = np.zeros((len(y), len(LABELS)))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs["oof"], y):
            m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xs["oof"][tr], y[tr])
            oof[va] = m.predict_proba(Xs["oof"][va])
        b = tune_bias(oof, ytr)
        f = score(oof, b, ytr)
        print(f"  meta logistic C={C}: train OOF (meta-level) {f:.4f}", flush=True)
        if best is None or f > best[0]:
            best = (f, C, b)
    f, C, b = best
    m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xs["oof"], y)
    P = {s: m.predict_proba(Xs[s]) for s in ["va", "tl", "te"]}
    print(f"STACK (C={C}): train OOF {f:.4f} | VALID {score(P['va'], b, yva):.4f} | TEST-LIKE {score(P['tl'], b, yva):.4f}")
    pred = np.array(LABELS)[(P["te"] * b).argmax(1)]
    sub = pd.DataFrame({"client_id": test_ids, "predicted_next_recurring_merchant": pred})
    assert sub["predicted_next_recurring_merchant"].isin(LABELS).all() and len(sub) == len(test_ids)
    sub.to_csv(OUT / f"submission_stack_{len(names)}experts.csv", index=False)
    print(f"wrote outputs/submission_stack_{len(names)}experts.csv (not submitted); test mix:", sub.iloc[:, 1].value_counts().to_dict())


if __name__ == "__main__":
    main()
