"""Hyperparameter search for the XGBoost stream model on cached data (stream_model.py --dump_cache).

Same clean protocol as stream_model.main: 5-fold out-of-fold on train (owner-grouped, replays of held-out
clients excluded) -> decision tuned on train OOF -> full fit -> valid and test-like valid scored once per config.
The config is chosen on test-like valid (model selection, as the README allows); the best config's scores are
saved in the expert format used by ensemble.py.

Usage: .venv/bin/python tune.py outputs/cache_<tag>.pkl [out.npz] [config-name-substring]
"""

import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from stream_model import FAMILIES, OUT, client_scores, decide, macro_f1, owner, tune_decision

BASE = dict(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=5, subsample=0.8,
            colsample_bytree=0.7, reg_lambda=5)
CONFIGS = {
    "base (d4, 600)": {},
    "deeper (d6, mcw10)": dict(max_depth=6, min_child_weight=10),
    "shallow long (d3, 1000)": dict(max_depth=3, n_estimators=1000),
    "d5 slow, col .5": dict(max_depth=5, learning_rate=0.02, n_estimators=1200, colsample_bytree=0.5),
    "d4 slow, strong reg": dict(learning_rate=0.02, n_estimators=1200, min_child_weight=20, reg_lambda=10),
    "d6 fast, gamma": dict(max_depth=6, learning_rate=0.05, n_estimators=400, min_child_weight=20, gamma=1.0),
}


def model(p):
    return xgb.XGBClassifier(**{**BASE, **p}, n_jobs=8, random_state=0, eval_metric="logloss")


def main():
    D = pd.read_pickle(sys.argv[1])
    lab, C, C_tl, cols = D["lab"], D["C"], D["C_tl"], D["cols"]
    P_rows, P_y, w_rows = D["P_rows"], D["P_y"], D["w_rows"]
    ytr, yva = lab["train"], lab["valid"]
    row_owner = owner(P_rows["client_id"])
    best = None
    only = sys.argv[3] if len(sys.argv) > 3 else None
    out_name = sys.argv[2] if len(sys.argv) > 2 else "xgb_scores_tuned.npz"
    for name, p in CONFIGS.items():
        if only and only not in name:
            continue
        S_oof = np.zeros((len(ytr), len(FAMILIES)))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(ytr.index, ytr):
            held = set(ytr.index[va])
            rm = ~np.isin(row_owner, list(held))
            m = model(p).fit(P_rows.loc[rm, cols], P_rows.loc[rm, "y"], sample_weight=w_rows[rm])
            ct = C["train"][C["train"]["client_id"].isin(held)]
            S_oof[va] = client_scores(ct, m.predict_proba(ct[cols])[:, 1], ytr.index[va])
        thr, bias = tune_decision(S_oof, ytr)
        m = model(p).fit(P_rows[cols], P_rows["y"], sample_weight=w_rows)
        S_va = client_scores(C["valid"], m.predict_proba(C["valid"][cols])[:, 1], yva.index)
        S_tl = client_scores(C_tl, m.predict_proba(C_tl[cols])[:, 1], yva.index)
        S_te = client_scores(C["test"], m.predict_proba(C["test"][cols])[:, 1], D["test_ids"])
        f_oof, f_va, f_tl = (macro_f1(ytr, decide(S_oof, thr, bias)), macro_f1(yva, decide(S_va, thr, bias)),
                             macro_f1(yva, decide(S_tl, thr, bias)))
        print(f"{name:26s} train OOF {f_oof:.4f} | VALID {f_va:.4f} | TEST-LIKE {f_tl:.4f}", flush=True)
        if best is None or f_tl > best[0]:
            best = (f_tl, name)
            np.savez(OUT / out_name, oof=S_oof, oof_ids=np.array(ytr.index), va=S_va, tl=S_tl, te=S_te,
                     thr=thr, bias=bias)
    print(f"best on test-like valid: {best[1]} ({best[0]:.4f}) -> outputs/{out_name}")


if __name__ == "__main__":
    main()
