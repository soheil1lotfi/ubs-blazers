"""Self-supervised labels: re-run the task at earlier cutoffs on histories we are allowed to train on.

Only sources allowed for training are used: unlabeled_pretrain and train histories
(never valid/test). At pseudo-cutoff T we build candidates from data < T exactly like at the real cutoff, then look at
[T, T+90d): the first subscription-MCC card payment that matches an existing stream (amount within
tol) gives the pseudo label = that stream's top family; no match -> `none`.
No real labels are used.

Usage: .venv/bin/python pseudo_labels.py [aug_level] [splits]   (writes outputs/pseudo_<split>[_aug<level>].pkl)
"""

import numpy as np
import pandas as pd

import sys

from stream_model import OUT, SUB_MCC, augment, build_candidates, load

HORIZON = pd.Timedelta(days=90)
CUTOFFS = [pd.Timestamp(d, tz="UTC") for d in ["2025-07-01", "2025-08-01", "2025-09-01", "2025-10-01"]]


def pseudo_label(tx, T, tol=0.03):
    c = build_candidates(tx, cutoff=T, tol=tol)
    top = c.sort_values(["sid", "fam_vote_rank", "fam_vote"], ascending=[True, True, False]).drop_duplicates("sid")
    fut = tx[(tx["timestamp"] >= T) & (tx["timestamp"] < T + HORIZON) & (tx["type"] == "card_payment")
             & tx["mcc"].isin(SUB_MCC)]
    m = fut.merge(top[["client_id", "sid", "amount", "family"]], on="client_id", suffixes=("", "_s"))
    m = m[(np.log(m["amount"]) - np.log(m["amount_s"])).abs() <= tol]
    first = m.sort_values("timestamp").drop_duplicates("client_id").set_index("client_id")["family"]
    clients = tx.loc[tx["timestamp"] < T, "client_id"].unique()
    y = first.reindex(clients).fillna("none")
    c = c.copy()
    c["y"] = (c["family"] == c["client_id"].map(y)).astype(int)
    c["client_id"] = c["client_id"] + "@" + T.strftime("%Y-%m")   # one pseudo-client per (client, cutoff)
    y.index = y.index + "@" + T.strftime("%Y-%m")
    return c, y


def main():
    level = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] not in ("0", "") else ""   # optional noise spec
    # valid is only replayed for the --use_valid setup (valid used as training data); test is never replayed
    splits = sys.argv[2].split(",") if len(sys.argv) > 2 else ["train", "unlabeled_pretrain"]
    assert "test" not in splits, "test histories are for predictions only"
    for split in splits:
        tx = load(split)
        if level:
            tx = augment(tx, level, seed=1)
        parts = [pseudo_label(tx, T) for T in CUTOFFS]
        C = pd.concat([p[0] for p in parts], ignore_index=True)
        Y = pd.concat([p[1] for p in parts])
        if level:  # distinct ids so augmented copies never collide with the clean ones
            C["client_id"] = C["client_id"] + "#aug"
            Y.index = Y.index + "#aug"
        pd.to_pickle((C, Y), OUT / (f"pseudo_{split}_aug{level}.pkl" if level else f"pseudo_{split}.pkl"))
        print(f"[{split}] {Y.size} pseudo-clients, {len(C)} rows | label mix %:",
              (Y.value_counts(normalize=True) * 100).round(1).to_dict(), flush=True)


if __name__ == "__main__":
    main()
