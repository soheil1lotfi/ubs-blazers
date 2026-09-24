"""Sequence model: a small Transformer over each client's raw payment timeline.

Instead of hand-made stream features, every client becomes a time-ordered sequence of events
(subscription-MCC card payments and refunds before the cutoff, most recent 64). Each event carries:
days before cutoff, day of month, log amount, refund flag, MCC, description keyword family / generic /
retail flags, hour, and same-amount context (how many payments share its amount, gap to the previous one,
whether it is the latest of its amount). The Transformer reads the whole timeline and outputs the
8-class answer directly.

Same protocol as stream_model.py: trained on train labels + replays (train/unlabeled histories) + noisy
copies; 20% of train clients (with all their replays/copies) held out for early stopping and tuning the
per-class decision bias; valid scored once (+ test-like valid); test only predicted.

Usage: .venv/bin/python seq_model.py [--aug 0.3] [--epochs 12]
"""

import argparse
import json
import os
from datetime import datetime

os.environ["STREAM_NO_XGB"] = "1"  # never load xgboost next to torch compute (OpenMP clash, see stream_model.py)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from stream_model import (CUTOFF, DATA, FAMILIES, GENERIC, LABELS, OUT, RUNS, SUB_MCC, augment, keyword_family, load,
                          log_eval, macro_f1)

L = 64
NF = 25
DESC_COLS = list(range(12, 21))  # keyword family one-hot, generic flag, retail flag
RETAIL = r"shop|market|dining|coffee|hotel|pharmacy|ride|foods|store|booking"
MCC_IDX = {m: i for i, m in enumerate(SUB_MCC)}
FAM_IDX = {f: i for i, f in enumerate(FAMILIES)}
LAB_IDX = {l: i for i, l in enumerate(LABELS)}


def sequences(tx, clients, cutoff):
    """X [n, L, NF] float16 and padding mask [n, L] (True = empty); most recent L events, right-aligned."""
    clients = pd.Index(clients)
    ev = tx[(tx["timestamp"] < cutoff) & tx["type"].isin(["card_payment", "refund"]) & tx["mcc"].isin(SUB_MCC)
            & tx["client_id"].isin(clients)].copy()
    ev["la"] = np.log(ev["amount"])
    ev = ev.sort_values(["client_id", "la"])
    ev["sid"] = ((ev["client_id"] != ev["client_id"].shift()) | (ev["la"].diff() > 0.03)).cumsum()
    ev = ev.sort_values(["client_id", "timestamp"])
    g = ev.groupby("sid")
    n_same = g["sid"].transform("size").to_numpy()
    gap_prev = (g["timestamp"].diff().dt.total_seconds() / 86400).fillna(0).clip(upper=365).to_numpy()
    is_last = (g.cumcount(ascending=False) == 0).to_numpy()
    days = ((cutoff - ev["timestamp"]).dt.total_seconds() / 86400).to_numpy()
    desc = ev["description"].str.lower()
    generic = desc.str.match(GENERIC).to_numpy()
    kw = keyword_family(desc).map(FAM_IDX).to_numpy()
    dom = ev["timestamp"].dt.day.to_numpy()

    F = np.zeros((len(ev), NF), dtype=np.float32)
    F[:, 0] = days / 400
    F[:, 1] = np.log1p(days) / 6
    F[:, 2] = np.sin(2 * np.pi * dom / 31)
    F[:, 3] = np.cos(2 * np.pi * dom / 31)
    F[:, 4] = (ev["la"].to_numpy() - 3.5) / 1.2
    F[:, 5] = (ev["type"] == "refund").to_numpy()
    F[np.arange(len(ev)), 6 + ev["mcc"].map(MCC_IDX).to_numpy()] = 1
    km = ~np.isnan(kw) & ~generic
    F[np.flatnonzero(km), 12 + kw[km].astype(int)] = 1
    F[:, 19] = generic
    F[:, 20] = desc.str.contains(RETAIL).to_numpy()
    F[:, 21] = ev["timestamp"].dt.hour.to_numpy() / 24
    F[:, 22] = np.minimum(n_same, 20) / 10
    F[:, 23] = gap_prev / 60
    F[:, 24] = is_last

    ci = clients.get_indexer(ev["client_id"])
    pos = ev.groupby("client_id").cumcount(ascending=False).to_numpy()  # 0 = most recent event
    keep = pos < L
    X = np.zeros((len(clients), L, NF), dtype=np.float16)
    M = np.ones((len(clients), L), dtype=bool)
    X[ci[keep], L - 1 - pos[keep]] = F[keep]
    M[ci[keep], L - 1 - pos[keep]] = False
    return X, M


class SeqModel(nn.Module):
    def __init__(self, d=64, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(NF, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, len(LABELS)))

    def forward(self, x, pad):
        h = self.inp(x)
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1)
        pad = torch.cat([torch.zeros(len(pad), 1, dtype=torch.bool), pad], 1)
        return self.head(self.enc(h, src_key_padding_mask=pad)[:, 0])


class GRUModel(nn.Module):
    """Recurrent alternative: a GRU reads the payments in time order; its last state + masked mean-pool feed the head."""
    def __init__(self, d=48):
        super().__init__()
        self.inp = nn.Linear(NF, d)
        self.rnn = nn.GRU(d, d, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, len(LABELS)))

    def forward(self, x, pad):
        h, _ = self.rnn(torch.relu(self.inp(x)))         # sequences are right-aligned: last step = most recent event
        keep = (~pad).unsqueeze(-1).float()
        mean = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
        return self.head(torch.cat([h[:, -1], mean], 1))


def predict_proba(model, X, M, bs=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.softmax(model(torch.from_numpy(X[i:i + bs].astype(np.float32)),
                                           torch.from_numpy(M[i:i + bs])), 1).numpy())
    return np.concatenate(out)


def tune_bias(P, y, rounds=3):
    b = np.ones(P.shape[1])
    grid = np.arange(0.3, 3.01, 0.05)
    for _ in range(rounds):
        for j in range(len(b)):
            def f(v):
                bb = b.copy(); bb[j] = v
                return macro_f1(y, np.array(LABELS)[(P * bb).argmax(1)])
            b[j] = max(grid, key=f)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="0.3", help="noise spec of the augmented copies / replays to include")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--pseudo_weight", type=float, default=0.5)
    ap.add_argument("--testlike", default="g0.27m0.04d0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=-1, help="0-4: hold out this fold of a fixed 5-fold split (OOF for all train)")
    ap.add_argument("--arch", choices=["transformer", "gru"], default="transformer")
    ap.add_argument("--no_desc", action="store_true", help="zero all description-derived inputs (robust to test's noise)")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(int(os.environ.get("SEQ_THREADS", "8")))
    tag = f"seq_{args.arch}_aug{args.aug}_s{args.seed}{'_nodesc' if args.no_desc else ''}{f'_f{args.fold}' if args.fold >= 0 else ''}"
    if args.no_desc:
        global sequences
        _seq = sequences
        def sequences(*a):
            X, M = _seq(*a)
            X[:, :, DESC_COLS] = 0
            return X, M
    writer = SummaryWriter(RUNS / f"{datetime.now():%m%d-%H%M%S}_{tag}")
    writer.add_text("config", json.dumps(vars(args)))

    ytr = pd.read_csv(DATA / "train_labels.csv", index_col=0)["target_next_recurring_merchant"]
    yva = pd.read_csv(DATA / "valid_labels.csv", index_col=0)["target_next_recurring_merchant"]
    test_ids = pd.read_csv(DATA / "sample_submission.csv")["client_id"]
    tx = {"train": load("train"), "unlabeled_pretrain": load("unlabeled_pretrain")}

    parts = []  # (X, M, y, weight, owner)
    Xr, Mr = sequences(tx["train"], ytr.index, CUTOFF)
    parts.append((Xr, Mr, ytr.values, 1.0, ytr.index.to_numpy()))
    X, M = sequences(augment(tx["train"], args.aug, seed=0), ytr.index, CUTOFF)
    parts.append((X, M, ytr.values, 1.0, ytr.index.to_numpy()))
    for split in ["train", "unlabeled_pretrain"]:
        for aug in [False, True]:
            _, Y = pd.read_pickle(OUT / (f"pseudo_{split}_aug{args.aug}.pkl" if aug else f"pseudo_{split}.pkl"))
            t = augment(tx[split], args.aug, seed=1) if aug else tx[split]
            ids = Y.index.str.replace("#aug", "", regex=False)
            owner, code = ids.str.split("@").str[0], ids.str.split("@").str[1]
            for c in sorted(set(code)):
                sel = np.asarray(code == c)
                X, M = sequences(t, owner[sel], pd.Timestamp(f"{c}-01", tz="UTC"))
                parts.append((X, M, Y.values[sel], args.pseudo_weight, np.asarray(owner[sel])))
            print(f"built {split} replays (aug={aug})", flush=True)

    X = np.concatenate([p[0] for p in parts]); M = np.concatenate([p[1] for p in parts])
    y = np.concatenate([[LAB_IDX[v] for v in p[2]] for p in parts]).astype(np.int64)
    w = np.concatenate([np.full(len(p[2]), p[3], dtype=np.float32) for p in parts])
    own = np.concatenate([p[4] for p in parts])

    # hold out 20% of real train clients (and every replay / copy of them)
    if args.fold >= 0:
        from sklearn.model_selection import StratifiedKFold
        hold = ytr.index[list(StratifiedKFold(5, shuffle=True, random_state=0).split(ytr.index, ytr))[args.fold][1]]
    else:
        _, hold = train_test_split(ytr.index, test_size=0.2, stratify=ytr, random_state=args.seed)
    tr_mask = ~np.isin(own, hold)
    hmask = ytr.index.isin(hold)
    Xh, Mh, yh = Xr[hmask], Mr[hmask], ytr[hmask]
    Xt, Mt, yt, wt = X[tr_mask], M[tr_mask], y[tr_mask], w[tr_mask]
    print(f"training sequences: {len(Xt)} | held-out real train clients: {len(yh)}", flush=True)

    counts = np.bincount(yt, weights=wt, minlength=len(LABELS))
    cw = torch.tensor(counts.sum() / (len(LABELS) * counts), dtype=torch.float32)
    model = SeqModel() if args.arch == "transformer" else GRUModel()
    print(f"{args.arch}: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.epochs * int(np.ceil(len(Xt) / 512)))
    ce = nn.CrossEntropyLoss(weight=cw, reduction="none")
    best, best_state = -1, None
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(Xt))
        tot = 0.0
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            logits = model(torch.from_numpy(Xt[b].astype(np.float32)), torch.from_numpy(Mt[b]))
            wb = torch.from_numpy(wt[b])
            loss = (ce(logits, torch.from_numpy(yt[b])) * wb).sum() / wb.sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item() * len(b)
        Ph = predict_proba(model, Xh, Mh)
        f1h = macro_f1(yh, np.array(LABELS)[Ph.argmax(1)])
        writer.add_scalar("seq/train_loss", tot / len(perm), ep)
        writer.add_scalar("seq/holdout_macro_f1", f1h, ep)
        print(f"epoch {ep + 1:2d}  train loss {tot / len(perm):.4f}  held-out macro-F1 {f1h:.4f}", flush=True)
        if f1h > best:
            best, best_state = f1h, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    Ph = predict_proba(model, Xh, Mh)
    bias = tune_bias(Ph, yh)
    print(f"held-out macro-F1: argmax {best:.4f} | with tuned bias {macro_f1(yh, np.array(LABELS)[(Ph * bias).argmax(1)]):.4f}")

    Xv, Mv = sequences(load("valid"), yva.index, CUTOFF)
    Pv = predict_proba(model, Xv, Mv)
    f1v = log_eval(writer, "seq_valid", yva, np.array(LABELS)[(Pv * bias).argmax(1)])
    Xl, Ml = sequences(augment(load("valid"), args.testlike, seed=7), yva.index, CUTOFF)
    Pl = predict_proba(model, Xl, Ml)
    f1l = log_eval(writer, "seq_valid_testlike", yva, np.array(LABELS)[(Pl * bias).argmax(1)])
    print(f"VALID macro-F1 {f1v:.4f} | TEST-LIKE VALID {f1l:.4f}")

    Xe, Me = sequences(load("test"), test_ids, CUTOFF)
    Pe = predict_proba(model, Xe, Me)
    np.savez(OUT / f"seq_scores_{tag}.npz", hold_ids=np.array(yh.index), hold=Ph, va=Pv, tl=Pl, te=Pe, bias=bias)
    pred = np.array(LABELS)[(Pe * bias).argmax(1)]
    pd.DataFrame({"client_id": test_ids, "predicted_next_recurring_merchant": pred}).to_csv(OUT / f"submission_{tag}.csv", index=False)
    print("test prediction mix:", pd.Series(pred).value_counts().to_dict())
    writer.add_hparams({"model": args.arch, "aug": args.aug, "seed": args.seed},
                       {"hparam/valid_macro_f1": f1v, "hparam/valid_testlike_macro_f1": f1l}, run_name=".")
    writer.close()


if __name__ == "__main__":
    main()
