"""Price clustering on the unlabeled data: P(family | stream amount).

Uses only unlabeled_pretrain transactions (no labels): streams whose family is unambiguous from MCC +
keywords (vote share >= 0.75; 5812 streams with a music/streaming keyword reach ~0.83) are binned by log-amount; per-bin family frequencies (Laplace-smoothed)
give a price-based family probability that is independent of the (noisy) descriptions.

Usage: .venv/bin/python price_prior.py   (writes outputs/price_prior.npz)
"""

import numpy as np

from stream_model import FAMILIES, OUT, build_candidates, load

EDGES = np.linspace(np.log(1.0), np.log(1000.0), 61)


def main():
    c = build_candidates(load("unlabeled_pretrain"))
    ref = c[(c["fam_vote"] >= 0.75) & (c["n"] >= 3)].drop_duplicates("sid")
    counts = np.zeros((len(EDGES) - 1, len(FAMILIES)))
    b = np.clip(np.digitize(np.log(ref["amount"]), EDGES) - 1, 0, len(EDGES) - 2)
    np.add.at(counts, (b, ref["fam_id"].to_numpy().astype(int)), 1)
    # smooth across neighbouring bins, then Laplace
    k = np.array([0.25, 0.5, 0.25])
    sm = np.stack([np.convolve(counts[:, j], k, mode="same") for j in range(len(FAMILIES))], axis=1)
    prob = (sm + 0.5) / (sm + 0.5).sum(1, keepdims=True)
    np.savez(OUT / "price_prior.npz", edges=EDGES, prob=prob)
    print(f"{len(ref)} reference streams from unlabeled clients")
    for amt in [5, 10, 15, 20, 40, 70, 150]:
        i = np.clip(np.digitize(np.log(amt), EDGES) - 1, 0, len(EDGES) - 2)
        top = np.argsort(-prob[i])[:3]
        print(f"  amount {amt:>4}: " + ", ".join(f"{FAMILIES[j]} {prob[i, j]:.2f}" for j in top))


if __name__ == "__main__":
    main()
