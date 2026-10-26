"""Interpolated trigram language model with Laplace-smoothed unigrams.

P(w | a, b) = l3 * C(a,b,w)/C(a,b)  +  l2 * C(b,w)/C(b)  +  l1 * (C(w)+1)/(N+V)
Weights of orders whose context was never seen are redistributed to the lower orders.
"""
import numpy as np
from collections import defaultdict


class NGram:
    def __init__(self, V, lambdas=(0.5, 0.35, 0.15)):
        self.V, self.lam = V, lambdas
        self.uni = np.zeros(V)
        self.bi, self.tri = {}, {}

    def fit(self, ids):
        bi, tri = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
        for i, w in enumerate(ids):
            self.uni[w] += 1
            if i >= 1: bi[ids[i - 1]][w] += 1
            if i >= 2: tri[(ids[i - 2], ids[i - 1])][w] += 1
        pack = lambda d: {k: (np.fromiter(v.keys(), int), np.fromiter(v.values(), float)) for k, v in d.items()}
        self.bi, self.tri = pack(bi), pack(tri)
        self.p1 = (self.uni + 1) / (self.uni.sum() + self.V)
        return self

    def probs(self, ctx):
        """ctx: list of token ids (uses the last two). Returns a dense distribution over the vocab."""
        l3, l2, l1 = self.lam
        p = np.zeros(self.V)
        t = self.tri.get(tuple(ctx[-2:])) if len(ctx) >= 2 else None
        b = self.bi.get(ctx[-1]) if len(ctx) >= 1 else None
        if t is None: l3 = 0.0
        if b is None: l2 = 0.0
        s = l1 + l2 + l3
        l1, l2, l3 = l1 / s, l2 / s, l3 / s
        p += l1 * self.p1
        if b is not None: p[b[0]] += l2 * b[1] / b[1].sum()
        if t is not None: p[t[0]] += l3 * t[1] / t[1].sum()
        return p / p.sum()
