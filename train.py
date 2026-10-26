"""Train both models on Gutenberg novels and save checkpoints + metrics.
   python get_data.py && python train.py
   python train.py --books carroll-alice austen-emma --epochs 8
"""
import argparse, json, math, os, pickle, time
import numpy as np
import torch
import torch.nn as nn
from nw.data import Vocab, clean, tokenize, DEFAULT_BOOKS
from nw.ngram import NGram
from nw.neural import GRULM, CONTEXT, window, probs as gru_probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--books", nargs="+", default=DEFAULT_BOOKS)
    p.add_argument("--vocab", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--bs", type=int, default=256)
    a = p.parse_args()
    torch.manual_seed(0); np.random.seed(0)

    toks = []
    for b in a.books:
        path = b if b.endswith(".txt") else f"data/gutenberg/{b}.txt"
        toks += tokenize(clean(open(path, encoding="latin-1").read()))
    vocab = Vocab.build(toks, a.vocab)
    ids = np.array(vocab.encode(toks))
    cut = int(len(ids) * 0.95)                       # last 5% of the text is held out
    train_ids, val_ids = ids[:cut], ids[cut:]
    print(f"tokens {len(ids):,} | vocab {len(vocab):,} | train {len(train_ids):,} | val {len(val_ids):,}")

    # ---------- trigram ----------
    ng = NGram(len(vocab)).fit(train_ids.tolist())
    rng = np.random.default_rng(0)
    vpos = rng.choice(np.arange(2, len(val_ids)), size=min(3000, len(val_ids) - 2), replace=False)

    def eval_fn(fn):
        top1 = top3 = nll = n = 0
        for i in vpos:
            tgt = val_ids[i]
            if not vocab.is_word(tgt): continue
            pr = fn(val_ids[max(0, i - CONTEXT):i].tolist())
            order = np.argsort(-np.where([vocab.is_word(j) for j in range(len(vocab))], pr, 0))[:3]
            top1 += order[0] == tgt; top3 += tgt in order
            nll += -math.log(max(pr[tgt], 1e-12)); n += 1
        return dict(top1=top1 / n, top3=top3 / n, perplexity=math.exp(nll / n))

    best, best_lam = None, None
    for lam in [(0.6, 0.3, 0.1), (0.5, 0.35, 0.15), (0.4, 0.4, 0.2), (0.3, 0.5, 0.2)]:
        ng.lam = lam; m = eval_fn(ng.probs)
        print("trigram lambdas", lam, {k: round(v, 3) for k, v in m.items()})
        if best is None or m["top3"] > best["top3"]: best, best_lam = m, lam
    ng.lam = best_lam
    pickle.dump(ng, open("checkpoints/trigram.pkl", "wb"))

    # ---------- GRU ----------
    pad = np.concatenate([np.zeros(CONTEXT, int), train_ids])
    X = np.stack([pad[i:i + CONTEXT] for i in range(len(train_ids))])     # (N, CONTEXT)
    Y = train_ids
    X, Y = torch.from_numpy(X), torch.from_numpy(Y)
    model = GRULM(len(vocab))
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, 3, 0.4)
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); perm = torch.randperm(len(X)); tot = 0
        for s in range(0, len(X), a.bs):
            idx = perm[s:s + a.bs]
            loss = lossf(model(X[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        m = eval_fn(lambda c: gru_probs(model, c))
        print(f"epoch {ep} | train loss {tot / len(X):.3f} | val top3 {m['top3']:.3f} ppl {m['perplexity']:.1f} | {time.time() - t0:.0f}s", flush=True)
    torch.save(model.state_dict(), "checkpoints/gru.pt")
    json.dump({"itos": vocab.itos}, open("checkpoints/vocab.json", "w"))
    gm = eval_fn(lambda c: gru_probs(model, c))
    json.dump({"trigram": best, "gru": gm, "tokens": int(len(ids)), "vocab": len(vocab),
               "books": a.books}, open("checkpoints/metrics.json", "w"), indent=1)
    print("trigram", best); print("gru", gm)


if __name__ == "__main__":
    main()
