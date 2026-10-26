"""FastAPI backend + static frontend.   Run:  uvicorn server:app --port 8000"""
import json, pickle
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from nw.data import Vocab, tokenize, detokenize, PUNCT
from nw.neural import GRULM, probs as gru_probs

vocab = Vocab(json.load(open("checkpoints/vocab.json"))["itos"])
gru = GRULM(len(vocab)); gru.load_state_dict(torch.load("checkpoints/gru.pt", map_location="cpu")); gru.eval()
trigram = pickle.load(open("checkpoints/trigram.pkl", "rb"))
metrics = json.load(open("checkpoints/metrics.json"))
IS_WORD = np.array([vocab.is_word(i) for i in range(len(vocab))])
app = FastAPI(title="Next-word prediction")


def split_text(text: str):
    """-> (context token ids, typed prefix). Text ending in a letter means the last word is unfinished."""
    prefix = ""
    if text and text[-1].isalpha() or (text and text[-1] == "'"):
        words = text.split()
        prefix = words[-1].lower() if words else ""
        text = text[: len(text) - len(words[-1])] if words else ""
    return vocab.encode(tokenize(text)), prefix


def distribution(model, ctx):
    return gru_probs(gru, ctx) if model == "gru" else trigram.probs(ctx)


def mask_prefix(p, prefix, words_only=True):
    p = p.copy()
    if words_only: p[~IS_WORD] = 0
    if prefix:
        keep = np.array([w.startswith(prefix) for w in vocab.itos])
        p[~keep] = 0
    s = p.sum()
    return p / s if s > 0 else p


def top_k(p, k):
    order = np.argsort(-p)[:k]
    return [{"word": vocab.itos[i], "prob": round(float(p[i]), 4)} for i in order if p[i] > 0]


class PredictReq(BaseModel):
    text: str = ""
    model: str = "gru"
    k: int = 3


class GenReq(BaseModel):
    text: str = ""
    model: str = "gru"
    length: int = 12
    temperature: float = 0.8
    top_k: int = 10


@app.get("/api/info")
def info():
    return metrics


@app.post("/api/predict")
def predict(r: PredictReq):
    ctx, prefix = split_text(r.text)
    out = {}
    for m in ("gru", "trigram"):
        p = mask_prefix(distribution(m, ctx), prefix)
        out[m] = top_k(p, max(1, min(r.k, 8)))
    return {"suggestions": out[r.model if r.model in out else "gru"], "other": out, "prefix": prefix}


@app.post("/api/generate")
def generate(r: GenReq):
    ctx, base = vocab.encode(tokenize(r.text)), r.text      # continue after the last word
    temp = max(0.05, min(r.temperature, 2.0))
    rng = np.random.default_rng()
    new = []
    for step in range(max(1, min(r.length, 60))):
        p = distribution(r.model, ctx)
        p[:2] = 0                                                     # never emit <pad>/<unk>
        if step > 0 and new and new[-1] in PUNCT: p[2:2 + len(PUNCT)] = 0     # no ".,"
        if step == 0: p[2:2 + len(PUNCT)] = 0
        logits = np.log(np.maximum(p, 1e-12)) / temp
        k = max(1, min(r.top_k, len(logits)))
        cut = np.sort(logits)[-k]
        logits[logits < cut] = -np.inf
        q = np.exp(logits - logits.max()); q /= q.sum()
        i = int(rng.choice(len(q), p=q))
        new.append(vocab.itos[i]); ctx = ctx + [i]
        if vocab.itos[i] in ".!?" and step >= 4: break
    piece = _capitalise(detokenize(new), start=(not base.strip()) or base.rstrip()[-1] in ".!?")
    if not base or base.endswith((" ", "\n")) or piece[0] in PUNCT:
        joined = base + piece
    else:
        joined = base + " " + piece
    return {"text": joined, "generated": piece}


def _capitalise(s, start):
    out, cap = [], start
    for ch in s:
        if cap and ch.isalpha(): ch = ch.upper(); cap = False
        elif ch in ".!?": cap = True
        out.append(ch)
    return "".join(out).replace(" i ", " I ").replace(" i'", " I'")


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
