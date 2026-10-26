"""Text cleaning, tokenisation, vocabulary."""
import re
from collections import Counter

PUNCT = ".,!?;:"
PAD, UNK = "<pad>", "<unk>"
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,!?;:]")
DEFAULT_BOOKS = ["carroll-alice", "austen-emma", "austen-persuasion", "austen-sense"]


def clean(raw: str) -> str:
    raw = re.sub(r"^\s*\[[^\]]*\]", " ", raw)                         # "[Emma by Jane Austen 1816]"
    raw = re.sub(r"(?im)^\s*(chapter|volume|book)\b.*$", " ", raw)     # headings
    raw = raw.replace("--", ", ").replace("_", "")
    return raw


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def detokenize(tokens):
    out = ""
    for t in tokens:
        out += t if (t in PUNCT and out) else (" " + t if out else t)
    return out


class Vocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, tokens, size=4000):
        words = [w for w, _ in Counter(tokens).most_common() if w not in PUNCT]
        return cls([PAD, UNK] + list(PUNCT) + words[: size - 2 - len(PUNCT)])

    def encode(self, tokens):
        u = self.stoi[UNK]
        return [self.stoi.get(t, u) for t in tokens]

    def __len__(self):
        return len(self.itos)

    def is_word(self, i):
        return i > 1 + len(PUNCT)      # excludes <pad>, <unk>, punctuation
