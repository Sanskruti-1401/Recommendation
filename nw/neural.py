"""Word-level GRU language model: embedding -> GRU -> softmax over the vocabulary."""
import numpy as np
import torch
import torch.nn as nn

CONTEXT = 8


class GRULM(nn.Module):
    def __init__(self, V, emb=128, hid=256, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(hid, V)

    def forward(self, x):                       # x: (B, CONTEXT) token ids
        h, _ = self.gru(self.drop(self.emb(x)))
        return self.fc(self.drop(h[:, -1]))     # logits for the next word


def window(ctx_ids):
    ctx = ctx_ids[-CONTEXT:]
    return [0] * (CONTEXT - len(ctx)) + list(ctx)   # left-pad with <pad>


@torch.no_grad()
def probs(model, ctx_ids):
    model.eval()
    x = torch.tensor([window(ctx_ids)])
    return torch.softmax(model(x)[0], -1).numpy().astype(np.float64)
