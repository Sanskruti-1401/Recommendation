# Next-word prediction keyboard

Two language models trained on Project Gutenberg novels, served through a phone-style typing UI.

| Model | Idea |
|---|---|
| Trigram | P(w \| a, b) from counts, interpolated with bigram and Laplace-smoothed unigram |
| Neural (GRU) | Embedding -> GRU -> softmax over a 4,000-word vocabulary, context of 8 words |

```
nw/data.py      cleaning, tokenising, vocabulary
nw/ngram.py     interpolated trigram model
nw/neural.py    GRU language model
train.py        trains both models, reports top-1 / top-3 / perplexity on held-out text
server.py       FastAPI: /api/predict, /api/generate, /api/info
static/         the UI
checkpoints/    trained models
```

## Run
```bash
pip install -r requirements.txt
uvicorn server:app --port 8000        # http://localhost:8000
```

## Retrain (optional)
```bash
python get_data.py                                   # downloads the novels into data/gutenberg/
python train.py                                      # ~12 min on one CPU core
python train.py --books carroll-alice austen-emma --epochs 8
python train.py --books my_novel.txt                 # any plain-text file
```
Held-out evaluation uses the last 5% of the combined text.
# Recommendation
