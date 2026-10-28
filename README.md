# triage-lens

Automatic triage for GitHub issues: predict what kind of issue it is and surface
likely duplicates before a maintainer has to read it.

> **Status:** work in progress.

## What it does

Given a new issue (title and body), triage-lens:

- predicts its type: **bug**, **feature**, **question** or **docs**, with a confidence score;
- returns the most similar past issues from the same project and flags likely duplicates.

## Data

Live issues collected through the GitHub REST API from two Python projects:

- [astral-sh/uv](https://github.com/astral-sh/uv) — primary source;
- [pandas-dev/pandas](https://github.com/pandas-dev/pandas) — second project, used to test
  whether models transfer across codebases.

Each project uses its own label names, mapped to four shared classes:

| Class    | uv              | pandas           |
|----------|-----------------|------------------|
| bug      | `bug`           | `Bug`            |
| feature  | `enhancement`   | `Enhancement`    |
| question | `question`      | `Usage Question` |
| docs     | `documentation` | `Docs`           |

Pull requests and bot-authored issues are excluded. Evaluation uses a time-based split:
models are trained on older issues and tested on the most recent ones, as they would be
in real use.

## Approaches

**Issue type classification** — three approaches compared on the same test set:

1. **Zero-shot** — a pretrained NLI model, no training at all;
2. **Embeddings + logistic regression** — sentence embeddings as features, trained on CPU;
3. **Fine-tuned transformer** — DistilBERT / ModernBERT, trained on a free cloud GPU.

**Duplicate search** — dense vector search vs. keyword search (BM25) vs. a hybrid of both,
evaluated with recall@k on known duplicate pairs.

## Questions this project answers

- How much does fine-tuning add over zero-shot and a cheap embedding baseline, and at what cost?
- How many labeled examples does it take to beat zero-shot?
- Does a model trained on one project work on another?
- Does semantic search find duplicates better than keyword search?

## Roadmap

- [ ] Issue collector with resumable downloads and rate-limit handling
- [ ] Dataset cleaning, label mapping, time-based split
- [ ] Zero-shot baseline
- [ ] Embeddings + logistic regression
- [ ] Fine-tuned transformer
- [ ] Duplicate pairs extraction and vector search
- [ ] Demo app
- [ ] Scheduled data refresh

## License

[MIT](LICENSE)