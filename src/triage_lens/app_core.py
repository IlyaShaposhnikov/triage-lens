"""Logic behind the demo app, kept free of Streamlit so it can be tested.

``DemoAssets`` loads what ``demo_data`` published (from a local folder or the
Hugging Face dataset) and answers two questions about a new issue:

* ``classify`` — probabilities of bug / feature / question / docs from the
  TF-IDF + logistic regression classifier;
* ``similar`` — the most similar earlier issues by cosine similarity of the
  search vectors.

The demo searches with exact cosine similarity in numpy. The evaluation showed
that the same vectors in a Chroma HNSW index return the same results (recall@10
within 0.2 pp), and exact search on ~40k vectors takes milliseconds, so the free
hosting tier is spared the extra service.

The query text is cleaned exactly like the indexed texts (``dataset.cleaned``),
so form boilerplate pasted along with an issue does not distort the match.
"""

from __future__ import annotations

import gzip
import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from triage_lens.dataset import CLASSES, cleaned

Encoder = Callable[[list[str]], np.ndarray]


@dataclass
class RepoAssets:
    issues: list[dict[str, Any]]
    vectors: np.ndarray  # float32, L2-normalised, same order as issues
    classifier: Any
    info: dict[str, Any]


def query_text(repo: str, title: str, body: str) -> str:
    return cleaned(repo, {"title": title, "body": body})[2]


class DemoAssets:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["scikit_learn"] != sklearn.__version__:
            warnings.warn(
                f"classifier was saved with scikit-learn {self.manifest['scikit_learn']}, "
                f"running {sklearn.__version__}; predictions may differ",
                stacklevel=2,
            )
        self.repos = {repo: self._load(repo, info) for repo, info in self.manifest["repos"].items()}

    @classmethod
    def from_hub(cls, repo_id: str, revision: str | None = None) -> DemoAssets:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision)
        return cls(Path(path))

    def _load(self, repo: str, info: dict[str, Any]) -> RepoAssets:
        folder = self.root / repo.replace("/", "__")
        with gzip.open(folder / "issues.jsonl.gz", "rt", encoding="utf-8") as f:
            issues = [json.loads(line) for line in f if line.strip()]
        vectors = np.load(self.root / info["vectors_file"]).astype(np.float32)
        if len(vectors) != len(issues):
            raise ValueError(f"{repo}: {len(vectors)} vectors for {len(issues)} issues")
        classifier = joblib.load(folder / "classifier.joblib")
        return RepoAssets(issues, vectors, classifier, info)

    @property
    def encoder_name(self) -> str:
        return self.manifest["encoder"]

    def classify(self, repo: str, title: str, body: str) -> dict[str, float]:
        """Class probabilities in CLASSES order."""
        classifier = self.repos[repo].classifier
        probs = classifier.predict_proba([query_text(repo, title, body)])[0]
        by_class = dict(zip(classifier.classes_, probs, strict=True))
        return {cls: float(by_class[cls]) for cls in CLASSES}

    def similar(
        self, repo: str, title: str, body: str, encode: Encoder, k: int = 5
    ) -> list[dict[str, Any]]:
        """The ``k`` most similar issues, best first, each with its similarity."""
        assets = self.repos[repo]
        vector = np.asarray(encode([query_text(repo, title, body)])[0], dtype=np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        scores = assets.vectors @ vector
        k = min(k, len(scores))
        best = np.argpartition(-scores, k - 1)[:k]
        best = best[np.argsort(-scores[best], kind="stable")]
        return [{**assets.issues[i], "similarity": float(scores[i])} for i in best]


def load_encoder(model_name: str) -> Encoder:
    """Sentence-transformers encoder on CPU (imported lazily: it pulls in torch)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    return encode
