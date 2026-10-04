"""Separate several voices inside one corpus (e.g. a game's villager, mayor and trader).

Each file gets a small voice fingerprint (average MFCC + median pitch); files are clustered and only
the chosen voice is used. The fitted model is stored in the state directory, so later zips are assigned
to the same voices without re-clustering.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio

F0_WEIGHT = 2.0
MIN_SILHOUETTE = 0.25     # below this the corpus is treated as a single voice
MIN_FILES = 20            # clustering is meaningless for a handful of files
FEATURE_SECONDS = 20.0


def fingerprint(wav16: Path) -> np.ndarray | None:
    """[19 mean MFCCs, log2 median f0] of the first seconds of speech, or None if unusable."""
    import torch
    import torchaudio
    x, _ = audio.read_wav(wav16, 0, FEATURE_SECONDS)
    if len(x) < 8000:
        return None
    f0 = audio.median_f0(x, audio.SR_REC)
    if not f0:
        return None
    mf = torchaudio.transforms.MFCC(sample_rate=audio.SR_REC, n_mfcc=20,
                                    melkwargs=dict(n_fft=512, hop_length=160, n_mels=40))
    e = audio.rms_envelope(x, audio.SR_REC, 10)
    m = mf(torch.from_numpy(x)).T.numpy()[:len(e)]
    m = m[e[:len(m)] > e.max() * 0.1]
    if len(m) < 10:
        return None
    return np.concatenate([m[:, 1:].mean(0), [np.log2(f0)]])


def _silhouette(X: np.ndarray, lab: np.ndarray, cap: int = 800) -> float:
    from scipy.spatial.distance import cdist
    idx = np.random.RandomState(0).choice(len(X), min(cap, len(X)), replace=False)
    Xs, ls = X[idx], lab[idx]
    D = cdist(Xs, Xs)
    s = []
    for i in range(len(Xs)):
        same = ls == ls[i]
        same[i] = False
        if not same.any():
            continue
        a = D[i][same].mean()
        b = min(D[i][ls == c].mean() for c in set(ls) if c != ls[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s)) if s else 0.0


@dataclass
class SpeakerModel:
    mean: np.ndarray
    std: np.ndarray
    centroids: np.ndarray          # clusters ordered by size, largest first
    sizes: list[int]
    silhouette: float

    def standardise(self, feats: np.ndarray) -> np.ndarray:
        z = (feats - self.mean) / self.std
        z[:, -1] *= F0_WEIGHT
        return z

    def assign(self, feats: np.ndarray) -> np.ndarray:
        z = self.standardise(feats)
        return np.argmin(((z[:, None, :] - self.centroids[None]) ** 2).sum(-1), axis=1)

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "centroids": self.centroids.tolist(),
                "sizes": self.sizes, "silhouette": self.silhouette}

    @classmethod
    def from_json(cls, d: dict) -> "SpeakerModel":
        return cls(np.array(d["mean"]), np.array(d["std"]), np.array(d["centroids"]), d["sizes"], d["silhouette"])


def fit(feats: np.ndarray, max_k: int = 4) -> SpeakerModel:
    from scipy.cluster.hierarchy import fcluster, linkage
    mean, std = feats.mean(0), feats.std(0) + 1e-9
    z = (feats - mean) / std
    z[:, -1] *= F0_WEIGHT
    tree = linkage(z, "ward")
    best = (0.0, np.zeros(len(z), int))
    for k in range(2, max_k + 1):
        lab = fcluster(tree, k, "maxclust") - 1
        if len(set(lab)) < 2:
            continue
        sil = _silhouette(z, lab)
        if sil > best[0]:
            best = (sil, lab)
    sil, lab = best
    if sil < MIN_SILHOUETTE:
        lab, sil = np.zeros(len(z), int), 0.0
    order = np.argsort([-(lab == c).sum() for c in sorted(set(lab))])
    remap = {c: i for i, c in enumerate(sorted(set(lab))[j] for j in order)}
    lab = np.array([remap[c] for c in lab])
    cents = np.stack([z[lab == c].mean(0) for c in range(len(remap))])
    return SpeakerModel(mean, std, cents, [int((lab == c).sum()) for c in range(len(remap))], sil)


def load_or_fit(path: Path | None, feats: np.ndarray) -> tuple[SpeakerModel, bool]:
    """Reuse the stored model (so voices stay consistent across zips) or fit and store a new one."""
    if path and path.exists():
        return SpeakerModel.from_json(json.loads(path.read_text(encoding="utf-8"))), False
    m = fit(feats)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(m.to_json()), encoding="utf-8")
    return m, True
