"""Cross-validate phoneme-recogniser labels using the speaker's own recordings.

The CTC recogniser is noisy (roughly 70-75% of mora labels are right on real speech). All
candidates come from one speaker, so they can vouch for each other:

  * consonant / vowel centroids: a candidate whose consonant (or vowel) sounds closer to a
    different label's robust centroid than to its own is rejected;
  * consensus: among the several recordings of one mora, the one most similar to the others
    (the medoid) is preferred.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from . import audio
from .extract import Candidate

HOP = 0.01
SEED_CONF = 0.7     # candidates at least this confident define the centroids
MIN_SEED = 4        # a centroid needs at least this many members


def _cons_key(c: Candidate):
    """Onset and coda consonants sound different, so they get separate centroids."""
    return (c.mora.kind == "VC", c.mora.cls)


class Validator:
    def __init__(self, wav16: dict[str, Path]):
        import torch
        import torchaudio
        self._torch = torch
        self._mfcc = torchaudio.transforms.MFCC(
            sample_rate=audio.SR_REC, n_mfcc=13, melkwargs=dict(n_fft=400, hop_length=160, n_mels=40))
        self.wav16 = wav16
        self._frames: dict[str, np.ndarray] = {}

    def _compute(self) -> None:
        raw = {}
        for sid, p in self.wav16.items():
            x, _ = audio.read_wav(p)
            raw[sid] = self._mfcc(self._torch.from_numpy(x)).T.numpy()
        allf = np.concatenate(list(raw.values()))
        mu, sd = allf.mean(0), allf.std(0) + 1e-6          # speaker/channel normalisation
        self._frames = {sid: (m - mu) / sd for sid, m in raw.items()}

    def _features(self, c: Candidate):
        m = self._frames[c.src]
        a, s, b = int(c.start / HOP), int(c.split / HOP), int(c.end / HOP)
        b = min(b, len(m))
        if b - a < 6:
            return None
        if c.mora.kind == "VC":          # vowel first, consonant after the split
            v0 = a + int(0.3 * (s - a))
            v1 = max(v0 + 1, s - int(0.1 * (s - a)))
            cons = m[s:b].mean(0) if b - s >= 2 else None
        else:
            v0 = s + int(0.3 * (b - s))
            v1 = max(v0 + 1, b - int(0.15 * (b - s)))
            cons = m[a:max(a + 1, s)].mean(0) if s - a >= 2 else None
        vowel = m[v0:v1].mean(0) if v1 <= len(m) and v0 < v1 else None
        shape = m[np.linspace(a, b - 1, 8).astype(int)].ravel()
        return vowel, cons, shape

    @staticmethod
    def _centroids(cands, which: int, keyf):
        d = defaultdict(list)
        for c in cands:
            v = c.feat[which]
            if v is not None:
                d[keyf(c)].append(v)
        return {k: np.median(np.stack(v), 0) for k, v in d.items() if len(v) >= MIN_SEED}

    @staticmethod
    def _margin(v, cen, k):
        """>0 when v is closer to centroid k than to any other centroid."""
        if v is None or k not in cen or len(cen) < 2:
            return None
        d = {kk: float(np.linalg.norm(v - cc)) for kk, cc in cen.items()}
        return min(x for kk, x in d.items() if kk != k) - d[k]

    def annotate(self, cands: list[Candidate]) -> None:
        """Attach `vmargin`, `cmargin`, `consensus` and `feat` to every candidate (in place)."""
        self._compute()
        for c in cands:
            c.feat = self._features(c)
            c.vmargin = c.cmargin = c.consensus = None
        usable = [c for c in cands if c.feat is not None]
        seed = [c for c in usable if c.conf >= SEED_CONF]
        vcen = self._centroids(seed, 0, lambda c: c.mora.vowel or "N")
        ccen = self._centroids(seed, 1, _cons_key)
        for c in usable:
            c.vmargin = self._margin(c.feat[0], vcen, c.mora.vowel or "N")
            c.cmargin = self._margin(c.feat[1], ccen, _cons_key(c))
        by = defaultdict(list)
        for c in usable:
            if c.vmargin is None or c.vmargin > 0:
                by[c.mora.key].append(c)
        for k, v in by.items():
            if len(v) < 3:
                continue
            f = np.stack([c.feat[2] for c in v])
            dist = np.linalg.norm(f[:, None] - f[None], axis=-1)
            for i, c in enumerate(v):
                c.consensus = -float(np.median(np.delete(dist[i], i)))
        vals = [c.consensus for c in usable if c.consensus is not None]
        if vals:
            mu, sd = float(np.mean(vals)), float(np.std(vals)) + 1e-9
            for c in usable:
                if c.consensus is not None:
                    c.consensus = (c.consensus - mu) / sd

    @staticmethod
    def accept(c: Candidate) -> bool:
        """Reject candidates that sound like a different consonant/vowel than their label."""
        if getattr(c, "feat", None) is None:
            return False
        return (c.vmargin is None or c.vmargin > 0) and (c.cmargin is None or c.cmargin > 0)

    @staticmethod
    def rank_score(c: Candidate, weight: float = 0.4) -> float:
        return c.score + (weight * c.consensus if c.consensus is not None else 0.0)
