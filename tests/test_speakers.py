import json

import numpy as np

from voice2utau import speakers


def synth(n_per=(120, 60, 25), seed=0):
    """Fake fingerprints: 3 voices that differ in timbre and pitch (log2 f0 in the last column)."""
    rng = np.random.RandomState(seed)
    centres = [(0.0, 7.8), (1.5, 8.4), (-1.5, 8.8)]       # (timbre offset, log2 f0)
    feats, truth = [], []
    for v, ((t, f), n) in enumerate(zip(centres, n_per)):
        base = np.full(19, t) + rng.randn(19) * 0.3
        feats += [np.concatenate([base + rng.randn(19) * 0.35, [f + rng.randn() * 0.15]]) for _ in range(n)]
        truth += [v] * n
    return np.stack(feats), np.array(truth)


def test_fit_finds_voices_ordered_by_size():
    F, truth = synth()
    m = speakers.fit(F)
    assert len(m.sizes) == 3 and m.sizes == sorted(m.sizes, reverse=True) and m.silhouette > 0.25
    lab = m.assign(F)
    assert (lab == truth).mean() > 0.95            # clusters are numbered by size = same order as truth


def test_single_voice_is_not_split():
    rng = np.random.RandomState(1)
    F = np.hstack([rng.randn(150, 19) * 0.3, 8.0 + rng.randn(150, 1) * 0.1])
    assert speakers.fit(F).sizes == [150]


def test_stored_model_assigns_later_zips_to_the_same_voices(tmp_path):
    F, truth = synth()
    path = tmp_path / "speakers.json"
    m1, fresh1 = speakers.load_or_fit(path, F[::2])
    G, truth2 = synth(seed=5)                        # a second zip, different samples, shuffled
    m2, fresh2 = speakers.load_or_fit(path, G)
    assert fresh1 and not fresh2 and json.loads(path.read_text())["sizes"] == m1.sizes
    assert (m2.assign(G) == truth2).mean() > 0.95    # same numbering as in the first zip
