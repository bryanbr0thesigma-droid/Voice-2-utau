import numpy as np

from voice2utau.spatial import binaural, drift, ear_response

SR = 44100


def _tone(n=SR):
    rng = np.random.default_rng(0)
    return rng.standard_normal(n).astype("float32") * 0.1


def test_front_is_centred():
    y = binaural(_tone(), SR, 0.0, reflections=())
    assert y.shape == (SR, 2)
    assert np.allclose(y[:, 0], y[:, 1], atol=1e-5)
    y = binaural(_tone(), SR, 0.0)                            # with the room: still balanced
    db = 20 * np.log10(np.sqrt((y[:, 0] ** 2).mean()) / np.sqrt((y[:, 1] ** 2).mean()))
    assert abs(db) < 0.5


def test_right_source_is_louder_and_earlier_on_the_right():
    x = _tone()
    y = binaural(x, SR, np.radians(60), reflections=())
    l, r = y[:, 0], y[:, 1]
    assert np.sqrt((r ** 2).mean()) > np.sqrt((l ** 2).mean()) * 1.2
    c = np.correlate(l[2000:6000], r[2000:6000], "full")
    lag = np.argmax(c) - (4000 - 1)                           # > 0: left lags right
    assert 10 <= lag <= 40                                    # ~0.5 ms interaural delay at 60 degrees


def test_head_shadow_cuts_highs_on_far_ear():
    f = np.array([200.0, 8000.0])
    hl, hr = ear_response(f, np.radians(90))
    assert abs(hl[1]) < abs(hl[0]) * 0.5                      # far ear loses highs
    assert abs(hr[1]) > abs(hr[0])                            # near ear gains a little


def test_drift_stays_in_front_and_keeps_level():
    az = drift(SR * 30, SR, 35, 24)
    assert np.degrees(np.abs(az)).max() <= 35.001 and abs(az[0]) < 1e-9
    x = _tone(SR * 3)
    y = binaural(x, SR, drift(len(x), SR, 35, 3))
    assert abs(np.sqrt((y ** 2).mean()) - np.sqrt((x ** 2).mean())) < 1e-3
