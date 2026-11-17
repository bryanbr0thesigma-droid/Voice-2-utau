"""The five characters as English recipes: the same voice sources, noise, clicks and finishing touches as their
Japanese CV banks (see voice2utau.characters, voice2utau.andro), plugged into voice2utau.english."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import characters, ninchan, synthbank
from .english import Recipe

VB = Path(__file__).resolve().parent.parent / "voicebanks"


def mado() -> Recipe:
    return Recipe("MADO_EN", readme="MADO in English: the plain synthesiser voice, as built like Adachi Rei.")


def ninchan_recipe() -> Recipe:
    N = VB / "ninchan" / "stage"
    clicks = [N / "wii" / "Click.wav", N / "wii" / "Button press.wav", N / "switch" / "Klick.wav", N / "switch" / "Tick.wav"]
    post = characters.ninchan_post()
    q = 2 ** 7

    def crunch(y: np.ndarray) -> np.ndarray:                 # 8-bit, at the bank's own level
        return np.round(np.clip(y, -1, 1) * q) / q

    return Recipe("nin-Chan_EN", source=characters.chip_source(), noise=characters.lfsr_noise(),
                  burst=ninchan.make_burst([p for p in clicks if p.exists()]),
                  onset_post=lambda y, c, rng: post(y, c, None, rng), all_post=crunch,
                  readme="nin-Chan in English: pulse-wave voice, NES noise, Nintendo clicks, a menu blip on each "
                         "syllable, 8-bit crunch.")


def bako() -> Recipe:
    F, T = VB / "bako" / "sfxstage", VB / "bako" / "stage"
    c = ninchan.pick_carriers([], top=1, keep=0.5, files=[F / "xboxstartup" / "6_series_xs_2020.wav"])
    noisy = ninchan.noisy_files([T / "og", F / "xboxdash"])
    whoosh = ninchan.make_noise(noisy, mix=0.9) if noisy else (lambda n, r: r.standard_normal(n))
    noise = ninchan.make_noise(noisy, mix=0.7) if noisy else None
    post = characters.bako_post(whoosh)
    return Recipe("BAKO_EN", source=characters.pad_source(c[0][3]), noise=noise,
                  onset_post=lambda y, c0, rng: post(y, c0, None, rng),
                  readme="BAKO in English: pad voice from the Xbox Series startup, a swell and an air whoosh into "
                         "each syllable, a low hum under the note.")


def kaso() -> Recipe:
    N = VB / "quest" / "stage" / "named"
    c = ninchan.pick_carriers([], top=1, keep=0.6, files=[N / "success.wav"])
    noise_files = [N / f"{n}.wav" for n in "swipe swipe_backward scroll surface_draw system_gesture paging".split()]
    clicks = [N / f"{n}.wav" for n in "general_click general_keypress press select backspace".split()]
    return Recipe("Kaso_EN", source=characters.airy_source(c[0][3], 0.35), noise=ninchan.make_noise(noise_files, mix=0.6),
                  burst=ninchan.make_burst(clicks),
                  readme="Kaso in English: a soft Quest menu-chime tone carried on breath, Quest clicks as releases. "
                         "Best kept dry (no reverb).")


def andro() -> Recipe:
    """ANDRO's blended core: every brand's tone, hiss and clicks pooled (the same choices as his Japanese bank)."""
    from . import andro as A
    stage = VB / "andro" / "clean"
    brands = [b for b in A.BRANDS if (stage / b).is_dir()]
    files = {b: sorted((stage / b).glob("*.wav")) for b in brands}
    stock = A.stock_names(stage / "and")
    own = {b: (files[b] if b == "and" else [p for p in files[b] if not A.is_stock(p, stock)]) or files[b] for b in brands}
    pooled = {"clicks": [], "noisy": [], "tonal": []}
    tones = []
    with A.male():
        for b in brands:
            r = A.roles(own[b])
            for k in pooled:
                pooled[k] += A.spread(r[k], 12)
            best = ninchan.pick_carriers([], top=1, keep=0.4, files=A.spread(r["tonal"], 300))
            if best:
                tones.append(best[0][3])
        src = ninchan.make_source(tones)
    return Recipe("ANDRO_EN", f0=A.MALE_F0, scale=A.MALE_SCALE, source=src,
                  noise=ninchan.make_noise(pooled["noisy"], mix=0.6) if pooled["noisy"] else None,
                  burst=ninchan.make_burst(pooled["clicks"]) if pooled["clicks"] else None,
                  readme="ANDRO in English: his blended main voice (every Android brand pooled), male.")


RECIPES = {"mado": mado, "ninchan": ninchan_recipe, "bako": bako, "kaso": kaso, "andro": andro}
