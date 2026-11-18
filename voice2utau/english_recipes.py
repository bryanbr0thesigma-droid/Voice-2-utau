"""The five characters as English recipes: the same voice sources, noise, clicks and finishing touches as their
Japanese CV banks (see voice2utau.characters, voice2utau.andro), plugged into voice2utau.arpasing.
Console-sound voices get a real glottal voice layered under them (Recipe.vowel_mix, 0.5 by default; nin-Chan 0.3 so the
chip still leads), so their vowels read as vowels while the button / chime / chip sound stays."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import characters, ninchan, synthbank
from .arpasing import Recipe

VB = Path(__file__).resolve().parent.parent / "voicebanks"


def mado() -> Recipe:
    """A woman in her early fifties: lower speaking pitch, slightly longer tract, warmer and a little breathy."""
    return Recipe("MADO_EN", f0=characters.MADO_F0, scale=characters.MADO_SCALE, source=characters.mature_source(),
                  vowel_mix=0.0,
                  readme="MADO in English: the synthesiser voice built like Adachi Rei, as a woman in her early "
                         "fifties - lower speaking pitch, a slightly longer vocal tract, warmer, a little breathy.")


def ninchan_recipe() -> Recipe:
    """A chip voice that still sings words: the formant-filtered pulse (with a little real voice under it) carries the
    vowels and consonants, a raw 12.5% pulse tone sits on the vowels only (no vocal filter, ~6 dB under the voice) for
    the chiptune, and a light 7-bit crunch. The full-chip version (raw tone on everything, sample-and-hold) scored 37%
    of word sounds recognised against MADO's 63%; this one keeps the character and most of the words."""
    N = VB / "ninchan" / "stage"
    clicks = [N / "wii" / "Click.wav", N / "wii" / "Button press.wav", N / "switch" / "Klick.wav", N / "switch" / "Tick.wav"]
    post = characters.ninchan_post()
    q = 2 ** 6                                                # 7-bit

    def crunch(y: np.ndarray) -> np.ndarray:
        return np.round(np.clip(y, -1, 1) * q) / q

    return Recipe("nin-Chan_EN", source=characters.chip_source(), noise=characters.lfsr_noise(), vowel_mix=0.4,
                  direct=characters.chip_source(0.125), direct_db=-6.0,
                  burst=ninchan.make_burst([p for p in clicks if p.exists()]),
                  onset_post=lambda y, c, rng: post(y, c, None, rng), all_post=crunch,
                  readme="nin-Chan in English: a chip voice that still sings words. Pulse wave through the vocal filter "
                         "(with some real voice under it) for the words, a raw 12.5% pulse tone on the vowels for "
                         "the chiptune, NES noise, Nintendo clicks, a menu blip on each syllable, 7-bit crunch.")


def bako() -> Recipe:
    F, T = VB / "bako" / "sfxstage", VB / "bako" / "stage"
    c = ninchan.pick_carriers([], top=1, keep=0.5, files=[F / "xboxstartup" / "6_series_xs_2020.wav"])
    noisy = ninchan.noisy_files([T / "og", F / "xboxdash"])
    noise = ninchan.make_noise(noisy, mix=0.7) if noisy else None
    post = characters.bako_post(None, swell_floor=0.8)        # no whoosh in English: it buried his consonants
    return Recipe("BAKO_EN", source=characters.pad_source(c[0][3]), noise=noise,
                  onset_post=lambda y, c0, rng: post(y, c0, None, rng),
                  readme="BAKO in English: pad voice from the Xbox Series startup with a real voice layered under it, "
                         "a gentle swell into each vowel and a low hum under the note (no air whoosh: in English it "
                         "buried his consonants).")


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
    """ANDRO's blended core: every brand's tone, hiss and clicks pooled (the same choices as his Japanese bank). The
    choices take ~10 minutes of analysis, so they are cached next to his sources."""
    import json
    from . import andro as A
    stage = VB / "andro" / "clean"
    cache = VB / "andro" / "english_recipe_cache.json"
    if cache.exists():
        c = json.loads(cache.read_text())
    else:
        brands = [b for b in A.BRANDS if (stage / b).is_dir()]
        files = {b: sorted((stage / b).glob("*.wav")) for b in brands}
        stock = A.stock_names(stage / "and")
        own = {b: (files[b] if b == "and" else [p for p in files[b] if not A.is_stock(p, stock)]) or files[b] for b in brands}
        c = {"tones": [], "noisy": [], "clicks": []}
        with A.male():
            for b in brands:
                r = A.roles(own[b])
                c["noisy"] += [str(p) for p in A.spread(r["noisy"], 12)]
                c["clicks"] += [str(p) for p in A.spread(r["clicks"], 12)]
                best = ninchan.pick_carriers([], top=1, keep=0.4, files=A.spread(r["tonal"], 300))
                if best:
                    c["tones"].append(best[0][0])
        cache.write_text(json.dumps(c, indent=1))
    with A.male():
        tables = [ninchan.pick_carriers([], top=1, keep=0.4, files=[Path(f)])[0][3] for f in c["tones"]]
        src = ninchan.make_source(tables)
    noisy, clicks = [Path(p) for p in c["noisy"]], [Path(p) for p in c["clicks"]]
    return Recipe("ANDRO_EN", f0=A.MALE_F0, scale=A.MALE_SCALE, source=src,
                  noise=ninchan.make_noise(noisy, mix=0.6) if noisy else None,
                  burst=ninchan.make_burst(clicks) if clicks else None,
                  readme="ANDRO in English: his blended main voice (every Android brand pooled), male.")


def rigo() -> Recipe:
    from . import rigo as R
    return R.recipe()


def puresu() -> Recipe:
    from . import puresu as P
    return P.recipe()


def unix() -> Recipe:
    from . import unix as X
    return X.recipe()


def linx() -> Recipe:
    from . import linx as L
    return L.recipe()


def joki() -> Recipe:
    from . import joki as J
    return J.recipe()


RECIPES = {"joki": joki, "linx": linx, "unix": unix, "puresu": puresu, "rigo": rigo, "mado": mado, "ninchan": ninchan_recipe, "bako": bako, "kaso": kaso, "andro": andro}
