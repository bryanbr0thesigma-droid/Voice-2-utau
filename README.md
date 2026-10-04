# Voice → UTAU

Upload a **zip of voice lines** or **one long recording (mp3 etc.)** and get a **UTAU voicebank**
(Japanese hiragana CV set, 101 aliases, `oto.ini` included).

* Sounds that occur in your recording are cut out and used as they are.
* Sounds the voice doesn't cover are taken from a **template voicebank** and converted to the
  character's voice with **your RVC model**.

Two bank types (`--language ja|en`, or the "Voicebank type" menu in the web app):

* **Japanese** – 101 hiragana CV units (か, きゃ, ん …). Works with any spoken language by mapping sounds to the
  nearest mora (English "see" → し).
* **English** – 675-unit ARPAbet CVVC bank: CV `k ae`, VC `ae t`, initial vowel `- ae` (files `cv_k_ae.wav`, …).
  Aliases are plain ARPAbet pairs; use them as lyrics in UTAU, or map them with a phonemizer/`prefix.map` in
  your editor. Impossible English combinations (e.g. an `ng` onset) are left out.

### Mixing English and German lines (English bank)

German can top up an English bank: put the files in `en/` and `de/` folders inside the zip (or add `_en` / `_de`
to the file name; a single file uses the "Language of the recording" setting). German is a **fallback only** — it
is used for an English unit only when no English line produced a usable one, and it is shown in the result grid
with a dotted underline and in the report (`lang`). Only German sounds with a close English match are used
(ɪ iː ʊ uː ɛ ɔ a ə, ay/aw/oy, p b t d k g f v s z sh zh h m n ng l y ch jh). Dropped on purpose: ʁ, x, ç, ts, pf, y, ʏ, ø, œ and the pure
vowels eː/oː, which have no honest English equivalent. If the German lines are voiced by a different actor than the
English ones, the recordings will not match in timbre — check the underlined cells by ear.

## How it works

```
zip / mp3 ──ffmpeg──▶ 44.1k + 16k wav
                        │
        wav2vec2 phoneme recogniser (IPA + timings, any language)
                        │
   consonant+vowel pairs → mora candidates (か, きゃ, ん, …)
                        │
   cross-validation against the speaker's own recordings   ← rejects mislabelled candidates
                        │
   best clip per mora ─────────────┐
                                   ├─▶ pitch-flatten, level, oto.ini ─▶ voicebank .zip
   missing morae → template clip → RVC (your model) ┘
```

Template: by default generated locally with **espeak-ng** (no licence concerns; robotic timbre, which RVC
replaces). You can instead upload any UTAU voicebank `.zip` as the template.

## Install

```bash
sudo apt install ffmpeg espeak-ng          # or brew install ffmpeg espeak-ng
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # torch is large; use the CPU wheel if you have no GPU
```

The phoneme model (`facebook/wav2vec2-xlsr-53-espeak-cv-ft`, ~1.2 GB) downloads on first use.

### RVC engine (needed to fill gaps with the character's voice)

RVC is not bundled. Pick one:

0. **Bundled runner `tools/rvc_infer.py`** (recommended; works without fairseq, tested here on CPU with a real RVC v2 model):
   ```bash
   python -m venv .venv-rvc && . .venv-rvc/bin/activate
   pip install -r requirements-rvc.txt && pip install --no-deps rvc-python
   export V2U_RVC_COMMAND="$PWD/.venv-rvc/bin/python $PWD/tools/rvc_infer.py --input {input} --output {output} --model {model} --index {index} --transpose {transpose}"
   export V2U_RVC_BATCH=40      # clips per RVC call: each call spends ~25 s loading models on CPU
   ```
   Base models (ContentVec + RMVPE, ~560 MB) download on first use. All checkpoints, including your `.pth`, are
   loaded with torch's restricted `weights_only` unpickler, so a booby-trapped model file cannot run code.
   On CPU, expect roughly 20–30 minutes for a full 675-unit English bank; use a GPU (`V2U_DEVICE=cuda:0`) if you have one.
1. `pip install rvc-python` – used automatically when installed. (`V2U_DEVICE=cuda:0` to use a GPU.)
2. Any RVC command-line tool (RVC WebUI, Applio, …) via a command template. Placeholders:
   `{input} {output} {model} {index} {transpose}`:
   ```bash
   export V2U_RVC_COMMAND='python /path/to/infer_cli.py --input {input} --output {output} --model {model} --index {index} --transpose {transpose}'
   ```
   `{index}` may be empty; empty arguments are dropped.

You also need the character's **RVC model (`.pth`, optional `.index`)**. No model? Either train one on the
dataset the app exports, or set `V2U_RVC_TRAIN_COMMAND` (`{dataset} {name} {out}` placeholders) and the
app will train one for you.

## Use

Web app (local only, no login — don't expose it to the internet):

```bash
python -m voice2utau.server            # http://127.0.0.1:8000
```

CLI:

```bash
python -m voice2utau voice.zip --name MyChar --rvc-model char.pth --rvc-index char.index
python -m voice2utau long.mp3  --name MyChar --gap-fill none        # recorded morae only
```

Output: `<name>_utau.zip` containing the wavs, `oto.ini` (Shift-JIS), `character.txt` and
`voice2utau_report.json` (which sample is a recording / RVC / raw template, confidence, pitch).

### Big corpora: feed several zips, one at a time

```bash
python -m voice2utau zip1.zip --language en --name MyChar --gap-fill none --state mychar_state -o out
python -m voice2utau zip2.zip --language en --name MyChar --gap-fill none --state mychar_state -o out   # adds to zip1
```
Each run keeps the best 12 clips per unit in `--state` and rebuilds the bank from everything seen so far, so a better
take in zip 3 replaces a worse one from zip 1, and you can test the bank after every zip. Sending the same zip twice is
detected and ignored. Selection prefers calm, steady-pitch takes near the speaker's usual pitch. The report shows
`coverage.speech_covered_by_recordings` (share of everyday English covered by real recordings) and the most-needed missing units.
(Measured on LJSpeech: 50 lines → 210 units / 73 % of speech; 100 lines → 263 units / 79 %.)

### Corpora with several voices (e.g. a game's villager, mayor, trader)

`--speaker largest` (or a voice number) clusters the files by voice and uses only one. Short audio previews of every
voice are written to `<out>/speaker_previews/voice_N.wav` so you can pick the right number, and the report lists each
voice's file count and pitch. With `--state`, the fitted voices are stored, so later zips are assigned to the same ones.
(The Minecraft villager pack separated into ~3 voices at about 237 / 335 / 444 Hz; silhouette 0.30 vs 0.001 for random labels.)

### Extra languages on an English bank (no RVC)

```bash
python -m voice2utau.crossling output/Villager -o output --name VLGR   # + Japanese (hiragana & romaji), 101 sounds
python -m voice2utau.mandarin  output/VLGR                             # + Mandarin (plain pinyin), 413 syllables, in place
```
Both reuse the English recordings (so they carry the English accent): sounds point at the closest English unit, and
sounds with no English unit are spliced from existing clips (`ja_*.wav`, `zh_*.wav`). Tables to tune are at the top of
`crossling.py` / `mandarin.py`. Where a plain spelling is both a pinyin syllable and a Japanese romaji alias with a different
sound (shi, chi, ru, za, ju …), pinyin wins; the Japanese sound stays available by its hiragana name.

### Gap-fill modes

| mode | behaviour |
|---|---|
| `auto` (default) | RVC when a model is supplied (or trainable); otherwise **leave the gaps** and export a training dataset. Never silently mixes in another voice. |
| `rvc` | Same, but errors out immediately if no model/engine is available. |
| `template` | Fill with the raw template — **not** the target voice. Preview only. |
| `none` | Recorded morae only. |

### Settings (environment)

`V2U_DATA_DIR` (default `./data`), `V2U_MAX_UPLOAD_MB` (2048), `V2U_MAX_MODEL_MB` (1024),
`V2U_JOB_TTL_HOURS` (72), `V2U_RVC_COMMAND`, `V2U_RVC_TRAIN_COMMAND`, `V2U_DEVICE`.

## Tests

```bash
pytest                         # fast, no model downloads
V2U_E2E=1 pytest tests/test_e2e.py   # full pipeline incl. phoneme model (downloads ~1.2 GB)
```

## What was verified — and what wasn't

Verified in development: decoding of zip/mp3/flac, the full pipeline on a 3-minute synthetic recording and on
real human speech (JSUT, 9 min), the web UI in a browser, `oto.ini` Shift-JIS output, zip-slip/size-limit
handling, RVC batching/splitting through a stand-in RVC command, and the 32 unit tests.

**Not verified:** the German fallback on real German speech (it is unit-tested and the sound mapping is conservative, but I have not measured its accuracy); a real RVC model/engine (no GPU or model here — `rvc-python` and CLI wiring are written to
their documented interfaces but only the plumbing was exercised), and loading the result in UTAU/OpenUTAU.

### Accuracy — please read

The phoneme recogniser is imperfect. On 9 minutes of real read speech measured against the transcript:

| | morae found | top pick per mora correct |
|---|---|---|
| Japanese: recogniser alone | 81 / 101 | ~70 % |
| Japanese: + cross-validation (default) | 67 / 101 | ~82 % |
| English (12 min LJSpeech): recogniser alone | 366 / 675 | ~82 % |
| English: confidence ≥ 0.7 (default) | 284 / 675 | ~91 % |

For English the acoustic cross-check removed half the units for +2 points, so it is off by default there
(Advanced → Cross-check). The English template was checked against textbook vowel formants; filled units
are generated by espeak-ng and rely entirely on your RVC model for the character's timbre.

(The reference itself is noisy — kanji readings, devoiced vowels — so true figures are probably a bit higher.)
Rejected morae are filled from the template via RVC rather than kept as doubtful recordings. Always audition
the result (click the kana in the UI) — each sample is colour-coded by origin.
Other limitations: samples are cut from running speech (not sustained vowels), so long notes stretch less
cleanly than in a purpose-recorded bank; consonant/vowel timing in `oto.ini` is estimated.

### Licences

* Check the **rights to the source voice** and to any **RVC model/template** you use before sharing the bank.
* The built-in template is generated by espeak-ng (GPL tool, output audio is not encumbered).
* No third-party UTAU bank is bundled; none was found with a clearly redistributable licence. If you use one as a
  template, its terms apply.
