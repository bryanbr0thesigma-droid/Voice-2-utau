"""End-to-end pipeline: upload -> recorded morae -> template+RVC gap fill -> UTAU voicebank."""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio, bank, profiles, rvc
from .corpus import Corpus, CorpusError, file_sha1
from .extract import Clip, choose, collect_options
from .ingest import IngestError, Source, prepare_sources
from .validate import Validator
from .templates import EspeakTemplate, TemplateError, VoicebankTemplate

GAP_FILL_MODES = ("auto", "rvc", "template", "none")
MAX_DATASET_SECONDS = 45 * 60


class PipelineError(RuntimeError):
    pass


@dataclass
class Options:
    name: str = "MyVoice"
    source_lang: str = "en"           # language of the recording(s); zip folders/file names tagged de/en override it
    language: str = "ja"             # ja = hiragana CV bank, en = English ARPAbet CVVC bank
    gap_fill: str = "auto"            # auto | rvc | template | none
    rvc_model: Path | None = None
    rvc_index: Path | None = None
    rvc_command: str | None = None
    template: Path | None = None      # custom UTAU bank (.zip / folder); default = espeak-ng
    flatten_pitch: bool = True
    state_dir: Path | None = None     # accumulate the best clips across several uploads in this directory
    min_conf: float | None = None     # None = language default
    validate: bool | None = None      # cross-check labels against the speaker's other recordings; None = language default


@dataclass
class Result:
    bank_dir: Path
    bank_zip: Path
    dataset_zip: Path | None
    report: dict = field(default_factory=dict)


Progress = Callable[[str, float, str], None]

_recognizer = None
_recognizer_lock = threading.Lock()


def get_recognizer():
    global _recognizer
    with _recognizer_lock:
        if _recognizer is None:
            from .phonemes import PhonemeRecognizer
            _recognizer = PhonemeRecognizer(device=os.environ.get("V2U_DEVICE_TORCH") or None)
        return _recognizer


def _speech_f0(sources: list[Source], limit_s: float = 120.0) -> float | None:
    """Fallback voice pitch estimate from raw speech when no morae were recorded."""
    vals = []
    budget = limit_s
    for s in sources:
        if budget <= 0:
            break
        x, sr = audio.read_wav(s.wav44, 0, min(budget, 60.0))
        budget -= len(x) / sr
        m = audio.median_f0(x, sr)
        if m:
            vals.append(m)
    return float(np.median(vals)) if vals else None


def run(upload: Path, work: Path, out_dir: Path, opt: Options,
        progress: Progress | None = None) -> Result:
    t0 = time.time()
    P = progress or (lambda stage, frac, msg: None)
    warnings: list[str] = []
    if opt.gap_fill not in GAP_FILL_MODES:
        raise PipelineError(f"gap_fill must be one of {GAP_FILL_MODES}")
    try:
        prof = profiles.get(opt.language)
    except ValueError as e:
        raise PipelineError(str(e)) from e
    units = prof.units
    min_conf = prof.default_min_conf if opt.min_conf is None else opt.min_conf
    validate = prof.default_validate if opt.validate is None else opt.validate
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fail fast: validate the RVC set-up before spending minutes on recognition.
    backend = None
    want_rvc = opt.gap_fill == "rvc" or (opt.gap_fill == "auto" and opt.rvc_model is not None)
    train_cmd = os.environ.get("V2U_RVC_TRAIN_COMMAND")
    if want_rvc and opt.rvc_model is not None:
        try:
            backend = rvc.make_backend(opt.rvc_model, opt.rvc_index, opt.rvc_command)
        except rvc.RVCError as e:
            raise PipelineError(str(e)) from e
    elif opt.gap_fill == "rvc" and not train_cmd:
        raise PipelineError("gap fill 'rvc' needs an RVC model (.pth). Upload one, choose 'template' "
                            "or 'none', or configure V2U_RVC_TRAIN_COMMAND to train one automatically.")

    # 1. ingest ------------------------------------------------------------------
    P("ingest", 0.0, "Decoding audio…")
    try:
        sources = prepare_sources(upload, work / "audio", lambda f: P("ingest", f, "Decoding audio…"),
                                  opt.source_lang)
    except IngestError as e:
        raise PipelineError(str(e)) from e
    lang_of = {s.id: s.lang for s in sources}
    total_s = sum(s.seconds for s in sources)
    P("ingest", 1.0, f"{len(sources)} file(s), {total_s / 60:.1f} min of audio")

    # 2. recognise phonemes ------------------------------------------------------
    P("recognise", 0.0, "Loading phoneme model (first run downloads ~1.2 GB)…")
    rec = get_recognizer()
    cands = []
    n_phones = 0
    for si, s in enumerate(sources):
        phones = rec.recognise_file(
            s.wav16, lambda f, si=si: P("recognise", (si + f) / len(sources),
                                        f"Listening… file {si + 1}/{len(sources)}"))
        n_phones += len(phones)
        cands += prof.build_candidates(phones, s.id, s.lang)
    P("recognise", 1.0, f"{n_phones} phonemes, {len(cands)} mora candidates")

    # 3. choose the best recording of every mora ---------------------------------
    P("select", 0.0, "Picking the cleanest sample for each mora…")
    wavs = {s.id: s.wav44 for s in sources}
    validator = Validator({s.id: s.wav16 for s in sources}) if validate else None
    options = collect_options(cands, wavs, min_conf, top_k=12, validator=validator)
    name_of = {s.id: s.original for s in sources}
    upload_tag = file_sha1(upload)[:10]
    for opts in options.values():
        for o in opts:
            o.clip.meta["lang"] = lang_of.get(o.clip.origin)
            o.clip.meta["file"] = name_of.get(o.clip.origin)
            o.clip.origin = f"{upload_tag}:{o.clip.origin}"
    corpus_info = None
    if opt.state_dir:
        try:
            corpus = Corpus.load(opt.state_dir, prof.code, units)
        except CorpusError as e:
            raise PipelineError(str(e)) from e
        sha = file_sha1(upload)
        if corpus.has_upload(sha):
            warnings.append("this exact upload was already added to the state directory; it was not added twice")
        else:
            corpus.add(options, {"sha1": sha, "name": upload.name, "files": len(sources),
                                 "seconds": round(total_s, 1), "lang": opt.source_lang})
            corpus.save()
        options = corpus.options
        corpus_info = corpus.stats()
    clips: dict[str, Clip] = choose(options)
    recorded = len(clips)
    P("select", 1.0, f"{recorded}/{len(units)} units found in the recording")
    native = prof.native_lang

    def is_fallback(c: Clip) -> bool:
        return bool(native) and c.source == "recorded" and (c.meta.get("lang") or native) not in (native, "mixed")
    ref_f0 = [c.f0 for c in clips.values() if c.f0 and not is_fallback(c)] or [c.f0 for c in clips.values() if c.f0]
    target_f0 = float(np.median(ref_f0)) if ref_f0 else None
    if target_f0 is None:
        target_f0 = _speech_f0(sources)
    missing = [k for k in units if k not in clips]

    # 4. fill the gaps -----------------------------------------------------------
    template_info = None
    transpose = 0
    dataset_zip: Path | None = None
    mode = opt.gap_fill
    if missing and mode != "none":
        if mode == "auto" and backend is None and not train_cmd:
            warnings.append(
                f"{len(missing)} morae are missing and no RVC model was supplied, so they were left out "
                "rather than filled with a different voice. Train an RVC model on the exported dataset and "
                "re-run, or pick gap fill = 'template' to fill them with the raw (unconverted) template.")
            mode = "none"
    if missing and mode != "none":
        P("fill", 0.0, f"Building template for {len(missing)} missing morae…")
        try:
            tpl = (VoicebankTemplate(opt.template, work, units=units) if opt.template
                   else EspeakTemplate(target_f0, lang=prof.code))
        except TemplateError as e:
            raise PipelineError(str(e)) from e
        template_info = {"name": tpl.name, "license": tpl.license}
        filled: list[Clip] = []
        for k in missing:
            c = tpl.get(units[k])
            if c is not None:
                filled.append(c)
        skipped = [k for k in missing if k not in {c.mora.key for c in filled}]
        if skipped:
            warnings.append("template has no usable sample for: " + ", ".join(units[k].kana for k in skipped[:40])
                            + (f" … (+{len(skipped) - 40} more)" if len(skipped) > 40 else ""))
        if mode in ("rvc", "auto"):
            if backend is None:        # auto-train path
                P("fill", 0.05, "Training an RVC model on your voice…")
                dataset = _export_dataset(sources, work / "rvc_dataset")
                try:
                    model, index = rvc.train_model(dataset, bank.safe_name(opt.name), work / "rvc_train")
                    backend = rvc.make_backend(model, index, opt.rvc_command)
                except rvc.RVCError as e:
                    raise PipelineError(str(e)) from e
            tpl_f0s = [c.f0 for c in filled if c.f0]
            tpl_f0 = float(np.median(tpl_f0s)) if tpl_f0s else None
            if target_f0 and tpl_f0:
                transpose = int(np.clip(round(12 * np.log2(target_f0 / tpl_f0)), -12, 12))
            P("fill", 0.15, f"Running {len(filled)} template samples through RVC (transpose {transpose:+d})…")
            try:
                filled, w = rvc.convert_clips(filled, backend, transpose,
                                              lambda f: P("fill", 0.15 + 0.85 * f, "Converting with RVC…"))
            except rvc.RVCError as e:
                raise PipelineError(str(e)) from e
            warnings += w
        else:
            warnings.append("Gap fill = 'template': the filled samples are NOT in the target voice.")
        for c in filled:
            clips[c.mora.key] = c
        P("fill", 1.0, f"Filled {len(filled)} morae")
    elif missing:
        P("fill", 1.0, "Gap fill skipped")

    # 4b. Clips from a fallback language (e.g. German) come from a different performer. Run them
    #     through the same RVC model so the whole bank shares one timbre.
    fallback = [c for c in clips.values() if is_fallback(c)]
    converted_fallback = 0
    if fallback and backend is not None:
        f0s = [c.f0 for c in fallback if c.f0]
        tr = int(np.clip(round(12 * np.log2(target_f0 / np.median(f0s))), -12, 12)) if target_f0 and f0s else 0
        P("fill", 1.0, f"Converting {len(fallback)} fallback-language clips with RVC…")
        try:
            conv, w = rvc.convert_clips(fallback, backend, tr)
        except rvc.RVCError as e:
            raise PipelineError(str(e)) from e
        for c in conv:
            clips[c.mora.key] = c
        converted_fallback = sum(c.source == "rvc" for c in conv)
        warnings += w
        warnings.append(f"{converted_fallback} clip(s) from {', '.join(sorted({str(c.meta.get('lang')) for c in fallback}))} "
                        "lines were converted with your RVC model so they match the main voice.")
    elif fallback:
        warnings.append(f"{len(fallback)} clip(s) come from fallback-language lines (a different voice actor) and were NOT "
                        "converted because no RVC model was used. Expect a timbre mismatch on those (dotted cells).")

    if not clips:
        raise PipelineError("No usable samples were produced. The recording may be too short, too noisy, "
                            "or silent; try a longer/cleaner recording or enable template gap fill.")

    # 5. unify pitch & level -----------------------------------------------------
    P("finish", 0.0, "Normalising pitch and level…")
    ref = [c.f0 for c in clips.values() if c.f0 and c.source == "recorded"] or \
          [c.f0 for c in clips.values() if c.f0]
    bank_f0 = float(np.median(ref)) if ref else None
    flattened = 0
    for i, (k, c) in enumerate(clips.items()):
        y = c.audio
        if opt.flatten_pitch and bank_f0:
            y, changed = audio.flatten_pitch(y, audio.SR_BANK, bank_f0)
            flattened += changed
        c.audio = audio.normalise_level(y)
        P("finish", (i + 1) / len(clips), "Normalising pitch and level…")
    if opt.flatten_pitch and flattened < len(clips):
        warnings.append(f"{len(clips) - flattened} sample(s) kept their original pitch "
                        "(too far from the bank pitch to flatten cleanly).")

    # 6. write -------------------------------------------------------------------
    P("write", 0.0, "Writing voicebank…")
    report = {
        "name": opt.name,
        "language": prof.code,
        "bank_f0_hz": round(bank_f0, 1) if bank_f0 else None,
        "counts": {"total_target": len(units),
                   "recorded": sum(c.source == "recorded" for c in clips.values()),
                   "rvc": sum(c.source == "rvc" for c in clips.values()),
                   "template_raw": sum(c.source == "template" for c in clips.values()),
                   "fallback_converted": converted_fallback,
                   "missing": len(units) - len(clips)},
        "gap_fill_mode": mode,
        "settings": {"min_confidence": min_conf, "cross_check": validate},
        "corpus": corpus_info,
        "rvc_transpose": transpose if mode in ("rvc", "auto") and any(c.source == "rvc" for c in clips.values()) else None,
        "template": template_info,
        "sources": [{"file": s.original, "seconds": round(s.seconds, 1), "lang": s.lang} for s in sources],
        "samples": [{"key": k, "alias": c.mora.kana, "source": c.source, "confidence": round(c.conf, 2),
                     "f0_hz": round(c.f0, 1) if c.f0 else None,
                     "duration_ms": round(len(c.audio) / audio.SR_BANK * 1000),
                     "lang": c.meta.get("lang"), "file": c.meta.get("file")}
                    for k, c in clips.items()],
        "missing": [{"key": k, "alias": units[k].kana} for k in units if k not in clips],
        "warnings": warnings,
        "elapsed_s": round(time.time() - t0, 1),
    }
    report["coverage"] = _coverage(clips, units, prof)
    root = bank.write_bank(clips, out_dir, opt.name, report, units)
    zpath = bank.zip_bank(root, out_dir / f"{root.name}_utau.zip")

    if report["counts"]["missing"] and backend is None and mode != "template":
        dataset_zip = _zip_dataset(_export_dataset(sources, work / "rvc_dataset"),
                                   out_dir / f"{root.name}_rvc_dataset.zip")
    P("write", 1.0, "Done")
    return Result(root, zpath, dataset_zip, report)


def _export_dataset(sources: list[Source], dest: Path) -> Path:
    """Write clean utterance-length clips of the source voice, for training an RVC model."""
    if dest.exists() and any(dest.iterdir()):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    total, n = 0.0, 0
    for s in sources:
        x, sr = audio.read_wav(s.wav44)
        for a, b in audio.split_utterances(x, sr):
            if total >= MAX_DATASET_SECONDS:
                return dest
            seg = x[int(a * sr): int(b * sr)]
            audio.write_wav(dest / f"{n:05d}.wav", seg, sr)
            total += len(seg) / sr
            n += 1
    return dest


def _zip_dataset(folder: Path, zpath: Path) -> Path | None:
    import zipfile
    files = sorted(folder.glob("*.wav"))
    if not files:
        return None
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
        for f in files:
            zf.write(f, f"dataset/{f.name}")
    return zpath


def _coverage(clips: dict[str, Clip], units: dict, prof) -> dict:
    """How much of everyday speech the bank covers, from real recordings and in total."""
    out: dict = {"recorded_units": sum(c.source == "recorded" for c in clips.values()),
                 "total_units": len(clips), "target_units": len(units)}
    w = prof.weights() if prof.weights else {}
    if not w:
        return out
    tot = sum(w.values()) or 1.0
    rec = sum(w.get(k, 0.0) for k, c in clips.items() if c.source == "recorded")
    allc = sum(w.get(k, 0.0) for k in clips)
    out["speech_covered_by_recordings"] = round(rec / tot, 4)
    out["speech_covered_total"] = round(allc / tot, 4)
    # units that are not real recordings (absent or filled by RVC/template), most common first
    needed = sorted((k for k in units if k in w and (k not in clips or clips[k].source != "recorded")),
                    key=lambda k: -w[k])
    out["most_needed_missing"] = [{"key": k, "alias": units[k].kana, "share": round(w[k], 5)} for k in needed[:25]]
    return out
