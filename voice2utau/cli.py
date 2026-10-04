"""Command line interface:  python -m voice2utau recording.zip --name MyVoice --rvc-model char.pth"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from . import pipeline, profiles


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="voice2utau", description="Turn a voice recording into a UTAU voicebank.")
    ap.add_argument("input", type=Path, nargs="?",
                    help="a .zip of voice lines, or one (long) audio file such as .mp3. Omit it together with --state "
                         "to rebuild the bank from the saved state (e.g. to add gap filling) without re-listening")
    ap.add_argument("--language", choices=sorted(profiles.PROFILES), default="ja",
                    help="voicebank type: ja = hiragana CV, en = English ARPAbet CVVC (default: ja)")
    ap.add_argument("--source-lang", choices=["en", "de", "mixed"], default="en",
                    help="language spoken in the recording; in a zip, folders/file names tagged en/de override it. "
                         "German sounds are only used as a fallback for units the English lines lack. "
                         "mixed = English and German interleaved: only sounds that mean the same in both are used.")
    ap.add_argument("--name", default="MyVoice", help="voicebank name")
    ap.add_argument("-o", "--out", type=Path, default=Path("output"), help="output directory")
    ap.add_argument("--gap-fill", choices=pipeline.GAP_FILL_MODES, default="auto",
                    help="how to fill morae missing from the recording (default: auto = RVC when a model is given)")
    ap.add_argument("--rvc-model", type=Path, help="RVC .pth model of the character")
    ap.add_argument("--rvc-index", type=Path, help="optional RVC .index file")
    ap.add_argument("--rvc-command", help="external RVC command template, e.g. "
                    "'python infer.py -i {input} -o {output} -m {model} -k {transpose}'")
    ap.add_argument("--template", type=Path, help="UTAU voicebank (.zip/folder) to use as template instead of espeak-ng")
    ap.add_argument("--no-flatten", action="store_true", help="do not monotonise sample pitch")
    ap.add_argument("--cross-check", choices=["auto", "on", "off"], default="auto",
                    help="acoustic cross-validation of phoneme labels (auto: on for ja, off for en)")
    ap.add_argument("--min-conf", type=float, default=None,
                    help="minimum phoneme confidence 0-1 (default: 0.45 for ja, 0.7 for en)")
    ap.add_argument("--state", type=Path,
                    help="directory that accumulates the best clips across several uploads: run once per zip "
                         "with the same --state and every run rebuilds the bank from everything seen so far")
    ap.add_argument("--speaker", default="all",
                    help="when the corpus holds several voices: 'all' (default), 'largest', or a voice number. "
                         "Previews of every voice are written to <out>/speaker_previews. With --state the "
                         "voices stay the same across zips.")
    ap.add_argument("--work", type=Path, help="working directory (default: temporary)")
    a = ap.parse_args(argv)
    if a.input is None and not a.state:
        ap.error("give an input file, or --state to rebuild from a saved state")

    def progress(stage: str, frac: float, msg: str) -> None:
        print(f"\r[{stage:9s}] {frac * 100:5.1f}%  {msg:70s}", end="", file=sys.stderr, flush=True)
        if frac >= 1.0:
            print(file=sys.stderr)

    opt = pipeline.Options(name=a.name, language=a.language, source_lang=a.source_lang, gap_fill=a.gap_fill, rvc_model=a.rvc_model, rvc_index=a.rvc_index,
                           rvc_command=a.rvc_command, template=a.template,
                           flatten_pitch=not a.no_flatten, state_dir=a.state, speaker=a.speaker, min_conf=a.min_conf,
                           validate={"auto": None, "on": True, "off": False}[a.cross_check])
    try:
        with tempfile.TemporaryDirectory() as td:
            res = pipeline.run(a.input, a.work or Path(td), a.out, opt, progress)
    except (pipeline.PipelineError, Exception) as e:   # noqa: BLE001 - show a clean message
        print(f"\nerror: {e}", file=sys.stderr)
        return 1
    c = res.report["counts"]
    print(f"\nVoicebank: {res.bank_zip}")
    print(f"  recorded: {c['recorded']}/{c['total_target']}  rvc: {c['rvc']}  raw template: {c['template_raw']}  missing: {c['missing']}")
    if res.dataset_zip:
        print(f"  RVC training dataset: {res.dataset_zip}")
    for w in res.report["warnings"]:
        print(f"  ! {w}")
    return 0
