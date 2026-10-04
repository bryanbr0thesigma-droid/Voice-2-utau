"""Command line interface:  python -m voice2utau recording.zip --name MyVoice --rvc-model char.pth"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from . import pipeline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="voice2utau", description="Turn a voice recording into a UTAU voicebank.")
    ap.add_argument("input", type=Path, help="a .zip of voice lines, or one (long) audio file such as .mp3")
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
    ap.add_argument("--no-validate", action="store_true", help="skip acoustic cross-validation of phoneme labels")
    ap.add_argument("--min-conf", type=float, default=0.45, help="minimum phoneme confidence (0-1)")
    ap.add_argument("--work", type=Path, help="working directory (default: temporary)")
    a = ap.parse_args(argv)

    def progress(stage: str, frac: float, msg: str) -> None:
        print(f"\r[{stage:9s}] {frac * 100:5.1f}%  {msg:70s}", end="", file=sys.stderr, flush=True)
        if frac >= 1.0:
            print(file=sys.stderr)

    opt = pipeline.Options(name=a.name, gap_fill=a.gap_fill, rvc_model=a.rvc_model, rvc_index=a.rvc_index,
                           rvc_command=a.rvc_command, template=a.template,
                           flatten_pitch=not a.no_flatten, min_conf=a.min_conf,
                           validate=not a.no_validate)
    try:
        with tempfile.TemporaryDirectory() as td:
            res = pipeline.run(a.input, a.work or Path(td), a.out, opt, progress)
    except (pipeline.PipelineError, Exception) as e:   # noqa: BLE001 - show a clean message
        print(f"\nerror: {e}", file=sys.stderr)
        return 1
    c = res.report["counts"]
    print(f"\nVoicebank: {res.bank_zip}")
    print(f"  recorded: {c['recorded']}  rvc: {c['rvc']}  raw template: {c['template_raw']}  missing: {c['missing']}")
    if res.dataset_zip:
        print(f"  RVC training dataset: {res.dataset_zip}")
    for w in res.report["warnings"]:
        print(f"  ! {w}")
    return 0
