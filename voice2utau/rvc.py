"""RVC voice conversion backends and the batching logic that applies them to template clips.

RVC itself is not bundled (it needs a GPU-friendly stack and the user's own model).
Two backends are supported:

  * "rvc-python"  - the `rvc-python` package (pip install rvc-python), used when importable.
  * "command"     - any external RVC CLI (RVC WebUI, Applio, ...) described by a command template
                    with the placeholders {input} {output} {model} {index} {transpose}.
                    Set it with the V2U_RVC_COMMAND environment variable or the --rvc-command flag.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio
from .extract import Clip

BATCH_SIZE = 16
GAP_S = 0.5


class RVCError(RuntimeError):
    pass


class RVCBackend(ABC):
    name = "rvc"

    @abstractmethod
    def convert(self, src: Path, dst: Path, transpose: int) -> None: ...


class CommandBackend(RVCBackend):
    name = "rvc-command"

    def __init__(self, template: str, model: Path, index: Path | None, timeout: int = 1800):
        self.template, self.model, self.index, self.timeout = template, model, index, timeout
        if "{input}" not in template or "{output}" not in template:
            raise RVCError("RVC command must contain {input} and {output} placeholders")

    def convert(self, src: Path, dst: Path, transpose: int) -> None:
        values = {"input": str(src), "output": str(dst), "model": str(self.model),
                  "index": str(self.index) if self.index else "", "transpose": str(transpose)}
        argv = [tok.format_map(values) for tok in shlex.split(self.template)]
        argv = [a for a in argv if a != ""]   # drop empty optional {index}
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        except FileNotFoundError as e:
            raise RVCError(f"RVC command not found: {argv[0]}") from e
        except subprocess.TimeoutExpired as e:
            raise RVCError(f"RVC command timed out after {self.timeout}s") from e
        if r.returncode != 0:
            raise RVCError(f"RVC command failed ({r.returncode}): {(r.stderr or r.stdout)[-400:]}")
        if not dst.exists():
            raise RVCError("RVC command succeeded but produced no output file")


class RvcPythonBackend(RVCBackend):
    name = "rvc-python"

    def __init__(self, model: Path, index: Path | None, device: str | None = None):
        try:
            from rvc_python.infer import RVCInference
        except ImportError as e:
            raise RVCError("rvc-python is not installed (pip install rvc-python)") from e
        self.model_path, self.index = model, index
        dev = device or os.environ.get("V2U_DEVICE") or "cpu:0"
        self.rvc = RVCInference(device=dev)
        try:
            self.rvc.load_model(str(model), index_path=str(index) if index else "")
        except TypeError:
            self.rvc.load_model(str(model))

    def convert(self, src: Path, dst: Path, transpose: int) -> None:
        self.rvc.set_params(f0method="rmvpe", f0up_key=transpose, index_rate=0.6,
                            filter_radius=3, rms_mix_rate=0.25, protect=0.33)
        self.rvc.infer_file(str(src), str(dst))
        if not dst.exists():
            raise RVCError("rvc-python produced no output")


def make_backend(model: Path | None, index: Path | None, command: str | None = None) -> RVCBackend:
    command = command or os.environ.get("V2U_RVC_COMMAND")
    if model is None:
        raise RVCError("no RVC model (.pth) was provided")
    if not Path(model).exists():
        raise RVCError(f"RVC model not found: {model}")
    if command:
        return CommandBackend(command, Path(model), Path(index) if index else None)
    try:
        import rvc_python  # noqa: F401
    except ImportError:
        raise RVCError("No RVC engine available. Install `rvc-python` or set V2U_RVC_COMMAND "
                       "to your RVC CLI (see README).") from None
    return RvcPythonBackend(Path(model), Path(index) if index else None)


def convert_clips(clips: list[Clip], backend: RVCBackend, transpose: int,
                  progress: Callable[[float], None] | None = None) -> tuple[list[Clip], list[str]]:
    """Run clips through RVC in batches. Returns (converted clips, warnings)."""
    sr = audio.SR_BANK
    gap = np.zeros(int(GAP_S * sr), dtype=np.float32)
    out: list[Clip] = []
    warnings: list[str] = []
    batches = [clips[i:i + BATCH_SIZE] for i in range(0, len(clips), BATCH_SIZE)]
    for bi, batch in enumerate(batches):
        pieces, spans, pos = [gap], [], len(gap)
        for c in batch:
            spans.append((pos, pos + len(c.audio)))
            pieces += [c.audio, gap]
            pos += len(c.audio) + len(gap)
        joined = np.concatenate(pieces)
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / "in.wav", Path(td) / "out.wav"
            audio.write_wav(src, joined, sr)
            backend.convert(src, dst, transpose)
            y, ysr = audio.read_wav(dst)
        y = audio.resample(y, ysr, sr)
        ratio = len(y) / len(joined)
        for c, (a, b) in zip(batch, spans):
            lo = max(0, int((a - 0.2 * sr) * ratio))
            hi = min(len(y), int((b + 0.2 * sr) * ratio))
            win = y[lo:hi]
            if len(win) < sr * 0.1 or float(np.abs(win).max()) < 1e-3:
                warnings.append(f"RVC output for '{c.mora.kana}' was silent; using unconverted template")
                out.append(c)
                continue
            s, e = audio.trim_silence(win, sr, rel_db=-34.0)
            seg = audio.fade(win[s:e], sr)
            # keep the consonant/vowel proportion of the source clip
            split = c.split_s * (len(seg) / max(1, len(c.audio)))
            f0 = audio.median_f0(seg[len(seg) // 3:], sr)
            out.append(Clip(c.mora, seg, split, f0, "rvc", 1.0, c.origin))
        if progress:
            progress((bi + 1) / len(batches))
    return out, warnings


def train_model(dataset: Path, name: str, out_dir: Path, command: str | None = None) -> tuple[Path, Path | None]:
    """Optionally train an RVC model through an external command (V2U_RVC_TRAIN_COMMAND).

    The command template may use {dataset} {name} {out}; it must leave a .pth (and optionally
    a .index) somewhere under {out}.
    """
    command = command or os.environ.get("V2U_RVC_TRAIN_COMMAND")
    if not command:
        raise RVCError("no RVC model supplied and no training command configured "
                       "(set V2U_RVC_TRAIN_COMMAND, or train a model on the exported dataset)")
    out_dir.mkdir(parents=True, exist_ok=True)
    values = {"dataset": str(dataset), "name": name, "out": str(out_dir)}
    argv = [t.format_map(values) for t in shlex.split(command)]
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        raise RVCError(f"RVC training command failed ({r.returncode}): {(r.stderr or r.stdout)[-400:]}")
    pths = sorted((p for p in out_dir.rglob("*.pth") if not p.name.startswith(("G_", "D_"))),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not pths:
        raise RVCError("training finished but no .pth model was found in the output directory")
    idx = sorted(out_dir.rglob("*.index"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pths[0], (idx[0] if idx else None)
