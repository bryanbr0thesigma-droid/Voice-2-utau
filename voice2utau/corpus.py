"""A growing pool of the best clips per unit, so a big corpus can be fed in several zips.

After every upload the best `keep` clips of each unit are written to disk; the next upload competes
against them. The state is small (a few thousand short wavs) compared with the audio it came from.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from . import audio
from .extract import Clip, Option, scorer

FORMAT = 1
KEEP = 12            # options kept per unit
KEEP_FOREIGN = 3     # of which at most this many from a fallback language


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


class CorpusError(RuntimeError):
    pass


class Corpus:
    def __init__(self, root: Path, language: str, units: dict):
        self.root, self.language, self.units = Path(root), language, units
        self.uploads: list[dict] = []
        self.options: dict[str, list[Option]] = {}

    # ---------------------------------------------------------------- persistence
    @classmethod
    def load(cls, root: Path, language: str, units: dict) -> "Corpus":
        c = cls(root, language, units)
        meta = Path(root) / "state.json"
        if not meta.exists():
            return c
        d = json.loads(meta.read_text(encoding="utf-8"))
        if d.get("format") != FORMAT:
            raise CorpusError(f"{meta}: unsupported state format")
        if d["language"] != language:
            raise CorpusError(f"this state directory holds a '{d['language']}' bank, not '{language}'")
        c.uploads = d["uploads"]
        for e in d["options"]:
            x, sr = audio.read_wav(Path(root) / "clips" / e["file"])
            clip = Clip(units[e["key"]], x, e["split_s"], e["f0"], e["source"], e["conf"], e["origin"], e["meta"])
            c.options.setdefault(e["key"], []).append(Option(e["total"], clip, e["foreign"]))
        return c

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        clips = self.root / "clips"
        shutil.rmtree(clips, ignore_errors=True)
        clips.mkdir()
        entries = []
        for key, opts in self.options.items():
            for i, o in enumerate(opts):
                fn = f"{key}__{i}.wav"
                audio.write_wav(clips / fn, o.clip.audio, audio.SR_BANK)
                entries.append({"key": key, "file": fn, "total": o.total, "foreign": o.foreign,
                                "split_s": o.clip.split_s, "f0": o.clip.f0, "source": o.clip.source,
                                "conf": o.clip.conf, "origin": o.clip.origin, "meta": o.clip.meta})
        tmp = self.root / "state.json.tmp"
        tmp.write_text(json.dumps({"format": FORMAT, "language": self.language, "uploads": self.uploads,
                                   "options": entries}), encoding="utf-8")
        tmp.replace(self.root / "state.json")

    # ---------------------------------------------------------------- merging
    def has_upload(self, sha1: str) -> bool:
        return any(u["sha1"] == sha1 for u in self.uploads)

    def add(self, options: dict[str, list[Option]], info: dict) -> None:
        for key, opts in options.items():
            self.options.setdefault(key, []).extend(opts)
        self.uploads.append(info)
        self._trim()

    def _trim(self) -> None:
        score = scorer(self.options)
        for key, opts in self.options.items():
            native = sorted((o for o in opts if not o.foreign), key=score, reverse=True)
            foreign = sorted((o for o in opts if o.foreign), key=score, reverse=True)
            self.options[key] = native[:KEEP] + foreign[:KEEP_FOREIGN]

    def stats(self) -> dict:
        return {"uploads": len(self.uploads),
                "files": sum(u.get("files", 0) for u in self.uploads),
                "seconds": round(sum(u.get("seconds", 0) for u in self.uploads), 1),
                "units_with_clips": sum(1 for o in self.options.values() if o),
                "clips_kept": sum(len(o) for o in self.options.values())}
