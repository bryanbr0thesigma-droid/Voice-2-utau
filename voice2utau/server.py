"""Local web app: upload a zip / long audio file, watch progress, download the UTAU voicebank."""
from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import pipeline, profiles
from .ingest import AUDIO_EXT, SOURCE_LANGS, IngestError

log = logging.getLogger("voice2utau")

DATA_DIR = Path(os.environ.get("V2U_DATA_DIR", "data")).resolve()
MAX_UPLOAD = int(float(os.environ.get("V2U_MAX_UPLOAD_MB", "2048")) * 1024 * 1024)
MAX_MODEL = int(float(os.environ.get("V2U_MAX_MODEL_MB", "1024")) * 1024 * 1024)
JOB_TTL_S = float(os.environ.get("V2U_JOB_TTL_HOURS", "72")) * 3600
ID_RE = re.compile(r"^[0-9a-f]{32}$")

STAGES = [("ingest", 0.05), ("recognise", 0.50), ("select", 0.10), ("fill", 0.20), ("finish", 0.10), ("write", 0.05)]
_STAGE_START = {}
_acc = 0.0
for _n, _w in STAGES:
    _STAGE_START[_n] = (_acc, _w)
    _acc += _w


@dataclass
class Job:
    id: str
    dir: Path
    status: str = "queued"          # queued | running | done | error
    stage: str = ""
    progress: float = 0.0
    message: str = "Queued"
    error: str | None = None
    log: list[str] = field(default_factory=list)
    report: dict | None = None
    bank_zip: Path | None = None
    dataset_zip: Path | None = None
    bank_dir: Path | None = None
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {"id": self.id, "status": self.status, "stage": self.stage, "progress": round(self.progress, 4),
                "message": self.message, "error": self.error, "log": self.log[-60:], "report": self.report,
                "has_bank": self.bank_zip is not None, "has_dataset": self.dataset_zip is not None}


app = FastAPI(title="Voice-2-UTAU")
JOBS: dict[str, Job] = {}
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v2u")
STATIC = Path(__file__).parent / "static"


def _job(job_id: str) -> Job:
    if not ID_RE.match(job_id) or job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    return JOBS[job_id]


async def _save_upload(up: UploadFile, dest: Path, limit: int) -> None:
    size = 0
    with open(dest, "wb") as f:
        while chunk := await up.read(1 << 20):
            size += len(chunk)
            if size > limit:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"{up.filename or 'file'} is larger than the {limit // 2**20} MB limit")
            f.write(chunk)


def _cleanup_old() -> None:
    now = time.time()
    for jid in [j for j, job in JOBS.items() if job.status in ("done", "error") and now - job.created > JOB_TTL_S]:
        shutil.rmtree(JOBS[jid].dir, ignore_errors=True)
        del JOBS[jid]


def engine_info() -> dict:
    try:
        import rvc_python  # noqa: F401
        has_py = True
    except ImportError:
        has_py = False
    cmd = bool(os.environ.get("V2U_RVC_COMMAND"))
    return {"rvc_available": has_py or cmd,
            "rvc_engine": "command" if cmd else ("rvc-python" if has_py else None),
            "rvc_training": bool(os.environ.get("V2U_RVC_TRAIN_COMMAND")),
            "espeak": shutil.which("espeak-ng") is not None,
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "audio_ext": sorted(AUDIO_EXT), "max_upload_mb": MAX_UPLOAD // 2**20,
            "languages": [{"code": p.code, "name": p.name, "description": p.description}
                          for p in profiles.PROFILES.values()],
            "units": {p.code: [{"key": m.key, "alias": m.kana, "group": p.group_of(m)} for m in p.units.values()]
                      for p in profiles.PROFILES.values()}}


@app.get("/api/config")
def config() -> dict:
    return engine_info()


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    name: str = Form("MyVoice"),
    language: str = Form("ja"),
    source_lang: str = Form("en"),
    gap_fill: str = Form("auto"),
    flatten_pitch: bool = Form(True),
    cross_check: str = Form("auto"),
    rvc_model: UploadFile | None = File(None),
    rvc_index: UploadFile | None = File(None),
    template: UploadFile | None = File(None),
) -> JSONResponse:
    if gap_fill not in pipeline.GAP_FILL_MODES:
        raise HTTPException(400, f"gap_fill must be one of {pipeline.GAP_FILL_MODES}")
    if language not in profiles.PROFILES:
        raise HTTPException(400, f"language must be one of {sorted(profiles.PROFILES)}")
    if source_lang not in SOURCE_LANGS:
        raise HTTPException(400, f"source_lang must be one of {SOURCE_LANGS}")
    if cross_check not in ("auto", "on", "off"):
        raise HTTPException(400, "cross_check must be auto, on or off")
    ext = Path(file.filename or "").suffix.lower()
    if ext != ".zip" and ext not in AUDIO_EXT:
        raise HTTPException(400, "upload a .zip of voice lines or an audio file (mp3, wav, flac, ogg, m4a, …)")
    jid = uuid.uuid4().hex
    jdir = DATA_DIR / "jobs" / jid
    (jdir / "in").mkdir(parents=True)
    try:
        upload = jdir / "in" / f"upload{ext}"
        await _save_upload(file, upload, MAX_UPLOAD)
        opt = pipeline.Options(name=(name.strip() or "MyVoice")[:60], language=language, source_lang=source_lang, gap_fill=gap_fill,
                               flatten_pitch=flatten_pitch,
                               validate={"auto": None, "on": True, "off": False}[cross_check])
        if rvc_model is not None and rvc_model.filename:
            if Path(rvc_model.filename).suffix.lower() != ".pth":
                raise HTTPException(400, "the RVC model must be a .pth file")
            opt.rvc_model = jdir / "in" / "model.pth"
            await _save_upload(rvc_model, opt.rvc_model, MAX_MODEL)
        if rvc_index is not None and rvc_index.filename:
            if Path(rvc_index.filename).suffix.lower() != ".index":
                raise HTTPException(400, "the RVC index must be a .index file")
            opt.rvc_index = jdir / "in" / "model.index"
            await _save_upload(rvc_index, opt.rvc_index, MAX_MODEL)
        if template is not None and template.filename:
            if Path(template.filename).suffix.lower() != ".zip":
                raise HTTPException(400, "a custom template must be a UTAU voicebank .zip")
            opt.template = jdir / "in" / "template.zip"
            await _save_upload(template, opt.template, MAX_UPLOAD)
    except BaseException:
        shutil.rmtree(jdir, ignore_errors=True)
        raise
    job = Job(jid, jdir)
    with _lock:
        _cleanup_old()
        JOBS[jid] = job
    _executor.submit(_run_job, job, upload, opt)
    return JSONResponse(job.public(), status_code=202)


def _run_job(job: Job, upload: Path, opt: pipeline.Options) -> None:
    job.status = "running"

    def progress(stage: str, frac: float, msg: str) -> None:
        start, w = _STAGE_START.get(stage, (0.0, 0.0))
        job.stage, job.message = stage, msg
        job.progress = max(job.progress, min(0.999, start + w * max(0.0, min(1.0, frac))))
        if not job.log or job.log[-1] != msg:
            if frac in (0.0, 1.0) or not job.log:
                job.log.append(msg)

    try:
        res = pipeline.run(upload, job.dir / "work", job.dir / "out", opt, progress)
        job.report, job.bank_zip, job.dataset_zip, job.bank_dir = res.report, res.bank_zip, res.dataset_zip, res.bank_dir
        job.progress, job.status, job.message = 1.0, "done", "Done"
    except (pipeline.PipelineError, IngestError) as e:
        job.status, job.error, job.message = "error", str(e), str(e)
    except Exception as e:   # noqa: BLE001
        log.error("job %s failed:\n%s", job.id, traceback.format_exc())
        job.status, job.error, job.message = "error", f"Unexpected error: {type(e).__name__}: {e}", "Failed"
    finally:
        shutil.rmtree(job.dir / "work", ignore_errors=True)   # big intermediate wavs
        for p in (job.dir / "in").glob("*"):
            p.unlink(missing_ok=True)                           # uploaded originals (and model) no longer needed


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _job(job_id).public()


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str) -> FileResponse:
    j = _job(job_id)
    if not j.bank_zip or not j.bank_zip.exists():
        raise HTTPException(404, "voicebank not ready")
    return FileResponse(j.bank_zip, media_type="application/zip", filename=j.bank_zip.name)


@app.get("/api/jobs/{job_id}/dataset")
def dataset(job_id: str) -> FileResponse:
    j = _job(job_id)
    if not j.dataset_zip or not j.dataset_zip.exists():
        raise HTTPException(404, "no dataset for this job")
    return FileResponse(j.dataset_zip, media_type="application/zip", filename=j.dataset_zip.name)


@app.get("/api/jobs/{job_id}/sample/{key}.wav")
def sample(job_id: str, key: str) -> FileResponse:
    j = _job(job_id)
    if not re.fullmatch(r"[a-z_]{1,16}", key) or j.bank_dir is None:
        raise HTTPException(404, "no such sample")
    p = j.bank_dir / f"{key}.wav"
    if not p.exists():
        raise HTTPException(404, "no such sample")
    return FileResponse(p, media_type="audio/wav")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    j = _job(job_id)
    if j.status == "running":
        raise HTTPException(409, "job is still running")
    shutil.rmtree(j.dir, ignore_errors=True)
    with _lock:
        JOBS.pop(job_id, None)
    return {"deleted": job_id}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


def main() -> None:
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser(description="Voice-2-UTAU web app")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: local only; there is no login)")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
