import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voice2utau import pipeline, server
from voice2utau.bank import write_bank, zip_bank
from voice2utau.extract import Clip
from voice2utau.morae import MORAE


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    server.JOBS.clear()

    def fake_run(upload, work, out, opt, progress=None):
        progress("ingest", 0.0, "start")
        progress("write", 1.0, "end")
        c = Clip(MORAE["a"], (0.3 * np.sin(np.arange(9000) / 20)).astype(np.float32), 0.02, 200.0)
        root = write_bank({"a": c}, out, opt.name, {"counts": {}})
        return pipeline.Result(root, zip_bank(root, out / "x.zip"), None,
                               {"counts": {"recorded": 1, "rvc": 0, "template_raw": 0, "missing": 100},
                                "samples": [{"key": "a", "source": "recorded"}], "warnings": [],
                                "model_seen": str(opt.rvc_model is not None)})
    monkeypatch.setattr(pipeline, "run", fake_run)
    return TestClient(server.app)


def wait(client, jid):
    for _ in range(200):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_full_job_cycle(client):
    r = client.post("/api/jobs", files={"file": ("v.mp3", b"abc"), "rvc_model": ("m.pth", b"w")},
                    data={"name": "Test", "gap_fill": "rvc"})
    assert r.status_code == 202
    j = wait(client, r.json()["id"])
    assert j["status"] == "done" and j["report"]["model_seen"] == "True"
    jid = j["id"]
    z = client.get(f"/api/jobs/{jid}/download")
    assert z.status_code == 200 and z.headers["content-type"] == "application/zip"
    assert client.get(f"/api/jobs/{jid}/sample/a.wav").status_code == 200
    assert client.get(f"/api/jobs/{jid}/sample/ka.wav").status_code == 404
    assert client.get(f"/api/jobs/{jid}/dataset").status_code == 404
    assert client.delete(f"/api/jobs/{jid}").status_code == 200
    assert client.get(f"/api/jobs/{jid}").status_code == 404


@pytest.mark.parametrize("files,data,code", [
    ({"file": ("evil.exe", b"x")}, {}, 400),
    ({"file": ("v.zip", b"x")}, {"gap_fill": "bogus"}, 400),
    ({"file": ("v.zip", b"x")}, {"language": "fr"}, 400),
    ({"file": ("v.zip", b"x")}, {"cross_check": "maybe"}, 400),
    ({"file": ("v.zip", b"x"), "rvc_model": ("m.bin", b"x")}, {}, 400),
    ({"file": ("v.zip", b"x"), "template": ("t.rar", b"x")}, {}, 400),
])
def test_input_validation(client, files, data, code):
    assert client.post("/api/jobs", files=files, data=data).status_code == code


def test_rejected_upload_leaves_no_files(client):
    client.post("/api/jobs", files={"file": ("v.zip", b"x"), "rvc_model": ("m.bin", b"x")})
    assert not list((server.DATA_DIR / "jobs").glob("*"))


def test_upload_size_limit(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_UPLOAD", 10)
    r = client.post("/api/jobs", files={"file": ("v.mp3", b"x" * 100)})
    assert r.status_code == 413 and not list((server.DATA_DIR / "jobs").glob("*"))


def test_job_id_traversal_and_unknown(client):
    assert client.get("/api/jobs/..%2F..%2Fetc").status_code == 404
    assert client.get("/api/jobs/" + "0" * 32).status_code == 404


def test_language_and_cross_check_reach_pipeline(client, monkeypatch):
    seen = {}
    real = pipeline.run

    def spy(upload, work, out, opt, progress=None):
        seen["lang"], seen["validate"] = opt.language, opt.validate
        return real(upload, work, out, opt, progress)
    monkeypatch.setattr(pipeline, "run", spy)
    jid = client.post("/api/jobs", files={"file": ("v.mp3", b"x")},
                      data={"language": "en", "cross_check": "off"}).json()["id"]
    wait(client, jid)
    assert seen == {"lang": "en", "validate": False}


def test_pipeline_error_reported(client, monkeypatch):
    def boom(*a, **k):
        raise pipeline.PipelineError("needs a model")
    monkeypatch.setattr(pipeline, "run", boom)
    jid = client.post("/api/jobs", files={"file": ("v.zip", b"x")}).json()["id"]
    j = wait(client, jid)
    assert j["status"] == "error" and j["error"] == "needs a model"


def test_index_and_config(client):
    assert "Voice" in client.get("/").text
    c = client.get("/api/config").json()
    assert len(c["units"]["ja"]) == 101 and len(c["units"]["en"]) == 675 and "rvc_available" in c
    assert {l["code"] for l in c["languages"]} == {"ja", "en"}
    assert c["units"]["en"][0] == {"key": "cv_b_aa", "alias": "b aa", "group": "CV · b"}
