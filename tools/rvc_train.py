#!/usr/bin/env python
"""Fine-tune an RVC v2 (32 kHz, f0) voice model on a folder of wavs, CPU-friendly and without the RVC WebUI code.

    python tools/rvc_train.py --dataset DIR --work WORK --name Near --out OUT --hours 3

Reuses the network code in the `rvc-python` wheel (see requirements-rvc.txt), the ContentVec/RMVPE helpers of
tools/rvc_infer.py and the official pretrained generator/discriminator (lj1995/VoiceConversionWebUI, pretrained_v2).
Writes OUT/<name>_s<step>.pth (loadable by tools/rvc_infer.py) every --save-every steps plus an OUT/<name>.index
retrieval index. Resumable: re-running with the same --work continues from WORK/state.pt.

Honest limits: this is a straight re-implementation of the standard RVC training step (mel L1 x45, KL, feature matching,
LSGAN), checked only by a short CPU run and by listening-free metrics; compare against the RVC WebUI before trusting it.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rvc_infer import CACHE, ensure_base_models, install_fairseq_stub, make_hubert  # noqa: E402

SR, HOP, N_FFT, N_MEL = 32000, 320, 1024, 80
SEG_FRAMES = 40                                       # 12800 samples at 32 kHz, as in RVC's configs
PRETRAINED = "lj1995/VoiceConversionWebUI"
CONFIG = [513, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]] * 3, [10, 8, 2, 2], 512, [20, 16, 4, 4],
          109, 256, SR]                               # same layout as the saved RVC v2 32k checkpoints


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ preprocessing
def prep(dataset: Path, work: Path, device: str) -> None:
    import librosa
    import soundfile as sf
    import torch
    from scipy import signal
    install_fairseq_stub()
    from rvc_python.lib.rmvpe import RMVPE
    from rvc_python.lib.slicer2 import Slicer

    ensure_base_models(CACHE)
    hubert = make_hubert(CACHE, device)
    rmvpe = RMVPE(str(CACHE / "base_model" / "rmvpe.pt"), is_half=False, device=device)
    bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=SR)
    slicer = Slicer(sr=SR, threshold=-42, min_length=1500, min_interval=400, hop_size=15, max_sil_kept=500)
    files = sorted(dataset.rglob("*.wav"))
    chunks: list[np.ndarray] = []
    for f in files:
        x, sr = sf.read(f, dtype="float32", always_2d=True)
        x = librosa.resample(x.mean(1), orig_sr=sr, target_sr=SR) if sr != SR else x.mean(1)
        x = signal.filtfilt(bh, ah, x).astype("float32")
        for seg in slicer.slice(x):
            per, ov = int(3.7 * SR), int(0.3 * SR)
            pos = 0
            while True:
                piece = seg[pos:pos + per + ov]
                if len(piece) >= 0.5 * SR:
                    chunks.append(piece)
                if pos + per + ov >= len(seg):
                    break
                pos += per
    log(f"{len(files)} files -> {len(chunks)} chunks, {sum(map(len, chunks)) / SR / 60:.1f} min")
    items = []
    t0 = time.time()
    for i, c in enumerate(chunks):
        mx = float(np.abs(c).max())
        if mx < 1e-3:
            continue
        c = (c / mx * 0.9 * 0.75 + (1 - 0.75) * c).astype("float32")
        w16 = librosa.resample(c, orig_sr=SR, target_sr=16000)
        with torch.no_grad():
            feats = hubert.extract_features(source=torch.from_numpy(w16).float().view(1, -1).to(device), output_layer=12)[0]
        phone = feats[0].cpu().numpy().astype("float16")                      # 50 Hz
        f0 = rmvpe.infer_from_audio(w16, thred=0.03).astype("float32")        # 100 Hz
        phone = np.repeat(phone, 2, axis=0)
        n = min(len(phone), len(f0), len(c) // HOP)
        if n < SEG_FRAMES:
            continue
        items.append({"phone": torch.from_numpy(phone[:n]), "f0": torch.from_numpy(f0[:n]),
                      "wav": torch.from_numpy(c[:n * HOP])})
        if (i + 1) % 20 == 0:
            log(f"features {i + 1}/{len(chunks)} ({time.time() - t0:.0f}s)")
    torch.save(items, work / "prep.pt")
    log(f"prepared {len(items)} training items")


def f0_coarse(f0: np.ndarray) -> np.ndarray:
    lo, hi = 1127 * math.log(1 + 50 / 700), 1127 * math.log(1 + 1100 / 700)
    mel = 1127 * np.log(1 + f0 / 700)
    mel[mel > 0] = (mel[mel > 0] - lo) * 254 / (hi - lo) + 1
    mel[mel <= 1] = 1
    mel[mel > 255] = 255
    return np.rint(mel).astype("int64")


# ------------------------------------------------------------------ spectrogram / losses
class Spec:
    def __init__(self):
        import librosa
        import torch
        self.win = torch.hann_window(N_FFT)
        self.basis = torch.from_numpy(librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MEL, fmin=0.0, fmax=None)).float()

    def linear(self, y):                                       # (B, T) -> (B, 513, frames)
        import torch
        import torch.nn.functional as F
        pad = (N_FFT - HOP) // 2
        y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        s = torch.stft(y, N_FFT, hop_length=HOP, win_length=N_FFT, window=self.win, center=False, return_complex=True)
        return torch.sqrt(s.real.pow(2) + s.imag.pow(2) + 1e-6)

    def mel(self, spec):
        import torch
        return torch.log(torch.clamp(torch.matmul(self.basis, spec), min=1e-5))


def kl_loss(z_p, logs_q, m_p, logs_p, mask):
    import torch
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p.float() - m_p.float()) ** 2) * torch.exp(-2.0 * logs_p.float())
    return torch.sum(kl * mask) / torch.sum(mask)


def feature_loss(fmap_r, fmap_g):
    import torch
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl.float().detach() - gl.float()))
    return loss * 2


def disc_loss(dr, dg):
    import torch
    return sum(torch.mean((1 - r.float()) ** 2) + torch.mean(g.float() ** 2) for r, g in zip(dr, dg))


def gen_loss(dg):
    import torch
    return sum(torch.mean((1 - g.float()) ** 2) for g in dg)


# ------------------------------------------------------------------ export / index
def export(net_g, path: Path, step: int) -> None:
    import torch
    sd = OrderedDict((k, v.detach().cpu().half()) for k, v in net_g.state_dict().items() if "enc_q" not in k)
    torch.save({"weight": sd, "config": CONFIG, "info": f"{step} steps (tools/rvc_train.py)", "sr": "32k", "f0": 1,
                "version": "v2"}, path)


def build_index(items, path: Path) -> None:
    import faiss
    feats = np.concatenate([it["phone"][::2].numpy().astype("float32") for it in items])
    np.random.default_rng(0).shuffle(feats)
    if len(feats) > 200_000:
        from sklearn.cluster import MiniBatchKMeans
        feats = MiniBatchKMeans(n_clusters=10_000, batch_size=4096, n_init=3, random_state=0).fit(feats).cluster_centers_
    n_ivf = max(1, min(int(16 * math.sqrt(len(feats))), len(feats) // 39))
    index = faiss.index_factory(768, f"IVF{n_ivf},Flat")
    index.train(feats)
    index.add(feats)
    faiss.write_index(index, str(path))
    log(f"index: {len(feats)} vectors, IVF{n_ivf} -> {path}")


# ------------------------------------------------------------------ training
def train(a) -> None:
    import torch
    from huggingface_hub import hf_hub_download
    install_fairseq_stub()
    from rvc_python.lib.infer_pack import commons
    from rvc_python.lib.infer_pack.models import MultiPeriodDiscriminatorV2, SynthesizerTrnMs768NSFsid

    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    work, out = Path(a.work), Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    if not (work / "prep.pt").exists():
        prep(Path(a.dataset), work, "cpu")
    items = torch.load(work / "prep.pt", weights_only=True)
    if not (out / f"{a.name}.index").exists():
        build_index(items, out / f"{a.name}.index")

    net_g = SynthesizerTrnMs768NSFsid(CONFIG[0], SEG_FRAMES, *CONFIG[2:], is_half=False)
    net_d = MultiPeriodDiscriminatorV2()
    og = torch.optim.AdamW(net_g.parameters(), a.lr, betas=(0.8, 0.99), eps=1e-9)
    od = torch.optim.AdamW(net_d.parameters(), a.lr, betas=(0.8, 0.99), eps=1e-9)
    step = 0
    state = work / "state.pt"
    if state.exists():
        ck = torch.load(state, map_location="cpu", weights_only=True)
        net_g.load_state_dict(ck["g"]); net_d.load_state_dict(ck["d"])
        og.load_state_dict(ck["og"]); od.load_state_dict(ck["od"]); step = ck["step"]
        log(f"resumed at step {step}")
    else:
        for net, fname in ((net_g, "pretrained_v2/f0G32k.pth"), (net_d, "pretrained_v2/f0D32k.pth")):
            p = hf_hub_download(PRETRAINED, fname, local_dir=CACHE / "pretrained")
            net.load_state_dict(torch.load(p, map_location="cpu", weights_only=True)["model"])
        log("loaded pretrained v2 32k generator/discriminator")
    net_g.train(); net_d.train()
    spec = Spec()
    rng = np.random.default_rng(step)
    t_start, t_log, last_save = time.time(), time.time(), step
    deadline = t_start + a.hours * 3600

    def batch():
        idx = rng.choice(len(items), a.batch, replace=len(items) < a.batch)
        T = max(len(items[i]["phone"]) for i in idx)
        B = len(idx)
        ph = torch.zeros(B, T, 768); pi = torch.zeros(B, T, dtype=torch.long); pf = torch.zeros(B, T)
        wv = torch.zeros(B, T * HOP); ln = torch.zeros(B, dtype=torch.long)
        for j, i in enumerate(idx):
            it = items[i]; n = len(it["phone"])
            ph[j, :n] = it["phone"].float(); pf[j, :n] = it["f0"]
            pi[j, :n] = torch.from_numpy(f0_coarse(it["f0"].numpy().copy())); wv[j, :n * HOP] = it["wav"]; ln[j] = n
        return ph, pi, pf, wv, ln

    while step < a.max_steps and time.time() < deadline:
        ph, pi, pf, wv, ln = batch()
        sp = spec.linear(wv)[:, :, :int(ln.max())]
        sid = torch.zeros(len(ln), dtype=torch.long)
        y_hat, ids, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(ph, ln, pi, pf, sp, ln, sid)
        mel = spec.mel(sp)
        y_mel = commons.slice_segments(mel, ids, SEG_FRAMES)
        y_hat_mel = spec.mel(spec.linear(y_hat.squeeze(1).float()))
        wave = commons.slice_segments(wv.unsqueeze(1), ids * HOP, SEG_FRAMES * HOP)
        dr, dg, _, _ = net_d(wave, y_hat.detach())
        l_d = disc_loss(dr, dg)
        od.zero_grad(); l_d.backward(); od.step()
        dr, dg, fr, fg = net_d(wave, y_hat)
        l_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel) * 45
        l_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
        l_g = gen_loss(dg) + feature_loss(fr, fg) + l_mel + l_kl
        og.zero_grad(); l_g.backward(); og.step()
        step += 1
        if step % a.log_every == 0 or step == 1:
            now = time.time()
            log(f"step {step} mel {l_mel.item():.2f} kl {l_kl.item():.2f} g {l_g.item():.1f} d {l_d.item():.2f} "
                f"({(now - t_log) / (1 if step == 1 else a.log_every):.1f}s/step)")
            t_log = now
        if step % a.save_every == 0:
            export(net_g, out / f"{a.name}_s{step}.pth", step)
            torch.save({"g": net_g.state_dict(), "d": net_d.state_dict(), "og": og.state_dict(), "od": od.state_dict(),
                        "step": step}, state)
            log(f"saved {a.name}_s{step}.pth"); last_save = step
    if step != last_save:
        export(net_g, out / f"{a.name}_s{step}.pth", step)
        torch.save({"g": net_g.state_dict(), "d": net_d.state_dict(), "og": og.state_dict(), "od": od.state_dict(),
                    "step": step}, state)
        log(f"saved {a.name}_s{step}.pth")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="voice")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--hours", type=float, default=3.0, help="wall-clock budget")
    ap.add_argument("--max-steps", type=int, default=10**9)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
