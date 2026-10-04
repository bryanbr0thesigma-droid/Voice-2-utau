#!/usr/bin/env python
"""Minimal RVC (v1/v2) inference CLI that works without fairseq.

Plug it into Voice-2-UTAU as the RVC command:

    export V2U_RVC_COMMAND='python tools/rvc_infer.py --input {input} --output {output} \
        --model {model} --index {index} --transpose {transpose}'

It uses the network code from the `rvc-python` wheel and replaces the one part that needs fairseq
(the HuBERT/ContentVec feature extractor) with the equivalent `transformers` model.

Install (see requirements-rvc.txt):  pip install -r requirements-rvc.txt && pip install --no-deps rvc-python

Safety: every checkpoint is loaded with torch's restricted `weights_only=True` unpickler, which refuses
anything but tensors/containers, so a booby-trapped .pth cannot run code. Models that need the full
unpickler are rejected rather than loaded.
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

CACHE = Path(os.environ.get("V2U_RVC_CACHE", Path.home() / ".cache" / "voice2utau" / "rvc"))
CONTENTVEC_REPO = "lengyue233/content-vec-best"        # ContentVec 768 = RVC's hubert_base, transformers format
RMVPE_REPO = "lj1995/VoiceConversionWebUI"             # the original RVC project's pitch model


def ensure_base_models(cache: Path) -> None:
    """Download the two base models once (both are plain weight files)."""
    from huggingface_hub import hf_hub_download
    rmvpe = cache / "base_model" / "rmvpe.pt"
    if not rmvpe.exists():
        hf_hub_download(RMVPE_REPO, "rmvpe.pt", local_dir=cache / "base_model")
    cv = cache / "contentvec"
    for f in ("config.json", "pytorch_model.bin"):
        if not (cv / f).exists():
            hf_hub_download(CONTENTVEC_REPO, f, local_dir=cv)


def install_fairseq_stub() -> None:
    """rvc_python imports fairseq at module level; we never call it, so a stub is enough."""
    if "fairseq" in sys.modules:
        return
    fs = types.ModuleType("fairseq")
    fs.checkpoint_utils = types.ModuleType("fairseq.checkpoint_utils")
    sys.modules["fairseq"] = fs
    sys.modules["fairseq.checkpoint_utils"] = fs.checkpoint_utils


def make_hubert(cache: Path, device: str):
    import torch
    from transformers import HubertConfig, HubertModel

    cfg = HubertConfig.from_pretrained(cache / "contentvec")
    model = HubertModel(cfg)
    final_proj = torch.nn.Linear(cfg.hidden_size, cfg.classifier_proj_size)
    sd = torch.load(cache / "contentvec" / "pytorch_model.bin", map_location="cpu", weights_only=True)
    proj = {k[len("final_proj."):]: v for k, v in sd.items() if k.startswith("final_proj.")}
    missing, _ = model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("final_proj.")}, strict=False)
    if [k for k in missing if "masked_spec_embed" not in k]:
        raise RuntimeError(f"ContentVec weights incomplete: {missing[:5]}")
    final_proj.load_state_dict(proj)

    class Hubert(torch.nn.Module):
        """Mimics the two members of the fairseq model that RVC calls."""

        def __init__(self):
            super().__init__()
            self.m, self.final_proj = model, final_proj

        def extract_features(self, source, padding_mask=None, output_layer=12):
            # fairseq: layer 12 of 12 = final output; layer 9 (v1) = after 10 layers
            hs = self.m(source, output_hidden_states=True).hidden_states
            return (hs[min(12, output_layer + 1)], None)

    return Hubert().to(device).float().eval()


def build_engine(cache: Path, device: str):
    install_fairseq_stub()
    import rvc_python.infer as rinfer
    import rvc_python.modules.vc.modules as vcm

    ensure_base_models(cache)
    rinfer.download_rvc_models = lambda lib_dir: None            # we manage base models ourselves
    hubert = make_hubert(cache, device)
    vcm.load_hubert = lambda config, lib_dir: hubert
    rvc = rinfer.RVCInference(models_dir=str(cache / "none"), device=device)
    rvc.vc.lib_dir = str(cache)                                  # pipeline looks for base_model/rmvpe.pt here
    return rvc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True, help="RVC .pth")
    ap.add_argument("--index", default="", help="optional .index (retrieval)")
    ap.add_argument("--transpose", type=int, default=0, help="semitones")
    ap.add_argument("--f0-method", default="rmvpe", choices=["rmvpe", "harvest", "pm", "crepe"])
    ap.add_argument("--index-rate", type=float, default=0.6)
    ap.add_argument("--protect", type=float, default=0.33)
    ap.add_argument("--rms-mix-rate", type=float, default=0.25)
    ap.add_argument("--device", default=os.environ.get("V2U_DEVICE", "cpu"),
                    help="'cpu' or e.g. 'cuda:0' (note: rvc_python treats exactly 'cpu' as full precision)")
    a = ap.parse_args(argv)
    if not Path(a.model).is_file():
        print(f"model not found: {a.model}", file=sys.stderr)
        return 2
    rvc = build_engine(CACHE, a.device)
    index = a.index if a.index and Path(a.index).is_file() else ""
    import torch
    version = torch.load(a.model, map_location="cpu", weights_only=True).get("version", "v1")
    rvc.load_model(a.model, version=version, index_path=index)
    rvc.set_params(f0method=a.f0_method, f0up_key=a.transpose, index_rate=a.index_rate if index else 0.0,
                   filter_radius=3, rms_mix_rate=a.rms_mix_rate, protect=a.protect)
    rvc.infer_file(a.input, a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
