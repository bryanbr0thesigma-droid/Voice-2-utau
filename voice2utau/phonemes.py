"""Language-independent phoneme recognition (wav2vec2 CTC, IPA output) with token timings."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio

MODEL_NAME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
FRAME_S = 0.02    # wav2vec2 stride


@dataclass
class Phone:
    sym: str
    start: float   # seconds, absolute in the source file
    end: float
    conf: float    # mean posterior over the token's frames


class PhonemeRecognizer:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCTC, Wav2Vec2FeatureExtractor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        vocab = json.load(open(hf_hub_download(model_name, "vocab.json"), encoding="utf-8"))
        self.id2tok = {v: k for k, v in vocab.items()}
        self.blank = vocab["<pad>"]
        self.skip = {vocab[t] for t in ("<pad>", "<s>", "</s>", "<unk>") if t in vocab}
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name).to(self.device).eval()

    def recognise_chunk(self, x16: np.ndarray, t0: float = 0.0) -> list[Phone]:
        torch = self.torch
        if len(x16) < audio.SR_REC // 10:
            return []
        inp = self.fe(x16, sampling_rate=audio.SR_REC, return_tensors="pt").input_values.to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(inp).logits[0].float(), dim=-1).cpu().numpy()
        ids = probs.argmax(-1)
        out: list[Phone] = []
        i, n = 0, len(ids)
        while i < n:
            tok = int(ids[i])
            j = i
            while j + 1 < n and ids[j + 1] == tok:
                j += 1
            if tok not in self.skip:
                conf = float(probs[i:j + 1, tok].mean())
                out.append(Phone(self.id2tok[tok], t0 + i * FRAME_S, t0 + (j + 1) * FRAME_S, conf))
            i = j + 1
        return out

    def recognise_file(self, wav16: Path, progress: Callable[[float], None] | None = None) -> list[Phone]:
        """Recognise a whole (possibly hours-long) 16 kHz wav, chunked at quiet points."""
        x, sr = audio.read_wav(wav16)
        assert sr == audio.SR_REC
        phones: list[Phone] = []
        bounds = audio.chunk_boundaries(x, sr)
        for k, (a, b) in enumerate(bounds):
            phones.extend(self.recognise_chunk(x[a:b], a / sr))
            if progress:
                progress((k + 1) / len(bounds))
        return phones
