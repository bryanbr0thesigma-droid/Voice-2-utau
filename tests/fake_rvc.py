"""Stand-in for an RVC CLI, used in tests: writes a 40 kHz, slightly quieter copy of the input.

Real RVC models output ~40/48 kHz audio of the same duration; this checks the same plumbing
(placeholders, resampling, batching and splitting) without needing a GPU or a model.
usage: fake_rvc.py INPUT OUTPUT [TRANSPOSE]
"""
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

x, sr = sf.read(sys.argv[1], dtype="float32")
y = resample_poly(x, 40000, sr).astype("float32") * 0.8
sf.write(sys.argv[2], y, 40000, subtype="PCM_16")
