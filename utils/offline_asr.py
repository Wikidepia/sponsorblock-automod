import json

import numpy as np
import onnxruntime as ort
import soundfile as sf
from pyctcdecode import build_ctcdecoder


class OfflineASR:
    def __init__(self, model_path: str, labels_path: str):
        self.model_path = model_path
        self.labels_path = labels_path
        self.decoder = build_ctcdecoder(json.load(open(labels_path)))
        self.session = ort.InferenceSession(model_path)

    def __call__(self, audio_path: str):
        audio, sr = sf.read(audio_path)
        assert sr == 16000
        audio = audio.astype(np.float32)
        ort_inputs = {"input": [audio]}
        ort_outs = self.session.run(None, ort_inputs)
        return self.decoder.decode(ort_outs[0][0])
