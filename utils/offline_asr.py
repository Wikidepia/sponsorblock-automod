import json

import numpy as np
import onnxruntime as ort
import soundfile as sf
from pyctcdecode import build_ctcdecoder


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


class OfflineASR:
    def __init__(self, model_path: str, labels_path: str, langdet_path: str):
        self.decoder = build_ctcdecoder(json.load(open(labels_path)))
        self.session = ort.InferenceSession(model_path)
        self.langdet_session = ort.InferenceSession(langdet_path)
        self.langdet_label = ["ru", "en", "de", "es"]

    def langdet(self, audio):
        ort_inputs = {"input": [audio]}
        outs = self.langdet_session.run(None, ort_inputs)
        outs = np.argmax(softmax(outs[2]), axis=1)[0]
        return self.langdet_label[outs]

    def __call__(self, audio_path: str):
        audio, sr = sf.read(audio_path)
        assert sr == 16000
        audio = audio.astype(np.float32)
        if self.langdet(audio) != "en":
            raise Exception("Language is not English")
        ort_inputs = {"input": [audio]}
        ort_outs = self.session.run(None, ort_inputs)
        return self.decoder.decode(ort_outs[0][0])
