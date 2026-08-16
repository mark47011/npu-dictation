"""Paraformer-zh-small (sherpa-onnx) @ CPU — 中文更强, 参考用 (NPU 因动态 Loop 不可用)"""
import os
import numpy as np
import sherpa_onnx

from .base import ASRModel
from ._paths import base_dir

DEFAULT_MODEL = r"models\sherpa-onnx-paraformer-zh-small-2024-03-09"


class ParaformerCPUModel(ASRModel):
    name = "paraformer-cpu"

    def __init__(self, model_dir: str = None, num_threads: int = 2):
        model_dir = model_dir or os.path.join(base_dir(), DEFAULT_MODEL)
        super().__init__(model_dir)
        self.num_threads = num_threads

    def load(self):
        self.rec = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=os.path.join(self.model_dir, "model.int8.onnx"),
            tokens=os.path.join(self.model_dir, "tokens.txt"),
            num_threads=self.num_threads,
            sample_rate=16000,
            feature_dim=80,
        )
        self._loaded = True

    def transcribe(self, audio: np.ndarray) -> str:
        if not self._loaded:
            raise RuntimeError("模型未加载, 先调用 load()")
        s = self.rec.create_stream()
        # sherpa-onnx normalize_samples=false 时内部会 ×32768, 传 [-1,1] 即可
        s.accept_waveform(16000, audio.astype(np.float32))
        self.rec.decode_stream(s)
        return s.result.text.strip()

