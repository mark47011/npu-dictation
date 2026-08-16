"""Whisper (openvino-genai) @ NPU — 默认模型, 已实测可用 (实时率 20x)"""
import os
import numpy as np
import openvino_genai as ov_genai

from .base import ASRModel
from ._paths import base_dir

DEFAULT_MODEL = r"models\whisper-base-int8-ov"


class WhisperNPUModel(ASRModel):
    name = "whisper-npu"

    def __init__(self, model_dir: str = None, device: str = "NPU"):
        super().__init__(model_dir or os.path.join(base_dir(), DEFAULT_MODEL))
        self.device = device
        self.pipe = None
        self.config = None

    def load(self):
        self.pipe = ov_genai.WhisperPipeline(self.model_dir, device=self.device)
        self.config = ov_genai.WhisperGenerationConfig()
        self.config.max_new_tokens = 128
        self._loaded = True

    def transcribe(self, audio: np.ndarray) -> str:
        if not self._loaded:
            raise RuntimeError("模型未加载, 先调用 load()")
        # whisper 要求 16k float32 [-1,1]
        return str(self.pipe.generate(audio, self.config)).strip()

    def close(self):
        self.pipe = None
        super().close()

