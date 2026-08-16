"""ASR 模型统一接口"""
from abc import ABC, abstractmethod
import numpy as np


class ASRModel(ABC):
    """所有 ASR 模型必须实现此接口.

    音频输入约定: float32, 16kHz, 单声道, 范围 [-1, 1]
    """

    name: str = "base"

    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """加载模型 (首次可能耗时, 放后台线程调用)"""

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        """转写音频 -> 文本. audio: float32 16k mono [-1,1]"""

    def warmup(self) -> None:
        """可选: 预热 (跑一次空转写)"""

    def close(self) -> None:
        """可选: 释放资源"""
        self._loaded = False
