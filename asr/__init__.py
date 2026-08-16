"""ASR 模型抽象层: 统一接口 + 注册表 + 工厂

切换模型只需改 config.json 的 "model" 字段:
  "sense-voice-npu" -> SenseVoice int8 @ NPU (默认, 中文准+空格分段, 已实测)
  "whisper-npu"     -> openvino-genai WhisperPipeline @ NPU (备选, 中文一般)
  "paraformer-cpu"  -> sherpa-onnx paraformer-zh-small @ CPU (对照用, 非 NPU)
  "funasr-nano"     -> OpenVINO funasr-nano @ NPU (规划中, 未实现)

音频输入统一: float32, 16kHz, 单声道, 范围 [-1, 1]
"""
import os
import sys

from ._paths import base_dir
from .base import ASRModel
from .sensevoice_npu import SenseVoiceNPUModel
from .funasr_nano import FunASRNanoModel

# paraformer-cpu / whisper-npu 是开发对照 (依赖 sherpa-onnx / openvino-genai), 发布版打包时排除
try:
    from .paraformer_cpu import ParaformerCPUModel
except ImportError:
    ParaformerCPUModel = None

try:
    from .whisper_npu import WhisperNPUModel
except ImportError:
    WhisperNPUModel = None



REGISTRY = {
    "sense-voice-npu": SenseVoiceNPUModel,
    "funasr-nano": FunASRNanoModel,
}
if ParaformerCPUModel is not None:
    REGISTRY["paraformer-cpu"] = ParaformerCPUModel
if WhisperNPUModel is not None:
    REGISTRY["whisper-npu"] = WhisperNPUModel


def create_model(name: str, model_dir: str = None) -> ASRModel:
    """按名称创建模型实例 (懒加载: 调用 load() 才真正加载)"""
    if name not in REGISTRY:
        raise ValueError(f"未知模型 '{name}', 可选: {list(REGISTRY)}")
    return REGISTRY[name](model_dir=model_dir)


__all__ = ["ASRModel", "create_model", "REGISTRY", "base_dir",
           "SenseVoiceNPUModel", "WhisperNPUModel", "ParaformerCPUModel", "FunASRNanoModel"]

