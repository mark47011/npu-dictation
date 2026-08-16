"""FunASR-Nano @ NPU — 规划中 (官方 notebook 支持 NPU, 中文更强)

参考: https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/funasr-nano
依赖: torch/nncf/optimum/optimum-intel/transformers/funasr (均需手动安装)
模型: FunAudioLLM/Fun-ASR-Nano-2512 (HF)
NPU 配置: NPU_USE_NPUW=YES, NPUW_LLM=YES, MAX_PROMPT_LEN=1024, NPUW_LLM_MIN_RESPONSE_LEN=512
"""
from .base import ASRModel


class FunASRNanoModel(ASRModel):
    name = "funasr-nano"

    def load(self):
        raise NotImplementedError(
            "funasr-nano 未实现. 需要先: "
            "1) pip install torch torchaudio nncf optimum optimum-intel transformers funasr (手动装) "
            "2) 下载模型 FunAudioLLM/Fun-ASR-Nano-2512 "
            "3) 按官方 notebook 转换为 OpenVINO IR 并验证 NPU 编译"
        )

    def transcribe(self, audio):
        raise RuntimeError("funasr-nano 未加载")
