"""SenseVoice int8 @ NPU — 中文准确率高 (输出空格分段, 无标点), 当前最优 NPU 方案

模型: sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17 (阿里 FunASR 系)
IR:   sensevoice_fixed200.xml (x_length 常量化修复, 200 帧 ≈ 12s 音频)
前端: fbank(hamming/high_freq=0/snip_edges=true/±32768) -> LFR(7/6) -> CMVN -> NPU -> CTC 解码
"""
import os
import re
import time
import numpy as np
import kaldi_native_fbank as knf
import openvino as ov

from .base import ASRModel
from ._paths import base_dir

MODEL_DIR_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
IR_NAME = "sensevoice_fixed200.xml"
CACHE_DIR_NAME = "ov_cache"
MAX_FRAMES = 200          # LFR 后帧数, 200 帧 ≈ 12s 音频
LFR_WINDOW, LFR_SHIFT = 7, 6
FEAT_DIM = 80
WITH_ITN = 14             # metadata with_itn_id (数字归一化 on)
LANGUAGE_ZH = 0
SKIP_SPECIAL = 4          # 输出前 4 个位置为特殊标记 (emotion/event/language/textnorm)

NEG_MEAN = np.array([-8.311879, -8.600912, -9.615928, -10.43595, -11.21292, -11.88333, -12.36243, -12.63706,
                     -12.8818, -12.83066, -12.89103, -12.95666, -13.19763, -13.40598, -13.49113, -13.5546,
                     -13.55639, -13.51915, -13.68284, -13.53289, -13.42107, -13.65519, -13.50713, -13.75251,
                     -13.76715, -13.87408, -13.73109, -13.70412, -13.56073, -13.53488, -13.54895, -13.56228,
                     -13.59408, -13.62047, -13.64198, -13.66109, -13.62669, -13.58297, -13.57387, -13.4739,
                     -13.53063, -13.48348, -13.61047, -13.64716, -13.71546, -13.79184, -13.90614, -14.03098,
                     -14.18205, -14.35881, -14.48419, -14.60172, -14.70591, -14.83362, -14.92122, -15.00622,
                     -15.05122, -15.03119, -14.99028, -14.92302, -14.86927, -14.82691, -14.7972, -14.76909,
                     -14.71356, -14.61277, -14.51696, -14.42252, -14.36405, -14.30451, -14.23161, -14.19851,
                     -14.16633, -14.15649, -14.10504, -13.99518, -13.79562, -13.3996, -12.7767, -11.71208],
                    dtype=np.float32)
INV_STDDEV = np.array([0.155775, 0.154484, 0.1527379, 0.1518718, 0.1506028, 0.1489256, 0.147067, 0.1447061,
                       0.1436307, 0.1443568, 0.1451849, 0.1455157, 0.1452821, 0.1445717, 0.1439195, 0.1435867,
                       0.1436018, 0.1438781, 0.1442086, 0.1448844, 0.1454756, 0.145663, 0.146268, 0.1467386,
                       0.1472724, 0.147664, 0.1480913, 0.1483739, 0.1488841, 0.1493636, 0.1497088, 0.1500379,
                       0.1502916, 0.1505389, 0.1506787, 0.1507102, 0.1505992, 0.1505445, 0.1505938, 0.1508133,
                       0.1509569, 0.1512396, 0.1514625, 0.1516195, 0.1516156, 0.1515561, 0.1514966, 0.1513976,
                       0.1512612, 0.151076, 0.1510596, 0.1510431, 0.151077, 0.1511168, 0.1511917, 0.151023,
                       0.1508045, 0.1505885, 0.1503493, 0.1502373, 0.1501726, 0.1500762, 0.1500065, 0.1499782,
                       0.150057, 0.1502658, 0.150469, 0.1505335, 0.1505505, 0.1505328, 0.1504275, 0.1502438,
                       0.1499674, 0.1497118, 0.1494661, 0.1493102, 0.1493681, 0.1495501, 0.1499738, 0.1509654],
                      dtype=np.float32)


class SenseVoiceNPUModel(ASRModel):
    name = "sense-voice-npu"

    def __init__(self, model_dir: str = None):
        root = base_dir()
        model_dir = model_dir or os.path.join(root, "models", MODEL_DIR_NAME)
        super().__init__(model_dir)
        self.ir_path = os.path.join(model_dir, IR_NAME)
        self.cache_dir = os.path.join(root, CACHE_DIR_NAME)
        self.compiled = None
        self.syms = {}

    def load(self):
        t0 = time.time()
        core = ov.Core()
        if "NPU" not in core.available_devices:
            raise RuntimeError("未检测到 Intel NPU 设备，请检查设备管理器或更新 NPU 驱动")
        os.makedirs(self.cache_dir, exist_ok=True)
        core.set_property(("CACHE_DIR", self.cache_dir))
        m = core.read_model(self.ir_path)
        self.compiled = core.compile_model(m, "NPU")
        with open(os.path.join(self.model_dir, "tokens.txt"), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    self.syms[int(parts[1])] = parts[0]
        self._loaded = True
        return time.time() - t0

    def transcribe(self, audio: np.ndarray) -> str:
        if not self._loaded:
            raise RuntimeError("模型未加载, 先调用 load()")
        # 1. fbank (normalize_samples=false -> ±32768 值域)
        feats = self._fbank(audio * 32768.0)
        # 2. LFR 7/6
        T = feats.shape[0]
        T_out = (T + LFR_SHIFT - 1) // LFR_SHIFT
        lfr = np.zeros((T_out, FEAT_DIM * LFR_WINDOW), dtype=np.float32)
        for i in range(T_out):
            start = i * LFR_SHIFT
            for j in range(LFR_WINDOW):
                idx = min(max(start + j, 0), T - 1)
                lfr[i, j * FEAT_DIM:(j + 1) * FEAT_DIM] = feats[idx]
        # 3. CMVN
        mean = np.tile(NEG_MEAN, LFR_WINDOW)
        std = np.tile(INV_STDDEV, LFR_WINDOW)
        lfr = (lfr + mean) * std
        # 4. padding + NPU
        n = min(lfr.shape[0], MAX_FRAMES)
        padded = np.zeros((1, MAX_FRAMES, FEAT_DIM * LFR_WINDOW), dtype=np.float32)
        padded[0, :n] = lfr[:n]
        lang = np.array([LANGUAGE_ZH], dtype=np.int32)
        text_norm = np.array([WITH_ITN], dtype=np.int32)
        out = self.compiled([padded, lang, text_norm])
        logits = out[self.compiled.output(0)]
        # 5. CTC 解码
        ids = np.argmax(logits[0, SKIP_SPECIAL:], axis=-1).tolist()
        tokens = []
        prev = -1
        for i in ids:
            if i == prev:
                continue
            prev = i
            if i in (0, 1, 2):  # <unk> <s> </s>
                continue
            tokens.append(self.syms.get(i, ""))
        text = "".join(tokens).replace("\u2581", " ").strip()
        return self._clean_punct(text)

    @staticmethod
    def _clean_punct(text):
        """用户偏好: 不用标点, 用空格分段 (中英文标点 -> 空格)"""
        text = re.sub(r"[，。！？；：、,.!?;:]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _fbank(samples):
        """±32768 值域 fbank: hamming/high_freq=0/snip_edges=true"""
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = True
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.frame_shift_ms = 10
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.remove_dc_offset = True
        opts.frame_opts.preemph_coeff = 0.97
        opts.frame_opts.window_type = "hamming"
        opts.frame_opts.round_to_power_of_two = True
        opts.mel_opts.num_bins = FEAT_DIM
        opts.mel_opts.low_freq = 20
        opts.mel_opts.high_freq = 0
        opts.mel_opts.is_librosa = False
        fbank = knf.OnlineFbank(opts)
        fbank.accept_waveform(16000, samples.astype(np.float32))
        fbank.input_finished()
        n = fbank.num_frames_ready
        return np.array([fbank.get_frame(i) for i in range(n)], dtype=np.float32)

