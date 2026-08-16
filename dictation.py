"""
NPU 全局热键听写 (托盘常驻版)
按住热键(默认 F8, 托盘可改, 支持键盘/鼠标单键)说话, 松开后实时全文重识别纠错并定格上屏.

- 实时: 每 0.6s 对全部累积音频重识别, 与已输出对比做最小编辑纠错 (保留前缀, 只改变化部分)
- 会话: 每次按下独立记录; 松开时最终任务插队 + 会话冻结 (防中间态闪烁)
- 窗口保护: 说话时切换窗口不会打错字; 松开时焦点已切走则暂存待补 (切回自动输入)
- 托盘: 蓝=空闲 红=录音(倒计时数字) 黄=转写 灰=加载 深红=错误 绿?=热键监听; 右键菜单改热键/设备/提示音
- 模型: sense-voice-npu (主) / whisper-npu / paraformer-cpu (可选)
- 日志: dictation.log (>1MB 自动轮转)
用法: pythonw.exe dictation.py (非管理员自动提权)
"""
import os
import sys
import json
import time
import logging
import threading
import winsound
import ctypes
import ctypes.wintypes as wt
from collections import deque

import numpy as np
import sounddevice as sd
from pynput import keyboard, mouse

from asr import create_model, base_dir

BASE = base_dir()
LOG_FILE = os.path.join(BASE, "dictation.log")
CONFIG_FILE = os.path.join(BASE, "config.json")
MUTEX_NAME = "NPU_Dictation_Mutex_20260813"

# 日志轮转: 超过 1MB 归档旧日志 (长期运行不无限增长)
if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 1_000_000:
    try:
        os.replace(LOG_FILE, LOG_FILE + ".1")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
logger = logging.getLogger("dictation")

DEFAULT_CONFIG = {
    "model": "sense-voice-npu",  # sense-voice-npu / whisper-npu / paraformer-cpu / funasr-nano
    "hotkey": "mouse.middle",    # 按住说话的热键 (f1-f20 / 单字符 / caps_lock / space / mouse.left|middle|right 等)
    "type_delay": 0.006,         # 打字逐字符间隔 (秒)
    "beep": True,                # 录音提示音
    "max_seconds": 60,           # 单次录音上限
    "min_seconds": 0.3,          # 小于此长度忽略
    "live_interval": 0.6,        # 全文重识别纠错间隔 (秒)
    "input_device": None,        # 录音设备索引 (None=系统默认, 托盘菜单可切换)
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as e:
            logger.warning(f"config.json 读取失败, 使用默认: {e}")
    return cfg


def save_config(cfg):
    """运行时修改 (如录音设备) 写回 config.json, 重启保持"""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"config.json 保存失败: {e}")


def is_first_run():
    """首次启动判定: ov_cache 无缓存文件 = 需要首次 NPU 编译 (约1-2分钟)"""
    cache = os.path.join(BASE, "ov_cache")
    if not os.path.isdir(cache):
        return True
    return not any(os.scandir(cache))


def parse_hotkey(s):
    s = str(s).strip().lower()
    special = {
        "caps_lock": keyboard.Key.caps_lock, "space": keyboard.Key.space,
        "enter": keyboard.Key.enter, "tab": keyboard.Key.tab,
        "esc": keyboard.Key.esc, "backspace": keyboard.Key.backspace,
        "scroll_lock": keyboard.Key.scroll_lock, "pause": keyboard.Key.pause,
        "insert": keyboard.Key.insert, "delete": keyboard.Key.delete,
        "home": keyboard.Key.home, "end": keyboard.Key.end,
        "page_up": keyboard.Key.page_up, "page_down": keyboard.Key.page_down,
        "left": keyboard.Key.left, "right": keyboard.Key.right,
        "up": keyboard.Key.up, "down": keyboard.Key.down,
    }
    if s in special:
        return special[s]
    if s.startswith("f") and s[1:].isdigit() and 1 <= int(s[1:]) <= 20:
        return getattr(keyboard.Key, s)
    if len(s) == 1:
        return keyboard.KeyCode.from_char(s)
    if s.startswith("vk:"):
        return keyboard.KeyCode.from_vk(int(s[3:]))
    raise ValueError(f"无法解析热键: {s!r}")


MOUSE_NAMES = {
    mouse.Button.left: "鼠标左键", mouse.Button.right: "鼠标右键",
    mouse.Button.middle: "鼠标中键", mouse.Button.x1: "鼠标侧键1",
    mouse.Button.x2: "鼠标侧键2",
}


def key_to_name(key):
    """pynput key/button -> 显示名"""
    if isinstance(key, mouse.Button):
        return MOUSE_NAMES.get(key, str(key))
    if isinstance(key, keyboard.Key):
        return key.name.upper()
    if isinstance(key, keyboard.KeyCode):
        return key.char.upper() if key.char else f"VK{key.vk}"
    return str(key)


def key_to_storage(key):
    """pynput key/button -> config 存储字符串"""
    if isinstance(key, mouse.Button):
        return f"mouse.{key.name}"
    if isinstance(key, keyboard.Key):
        return key.name
    if isinstance(key, keyboard.KeyCode):
        return key.char if key.char else f"vk:{key.vk}"
    return str(key)


def parse_hotkey_config(s):
    """config 字符串 -> (type, pynput对象); type: 'key' 或 'mouse'"""
    s = str(s).strip().lower()
    if s.startswith("mouse."):
        return ("mouse", mouse.Button[s.split(".", 1)[1]])
    return ("key", parse_hotkey(s))


# ---------- SendInput Unicode 打字 (支持中文) ----------
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002
INPUT_KEYBOARD = 1


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD),
                ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.c_void_p)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_void_p)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wt.DWORD), ("wParamL", wt.WORD), ("wParamH", wt.WORD)]


class INPUT(ctypes.Structure):
    """union 必须含全部 3 个成员, 否则 sizeof(INPUT) 不对 (正确=40, 只含 ki=32)"""
    class _U(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

    _anonymous_ = ("_u",)
    _fields_ = [("type", wt.DWORD), ("_u", _U)]


assert ctypes.sizeof(INPUT) == 40, f"INPUT size 应为 40, 实际 {ctypes.sizeof(INPUT)}"


def send_unicode_text(text, delay):
    """逐字符 SendInput (KEYEVENTF_UNICODE), 中英文均可"""
    for ch in text:
        if ch == "\n":
            ch = "\r"
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.ki.wVk = 0
        inp.ki.wScan = ord(ch)
        inp.ki.dwFlags = KEYEVENTF_UNICODE
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        inp.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        time.sleep(delay)


VK_BACK = 0x08


def send_backspace(n, delay=0.005):
    """按 n 次退格键 (删除已输出文字, 用于纠错)"""
    for _ in range(n):
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.ki.wVk = VK_BACK
        inp.ki.wScan = 0
        inp.ki.dwFlags = 0
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        inp.ki.dwFlags = KEYEVENTF_KEYUP
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        time.sleep(delay)


# ---------- 托盘图标 ----------
def make_icon(status="idle"):
    """64x64: 圆底 + '听'字 (蓝=空闲 红=录音 黄=转写 灰=加载 深红=错误 绿?=热键监听)"""
    from PIL import Image, ImageDraw, ImageFont
    colors = {"idle": (37, 99, 235, 255), "recording": (220, 38, 38, 255),
              "transcribing": (217, 119, 6, 255), "loading": (107, 114, 128, 255),
              "error": (153, 27, 27, 255), "listening": (22, 163, 74, 255)}
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([2, 2, 62, 62], fill=colors.get(status, colors["idle"]))
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 34)
    except Exception:
        font = ImageFont.load_default()
    if status == "listening":
        text = "?"
        bbox = d.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        d.text(((64 - w) / 2 - bbox[0], (64 - h) / 2 - bbox[1]), text, fill="white", font=font)
    else:
        d.text((15, 10), "听", fill="white", font=font)
    return img


def make_countdown_icon(seconds):
    """录音倒计时图标: 红底 + 剩余秒数 (12..0)"""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([2, 2, 62, 62], fill=(220, 38, 38, 255))
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 32)
    except Exception:
        font = ImageFont.load_default()
    text = str(max(0, seconds))
    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    # 用 bbox 左上角偏移修正居中 (textbbox 原点可能非 0)
    d.text(((64 - w) / 2 - bbox[0], (64 - h) / 2 - bbox[1]), text, fill="white", font=font)
    return img


# ---------- 听写器 ----------
class Dictator:
    def __init__(self, config, tray=None):
        self.config = config
        self.tray = tray
        # 热键: 支持键盘单键 + 鼠标单键 (无组合键)
        self.hotkey_type, self.hotkey = parse_hotkey_config(config["hotkey"])
        self.hotkey_display = key_to_name(self.hotkey)
        self.hotkey_listening = False   # 托盘"按键更改"监听模式
        self.mouse_listener = None
        self.live_interval = float(config.get("live_interval", 1.2))
        self.recording = False
        self.frames = []
        self.stream = None
        self.rec_start = 0.0
        # 任务队列: FIFO 严格按时间顺序处理 (不丢弃/不抑制/不合并)
        self.tasks = deque()
        self.task_cv = threading.Condition()
        self.ready = threading.Event()
        self.listener = None
        self._stop = False
        self.model = create_model(config["model"])
        self.input_device = config.get("input_device")  # 录音设备索引 (None=默认)
        # 会话内纠错状态: 每个 session 独立记录已输出文本
        self.session = 0                 # 当前录音会话 ID (每次录音 +1)
        self.session_outputs = {}        # session -> 该会话已输出文本
        self.session_frozen = {}         # session -> 已冻结 (松开后过期任务作废, 防中间态闪烁)
        self.target_hwnd = None          # 录音开始时的前台窗口 (窗口切换保护)
        self.pending = None              # 待补输出: {hwnd, text} (松开时焦点已切走, 切回自动补)
        self._had_devices = False        # 是否见过录音设备 (设备监控维护)
        self._last_devices_seen = 0.0    # 上次见到设备的时间 (区分"检测中"与"无设备")

    # ---- 状态 ----
    def set_status(self, status, title=None):
        if self.tray is not None:
            try:
                self.tray.icon = make_icon(status)
                if title:
                    self.tray.title = title
            except Exception as e:
                logger.warning(f"tray update failed: {e}")

    def start(self):
        """后台加载模型 + 启动热键监听"""
        def _load():
            try:
                logger.info(f"加载模型 {self.model.name} ...")
                if is_first_run():
                    self.set_status("loading", "NPU听写 - 首次启动加载中（约1-2分钟）")
                else:
                    self.set_status("loading", "NPU听写 - 加载中...")
                t0 = time.time()
                self.model.load()
                dt = time.time() - t0
                self.ready.set()
                logger.info(f"模型就绪 ({dt:.1f}s), 按住 {self.config['hotkey']} 说话")
                self.set_status("idle", f"NPU听写 - 按住 {self.config['hotkey']} 说话, 实时纠错上屏")
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                # 错误信息通过托盘图标显示 (深红 + tooltip), 不弹窗
                self.set_status("error", f"NPU听写 - 启动失败: {e}")

        threading.Thread(target=_load, daemon=True).start()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.mouse_listener = mouse.Listener(on_click=self._on_mouse_click)
        self.mouse_listener.start()
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._pending_loop, daemon=True).start()
        threading.Thread(target=self._device_watch_loop, daemon=True).start()

    def stop(self):
        self._stop = True
        if self.recording:
            self.recording = False
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        if self.listener:
            self.listener.stop()
        if self.mouse_listener:
            self.mouse_listener.stop()
        try:
            self.model.close()
        except Exception:
            pass

    # ---- 录音 (统一输出 float32 16k mono [-1,1]) ----
    def _resolve_input_device(self):
        """解析录音设备 (与硬件实时同步, 绝不返回 None 除非真无设备):
        1. 配置为 None -> 系统默认实时查询 -> 枚举兜底
        2. 配置为 str(设备名) -> 按名字在当前硬件中查找 (插拔后 index 会变, 名字稳定);
           设备已拔 -> 回退系统默认/枚举 (不失败)
        3. 配置为 int(旧版 index) -> 兼容直接使用
        """
        import sounddevice as sd   # 函数内 import: 便于测试 mock, 且每次实时查询
        try:
            devs = sd.query_devices()
        except Exception:
            devs = []
        # 1. 系统默认
        if self.input_device is None:
            try:
                return sd.query_devices(kind="input")["index"]
            except Exception:
                pass
        # 2. 按名字查找 (手动选择的设备, 跟随硬件变化)
        elif isinstance(self.input_device, str):
            name = self.input_device
            for i, d in enumerate(devs):
                if d["max_input_channels"] > 0 and d["name"] == name:
                    return i
            logger.info(f"录音设备 '{name}' 未在当前硬件中找到, 回退系统默认")
        # 3. 旧版 int index 兼容
        else:
            return self.input_device
        # 兜底: 枚举第一个可用输入设备 (跳过虚拟设备)
        for i, d in enumerate(devs):
            if d["max_input_channels"] > 0 and not any(
                    k in d["name"] for k in ("声音映射器", "主声音捕获", "驱动程序")):
                return i
        return None

    def start_recording(self):
        if self.recording or not self.ready.is_set():
            return
        self.recording = True
        self.frames = []
        self.rec_start = time.time()
        self.session += 1   # 按下 = 新会话
        self.session_outputs.setdefault(self.session, "")
        # 锁定目标窗口: 录音期间窗口切换时, 不输出到错误窗口
        self.target_hwnd = ctypes.windll.user32.GetForegroundWindow()
        try:
            buf = ctypes.create_unicode_buffer(256)
            ctypes.windll.user32.GetWindowTextW(self.target_hwnd, buf, 256)
            logger.info(f"锁定目标窗口: '{buf.value}' (hwnd={self.target_hwnd})")
        except Exception:
            logger.info(f"锁定目标窗口 hwnd={self.target_hwnd}")
        # 设备解析: 实时查询/枚举兜底, 避免落到 sounddevice 启动缓存 (-1)
        dev = self._resolve_input_device()
        try:
            self.stream = sd.InputStream(samplerate=16000, channels=1,
                                         dtype="float32", device=dev,
                                         callback=self._on_audio)
            self.stream.start()
        except Exception as e:
            # 重试: 设备枚举一次 (新插入设备可能未刷新)
            logger.warning(f"录音设备打开失败 ({e}), 重试枚举...")
            try:
                alt = None
                for i, d in enumerate(sd.query_devices()):
                    if d["max_input_channels"] > 0 and i != dev:
                        alt = i
                        break
                if alt is not None:
                    dev = alt
                    self.stream = sd.InputStream(samplerate=16000, channels=1,
                                                 dtype="float32", device=dev,
                                                 callback=self._on_audio)
                    self.stream.start()
                    logger.info(f"已切换备用录音设备: index={dev}")
                else:
                    raise
            except Exception as e2:
                logger.error(f"录音失败: {e2}")
                self.set_status("error", f"NPU听写 - 录音失败: {e2}")
                self.recording = False
                return
        # 日志记录实际录音设备 (便于验证系统默认跟随)
        try:
            dinfo = sd.query_devices(dev)
            logger.info(f"录音设备: {dinfo['name']} (index={dev})")
        except Exception:
            pass
        if self.config["beep"]:
            winsound.Beep(880, 80)
        logger.info("录音中... (实时纠错)")
        self.set_status("recording", "● 录音中... (松开热键)")
        threading.Thread(target=self._live_loop, daemon=True).start()
        threading.Thread(target=self._countdown_loop, daemon=True).start()

    def _countdown_loop(self, total=12):
        """录音倒计时: 每秒更新托盘图标为剩余秒数 (0 表示已到模型上限, 超出部分不转写).
        注意: set 图标在 sleep 之后 + 再检查 recording, 确保松开后绝不再覆盖状态图标
        """
        try:
            while self.recording:
                time.sleep(0.5)
                if not self.recording:
                    break
                remain = int(total - (time.time() - self.rec_start))
                if remain < 0:
                    remain = 0
                if self.tray is not None:
                    try:
                        self.tray.icon = make_countdown_icon(remain)
                        if remain <= 0:
                            self.tray.title = "● 已达 12 秒上限, 超出部分将不被转写"
                        else:
                            self.tray.title = f"● 录音中... 剩余 {remain}s (松开定格)"
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"倒计时异常: {e}")

    def _on_audio(self, indata, frames, t, status):
        if self.recording:
            self.frames.append(indata.copy())

    def _current_audio(self):
        return np.concatenate(self.frames).flatten() if self.frames else np.zeros(0, np.float32)

    def enqueue_full(self, session, audio, front=False):
        """全文重识别任务入队 (FIFO, 按时间顺序)
        front=True: 最终任务 (松开瞬间): 冻结会话 + 移除同会话旧任务 + 插队到队首
        """
        with self.task_cv:
            if front:
                self.session_frozen[session] = True
                # 移除同会话旧任务 (内容无损: 最终任务包含全部音频), 插到队首
                self.tasks = deque(t for t in self.tasks if t[1] != session)
                self.tasks.appendleft(("full", session, audio, True))
            else:
                self.tasks.append(("full", session, audio, False))
            self.task_cv.notify()

    def _live_loop(self):
        """每 live_interval 秒对全部累积音频重识别, 与已输出全文对比纠错"""
        last_check = 0.0
        try:
            while self.recording:
                time.sleep(0.1)
                if not self.recording:
                    break
                if time.time() - last_check >= self.live_interval:
                    last_check = time.time()
                    audio = self._current_audio()
                    if len(audio) >= int(self.config["min_seconds"] * 16000):
                        self.enqueue_full(self.session, audio.copy())
        except Exception as e:
            import traceback
            logger.error(f"live_loop 异常退出: {e}\n{traceback.format_exc()}")

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        audio = self._current_audio()
        dur = time.time() - self.rec_start
        if len(audio) == 0 or dur < self.config["min_seconds"]:
            logger.info("太短, 忽略")
            self.set_status("idle", f"NPU听写 - 按住 {self.config['hotkey']} 说话, 实时纠错上屏")
            return
        if len(audio) > self.config["max_seconds"] * 16000:
            audio = audio[:self.config["max_seconds"] * 16000]
        logger.info(f"录音 {dur:.1f}s, 最后纠错中...")
        self.set_status("transcribing", "转写中...")
        self.enqueue_full(self.session, audio.copy(), front=True)   # 插队: 松开立即定格

    def _focus_ok(self):
        """输出前检查: 焦点是否仍在录音开始时的目标窗口 (防打错窗口/删错内容)"""
        if not self.target_hwnd:
            return True
        return ctypes.windll.user32.GetForegroundWindow() == self.target_hwnd

    # ---- 录音设备 ----
    @staticmethod
    def list_input_devices():
        """返回 [(index, name), ...] 可用录音设备.
        策略: 全 API 枚举 + 按名字去重 (同一物理设备多 API 视图只留一个) +
              实测 16kHz 兼容性 (PortAudio 部分设备视图不接受 16k, 如 WASAPI 48000 端点)
        """
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            seen = set()
            result = []
            for i, d in enumerate(devs):
                if d["max_input_channels"] <= 0:
                    continue
                name = d["name"]
                # 过滤虚拟设备 (映射器/主声音捕获 = 默认设备别名, 避免混淆)
                if any(k in name for k in ("声音映射器", "主声音捕获", "驱动程序")):
                    continue
                if name in seen:
                    continue
                seen.add(name)
                # 16k 兼容性实测 (开/关一次流)
                try:
                    s = sd.InputStream(samplerate=16000, channels=1,
                                       dtype="float32", device=i)
                    s.stop()
                    s.close()
                except Exception:
                    continue  # 不能开 16k, 跳过
                result.append((i, name))
            if result:
                return result
        except Exception:
            pass
        return []

    def set_input_device(self, index):
        """切换录音设备 (下次录音生效), 持久化到 config.
        存储设备名而非 index: 硬件插拔后 index 会变, 名字稳定, 保证与硬件同步
        """
        name = None  # None = 系统默认
        if index is not None:
            for i, n in self.list_input_devices():
                if i == index:
                    name = n
                    break
        self.input_device = name
        self.config["input_device"] = name
        save_config(self.config)
        display = "系统默认" if name is None else name
        logger.info(f"录音设备已切换: {display} (下次录音生效)")
        if self.tray is not None:
            try:
                self.tray.title = f"NPU听写 - 录音设备: {display}"
            except Exception:
                pass

    def toggle_beep(self):
        """切换提示音开关, 持久化到 config"""
        self.config["beep"] = not self.config.get("beep", True)
        save_config(self.config)
        state = "开" if self.config["beep"] else "关"
        logger.info(f"提示音: {state}")
        if self.tray is not None:
            try:
                self.tray.title = f"NPU听写 - 提示音: {state}"
            except Exception:
                pass

    @staticmethod
    def _minimal_edit(prev, new):
        """最小编辑: 保留最长公共前后缀, 只替换中间变化部分.
        返回 (backspace次数, 需追加文本).
        光标在末尾, 所以删除 = 删到前缀后, 追加 = 重打分叉点之后的新文本.
        """
        if prev == new:
            return 0, ""
        if not new:
            return len(prev), ""
        if not prev:
            return 0, new
        # 最长公共前缀
        p = 0
        while p < len(prev) and p < len(new) and prev[p] == new[p]:
            p += 1
        # 删除: 前缀之后全部 (含被替换的中间与旧后缀), 追加: 分叉点后的新文本
        return len(prev) - p, new[p:]

    # ---- 转写 + 打字 (工作线程, FIFO 顺序处理) ----
    def _worker(self):
        while not self._stop:
            with self.task_cv:
                while not self.tasks and not self._stop:
                    self.task_cv.wait(timeout=0.5)
                if self._stop:
                    break
                item = self.tasks.popleft()
            self._process_item(item)
            # 仅非录音时恢复空闲 (录音中保持"录音中"状态, 不覆盖)
            if not self.recording:
                self.set_status("idle", f"NPU听写 - 按住 {self.config['hotkey']} 说话, 实时纠错上屏")

    def _process_item(self, item):
        """处理单个全文重识别任务 (独立方法, 便于单测).
        返回: True=已处理, False=跳过 (窗口保护/冻结)
        窗口保护: 非最终任务焦点切走=跳过; 最终任务焦点切走=暂存待补 (切回自动输入)
        """
        _, session, payload, is_final = item
        try:
            t0 = time.time()
            # 窗口切换保护: 焦点不在录音开始时的窗口
            if not self._focus_ok():
                if not is_final:
                    logger.info(f"会话#{session} 焦点已切换, 跳过输出 (文字保留在日志)")
                    return False
                # 最终任务: 继续转写, 转写后暂存待补
                text = self.model.transcribe(payload)
                if text:
                    self._set_pending(text)
                return False
            # 冻结检查1: 转写前 — 会话已冻结且非最终任务 -> 跳过 (不转写)
            if not is_final and self.session_frozen.get(session):
                logger.info(f"会话#{session} 已冻结, 跳过过期任务")
                return False
            # 会话内全文重识别纠错 (最小编辑: 只改变化部分, 不动保留部分)
            text = self.model.transcribe(payload)
            dt = time.time() - t0
            # 冻结检查2: 输出前 — 转写期间会话被冻结 (松开瞬间取到任务后 0.4s 窗口) -> 丢弃结果不输出
            if not is_final and self.session_frozen.get(session):
                logger.info(f"会话#{session} 转写期间冻结, 丢弃中间态")
                return False
            # 最终任务: 转写后焦点又切走 (转写 0.4s 窗口内) -> 暂存待补
            if is_final and not self._focus_ok():
                if text:
                    self._set_pending(text)
                return False
            cur = self.session_outputs.get(session, "")
            if text and text != cur:
                n_del, append = self._minimal_edit(cur, text)
                if n_del:
                    send_backspace(n_del)
                if append:
                    send_unicode_text(append, self.config["type_delay"])
                logger.info(f"会话#{session} 最小纠错: 删{n_del} 加{len(append)} [{cur}] -> [{text}]")
                self.session_outputs[session] = text
            elif not text:
                logger.info("全文转写为空")
            return True
        except Exception as e:
            logger.error(f"转写失败: {e}")
            return False

    def _set_pending(self, text):
        """暂存待补输出: 松开时焦点已切走, 切回锁定窗口后自动补输入 (纯追加, 不删除)"""
        self.pending = {"hwnd": self.target_hwnd, "text": text}
        logger.info(f"文字暂存待补: {text} (切回窗口后自动输入)")
        if self.tray is not None:
            try:
                self.tray.title = "文字待补: 切回原窗口后自动输入"
            except Exception:
                pass

    def _device_watch_loop(self):
        """监控录音设备变化 (硬件插拔): 设备列表变化时重建托盘菜单, 保持与硬件同步.
        注意: PortAudio 设备列表是进程级缓存, 插拔后需 _terminate/_initialize 才刷新
        """
        last = None
        while not self._stop:
            time.sleep(1)
            if self.recording:
                continue
            try:
                import sounddevice as sd
                # 重新初始化 PortAudio 刷新设备列表 (录音中跳过, 不影响活动流)
                try:
                    sd._terminate()
                    sd._initialize()
                except Exception:
                    pass
                names = [(i, d["name"]) for i, d in enumerate(sd.query_devices())
                         if d["max_input_channels"] > 0]
                if names:
                    self._had_devices = True
                    self._last_devices_seen = time.time()
                if names != last:
                    changed = last is not None
                    last = names
                    if changed and self.tray is not None:
                        try:
                            import pystray
                            self.tray.menu = build_menu(self, self.config, pystray)
                            logger.info(f"设备列表已更新: {[n for _, n in names]}")
                        except Exception:
                            pass
            except Exception as e:
                logger.error(f"设备监控异常: {e}")

    def _pending_loop(self):
        """轮询: 焦点回到锁定窗口时补输出暂存文字 (窗口失效则放弃)"""
        while not self._stop:
            time.sleep(0.5)
            if not self.pending:
                continue
            hwnd = self.pending["hwnd"]
            try:
                if not ctypes.windll.user32.IsWindow(hwnd):
                    logger.info("待补窗口已关闭, 放弃暂存文字")
                    self.pending = None
                    continue
                if ctypes.windll.user32.GetForegroundWindow() == hwnd:
                    text = self.pending["text"]
                    send_unicode_text(text, self.config["type_delay"])
                    logger.info(f"已补输出: {text}")
                    self.pending = None
                    if not self.recording:
                        self.set_status("idle", f"NPU听写 - 按住 {self.config['hotkey']} 说话, 实时纠错上屏")
            except Exception as e:
                logger.error(f"补输出异常: {e}")

    # ---- 热键事件 (回调异常不能杀死监听线程) ----
    def on_press(self, key):
        try:
            if self.hotkey_listening:
                self.set_hotkey(key)
                return
            if self.hotkey_type == "key" and key == self.hotkey:
                self.start_recording()
        except Exception as e:
            logger.error(f"on_press 异常: {e}")

    def on_release(self, key):
        try:
            if self.hotkey_listening:
                return
            if self.hotkey_type == "key" and key == self.hotkey:
                self.stop_recording()
        except Exception as e:
            logger.error(f"on_release 异常: {e}")

    def _on_mouse_click(self, x, y, button, pressed):
        """鼠标监听: 热键监听模式捕获 / 鼠标热键触发录音"""
        try:
            if self.hotkey_listening:
                if pressed:
                    self.set_hotkey(button)
                return
            if self.hotkey_type == "mouse" and button == self.hotkey:
                if pressed:
                    self.start_recording()
                else:
                    self.stop_recording()
        except Exception as e:
            logger.error(f"鼠标监听异常: {e}")

    def set_hotkey(self, key):
        """设置新热键 (键盘键或鼠标键), 持久化到 config"""
        self.hotkey_listening = False
        self.hotkey = key
        self.hotkey_type = "mouse" if isinstance(key, mouse.Button) else "key"
        self.hotkey_display = key_to_name(key)
        self.config["hotkey"] = key_to_storage(key)
        save_config(self.config)
        logger.info(f"热键已更改: {self.hotkey_display}")
        if self.config.get("beep", True):
            winsound.Beep(1320, 80)   # 成功高音
        self.set_status("idle", f"NPU听写 - 热键: {self.hotkey_display} (按住说话)")
        if self.tray is not None:
            try:
                self.tray.update_menu()   # 刷新菜单文字 (Windows 后端需显式重建)
            except Exception:
                pass

    def toggle_hotkey_listening(self):
        """托盘菜单: 进入/取消 热键监听模式 (图标变绿? + 提示音, 菜单关闭后仍可见)"""
        self.hotkey_listening = not self.hotkey_listening
        if self.hotkey_listening:
            logger.info("热键监听中: 按下新按键...")
            if self.config.get("beep", True):
                winsound.Beep(880, 80)
                winsound.Beep(660, 80)   # 双声 = 进入监听
            self.set_status("listening", "按下新按键... (再次点击菜单项取消)")
        else:
            logger.info("热键更改已取消")
            if self.config.get("beep", True):
                winsound.Beep(440, 80)
            self.set_status("idle", f"NPU听写 - 热键: {self.hotkey_display} (按住说话)")
        if self.tray is not None:
            try:
                self.tray.update_menu()
            except Exception:
                pass


# ---------- 单实例锁 ----------
def acquire_singleton():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW(None, False, MUTEX_NAME)
    return ctypes.get_last_error() != 183  # 183 = ERROR_ALREADY_EXISTS


def is_admin():
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def relaunch_as_admin():
    """非管理员时以管理员权限重启自己 (弹 UAC); 失败/被拒时提示"""
    result = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, "", BASE, 1)
    if result <= 32:  # ShellExecuteW 返回值 <=32 为失败
        ctypes.windll.user32.MessageBoxW(
            None,
            "NPU听写需要管理员权限才能运行（向游戏/其他窗口注入文字）。\n"
            "请在弹出的 UAC 窗口（\"是否允许此应用更改设备\"）中点击「是」。\n"
            "若没有弹出 UAC，请右键本程序 → 以管理员身份运行。",
            "NPU听写 - 需要管理员权限", 0x10)


# ---------- 主入口 ----------
def build_menu(d, config, pystray):
    """构建完整托盘菜单 (设备子菜单每次构建时实时枚举, 用于设备变化时重建)"""
    def on_quit(icon, item):
        logger.info("退出")
        icon.stop()
        d.stop()

    # 录音设备子菜单 (radio)
    def build_device_menu():
        def make_action(i):
            return lambda icon, item: d.set_input_device(i)
        def make_checked(name):
            return lambda item: d.input_device == name
        devices = Dictator.list_input_devices()
        if not devices:
            # 区分"检测中"与"无录音设备": 3 秒内曾见设备 = 正在变化/驱动加载, 否则稳定无设备
            if d._had_devices and time.time() - d._last_devices_seen < 3:
                return pystray.Menu(pystray.MenuItem("检测中…", None, enabled=False))
            return pystray.Menu(pystray.MenuItem("无录音设备", None, enabled=False))
        items = [pystray.MenuItem("系统默认",
                                  lambda icon, item: d.set_input_device(None),
                                  radio=True,
                                  checked=lambda item: d.input_device is None)]
        for idx, name in devices:
            short = name if len(name) <= 28 else name[:25] + "..."
            items.append(pystray.MenuItem(short, make_action(idx),
                                          radio=True, checked=make_checked(name)))
        return pystray.Menu(*items)

    def hotkey_label(item):
        if d.hotkey_listening:
            return "按下新按键... (再次点击取消)"
        return f"按键更改（当前 {d.hotkey_display}）"

    return pystray.Menu(
        pystray.MenuItem(lambda item: f"NPU听写 (按住 {d.hotkey_display} 说话, 实时纠错)",
                         None, enabled=False),
        pystray.MenuItem(f"模型: {d.model.name} ({config['model']})", None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(hotkey_label, lambda icon, item: d.toggle_hotkey_listening()),
        pystray.MenuItem("录音设备", build_device_menu()),
        pystray.MenuItem("提示音", lambda icon, item: d.toggle_beep(),
                         checked=lambda item: d.config.get("beep", True)),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("退出", on_quit),
    )


def main():
    # 管理员提权: 绿色软件双击即弹 UAC, 无需 .cmd 启动器
    if not is_admin():
        relaunch_as_admin()
        return

    config = load_config()
    if not acquire_singleton():
        logger.error("已有实例在运行")
        return

    import pystray

    d = Dictator(config)
    menu = build_menu(d, config, pystray)
    loading_title = "NPU听写 - 首次启动加载中（约1-2分钟）" if is_first_run() else "NPU听写 - 加载中..."
    icon = pystray.Icon("npu-dictation", make_icon("loading"), loading_title, menu)
    d.tray = icon

    d.start()
    icon.run()  # 阻塞在托盘消息循环


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        with open(os.path.join(BASE, "dictation_err.log"), "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        raise
