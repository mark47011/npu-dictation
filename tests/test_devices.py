"""录音设备列表过滤测试 (mock sounddevice)"""
import sys
import types

import pytest

from dictation import Dictator


class FakeStream:
    def __init__(self, *a, **kw):
        pass

    def stop(self):
        pass

    def close(self):
        pass


def _devices():
    """模拟 PortAudio 全 API 设备视图:
    0=映射器(虚拟) 1=耳机MME(44100) 6=主声音捕获(虚拟) 7=耳机DS(44100) 15=耳机WASAPI(48000)
    """
    return [
        {"name": "Microsoft 声音映射器 - Input", "max_input_channels": 1, "hostapi": 0, "index": 0},
        {"name": "USB Audio Device", "max_input_channels": 1, "hostapi": 0, "index": 1},
        {"name": "主声音捕获驱动程序", "max_input_channels": 1, "hostapi": 1, "index": 6},
        {"name": "USB Audio Device", "max_input_channels": 1, "hostapi": 1, "index": 7},
        {"name": "USB Audio Device", "max_input_channels": 1, "hostapi": 2, "index": 15},
        {"name": "扬声器", "max_input_channels": 0, "hostapi": 0, "index": 3},  # 输出设备, 排除
    ]


def _make_sd(fail_16k_indices=frozenset()):
    """构造 fake sounddevice 模块: query_devices + InputStream (可指定 16k 失败设备)"""
    sd = types.ModuleType("sounddevice")
    devs = _devices()

    def query_devices(i=None, **kw):
        if i is None:
            return devs
        return devs[i]

    class InputStream(FakeStream):
        def __init__(self, samplerate, channels, dtype, device=None, **kw):
            if device in fail_16k_indices:
                raise Exception(f"Invalid sample rate (device {device})")
            FakeStream.__init__(self)

    sd.query_devices = query_devices
    sd.InputStream = InputStream
    return sd


@pytest.fixture
def fake_sd(monkeypatch):
    sd = _make_sd()
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd


def test_去重与过滤(fake_sd):
    """同物理设备多视图只留一个; 虚拟设备排除; 输出设备排除"""
    result = Dictator.list_input_devices()
    names = [n for _, n in result]
    assert names.count("USB Audio Device") == 1
    assert not any("映射器" in n or "主声音" in n for _, n in result)


def test_16k不兼容设备被过滤(monkeypatch):
    """WASAPI 48000 端点 (index 15) 开 16k 失败 -> 过滤"""
    sd = _make_sd(fail_16k_indices={15})
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    result = Dictator.list_input_devices()
    idxs = [i for i, _ in result]
    assert 15 not in idxs
    assert 1 in idxs  # MME 44100 保留


def test_无输入设备返回空(monkeypatch):
    sd = types.ModuleType("sounddevice")
    sd.query_devices = lambda i=None, **kw: [] if i is None else None
    sd.InputStream = FakeStream
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    assert Dictator.list_input_devices() == []


def test_枚举异常容错(monkeypatch):
    """query_devices 抛异常 -> 返回空列表不崩溃"""
    sd = types.ModuleType("sounddevice")
    sd.query_devices = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    assert Dictator.list_input_devices() == []


def test_设备解析_系统默认实时查询(dictator, monkeypatch):
    """input_device=None -> 实时查询系统默认"""
    d = dictator
    d.input_device = None
    sd = _make_sd()
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    assert d._resolve_input_device() == 1  # 默认输入 index 1


def test_设备解析_按名字匹配(dictator, monkeypatch):
    """手动选择存名字 -> 按名字找到当前 index (插拔后 index 变也能匹配)"""
    d = dictator
    d.input_device = "USB Audio Device"
    sd = _make_sd()
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    assert d._resolve_input_device() == 1


def test_设备解析_名字未找到回退枚举(dictator, monkeypatch):
    """手动选择的设备已拔 -> 回退枚举第一个可用设备 (不失败)"""
    d = dictator
    d.input_device = "已拔掉的设备"
    sd = _make_sd()
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    assert d._resolve_input_device() == 1  # 回退到枚举的第一个


def test_设备解析_旧版int兼容(dictator, monkeypatch):
    """旧 config 的 int index 直接使用"""
    d = dictator
    d.input_device = 7
    monkeypatch.setitem(sys.modules, "sounddevice", _make_sd())
    assert d._resolve_input_device() == 7

