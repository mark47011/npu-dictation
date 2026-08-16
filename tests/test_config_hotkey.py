"""热键解析 + 配置加载测试"""
import pytest

from dictation import parse_hotkey, load_config, DEFAULT_CONFIG
from pynput import keyboard


def test_parse_f键():
    assert parse_hotkey("f8") == keyboard.Key.f8


def test_parse_特殊键():
    assert parse_hotkey("caps_lock") == keyboard.Key.caps_lock
    assert parse_hotkey("space") == keyboard.Key.space


def test_parse_单字符():
    assert parse_hotkey("a") == keyboard.KeyCode.from_char("a")


def test_parse_非法值():
    with pytest.raises(ValueError):
        parse_hotkey("xyz")
    with pytest.raises(ValueError):
        parse_hotkey("")


def test_load_config_默认合并(tmp_path, monkeypatch):
    """无配置文件 -> 全默认"""
    monkeypatch.setattr("dictation.CONFIG_FILE", str(tmp_path / "no.json"))
    cfg = load_config()
    assert cfg["hotkey"] == "mouse.middle"
    assert cfg["live_interval"] == 0.6


def test_load_config_自定义覆盖(tmp_path, monkeypatch):
    """配置文件覆盖默认值"""
    import json
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"hotkey": "caps_lock", "live_interval": 1.0}), encoding="utf-8")
    monkeypatch.setattr("dictation.CONFIG_FILE", str(p))
    cfg = load_config()
    assert cfg["hotkey"] == "caps_lock"
    assert cfg["live_interval"] == 1.0
    assert cfg["model"] == DEFAULT_CONFIG["model"]  # 未配置项用默认


def test_load_config_损坏json容错(tmp_path, monkeypatch):
    """损坏 json -> 回退默认, 不崩溃"""
    p = tmp_path / "config.json"
    p.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr("dictation.CONFIG_FILE", str(p))
    cfg = load_config()
    assert cfg["hotkey"] == "mouse.middle"
