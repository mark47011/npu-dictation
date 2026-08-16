"""会话核心逻辑测试: 输出/纠错/冻结/窗口保护"""
import pytest

import dictation as D
from conftest import dictator, cfg  # noqa: F401


def _item(session, text, is_final=False):
    """构造 (kind, session, payload, is_final) 任务; transcribe 返回 text"""
    return ("full", session, text, is_final)


def test_新会话首次输出(dictator, monkeypatch):
    monkeypatch.setattr(D, "send_unicode_text", lambda t, d: D._sent.append(t))
    D._sent = []
    dictator.model.transcribe = lambda a: "你好"
    ok = dictator._process_item(_item(1, "audio"))
    assert ok is True
    assert D._sent == ["你好"]
    assert dictator.session_outputs[1] == "你好"


def test_追加纠错只加后缀(dictator, monkeypatch):
    monkeypatch.setattr(D, "send_unicode_text", lambda t, d: D._sent.append(t))
    monkeypatch.setattr(D, "send_backspace", lambda n: D._del.append(n))
    D._sent, D._del = [], []
    dictator.model.transcribe = lambda a: "你好世界"
    dictator.session_outputs[1] = "你好"
    dictator._process_item(_item(1, "audio"))
    assert D._sent == ["世界"]       # 只加后缀
    assert D._del == []             # 前缀不动, 不删
    assert dictator.session_outputs[1] == "你好世界"


def test_中间改字先删后加(dictator, monkeypatch):
    monkeypatch.setattr(D, "send_unicode_text", lambda t, d: D._sent.append(t))
    monkeypatch.setattr(D, "send_backspace", lambda n: D._del.append(n))
    D._sent, D._del = [], []
    dictator.model.transcribe = lambda a: "生存还是毁灭"
    dictator.session_outputs[1] = "生择还是毁灭"
    dictator._process_item(_item(1, "audio"))
    assert D._del == [5]
    assert D._sent == ["存还是毁灭"]
    assert dictator.session_outputs[1] == "生存还是毁灭"


def test_结果相同不输出(dictator, monkeypatch):
    monkeypatch.setattr(D, "send_unicode_text", lambda t, d: D._sent.append(t))
    D._sent = []
    dictator.model.transcribe = lambda a: "一样"
    dictator.session_outputs[1] = "一样"
    dictator._process_item(_item(1, "audio"))
    assert D._sent == []


def test_冻结后过期任务跳过(dictator, monkeypatch):
    monkeypatch.setattr(D, "send_unicode_text", lambda t, d: D._sent.append(t))
    D._sent = []
    dictator.session_frozen[1] = True
    dictator.model.transcribe = lambda a: "不该出现"
    ok = dictator._process_item(_item(1, "audio", is_final=False))
    assert ok is False
    assert D._sent == []
    assert 1 not in dictator.session_outputs


def test_最终任务不受冻结影响(dictator, monkeypatch):
    monkeypatch.setattr(D, "send_unicode_text", lambda t, d: D._sent.append(t))
    D._sent = []
    dictator.session_frozen[1] = True
    dictator.model.transcribe = lambda a: "定格版"
    ok = dictator._process_item(_item(1, "audio", is_final=True))
    assert ok is True
    assert D._sent == ["定格版"]
    assert dictator.session_outputs[1] == "定格版"


def test_窗口切换保护跳过(dictator, monkeypatch):
    monkeypatch.setattr(D, "send_unicode_text", lambda t, d: D._sent.append(t))
    D._sent = []
    dictator._focus_ok = lambda: False
    dictator.model.transcribe = lambda a: "不该输出"
    ok = dictator._process_item(_item(1, "audio"))
    assert ok is False
    assert D._sent == []
    assert 1 not in dictator.session_outputs


def test_最终任务焦点切走_暂存待补(dictator, monkeypatch):
    """松开时焦点不在 -> 不丢, 暂存 pending (切回自动补)"""
    monkeypatch.setattr(D, "send_unicode_text", lambda t, d: D._sent.append(t))
    D._sent = []
    dictator._focus_ok = lambda: False
    dictator.target_hwnd = 12345
    dictator.model.transcribe = lambda a: "最终文本"
    ok = dictator._process_item(_item(1, "audio", is_final=True))
    assert ok is False
    assert D._sent == []                     # 不输出
    assert dictator.pending == {"hwnd": 12345, "text": "最终文本"}  # 暂存
    assert 1 not in dictator.session_outputs


def test_最终任务焦点正常_不暂存(dictator, monkeypatch):
    monkeypatch.setattr(D, "send_unicode_text", lambda t, d: D._sent.append(t))
    D._sent = []
    dictator._focus_ok = lambda: True
    dictator.model.transcribe = lambda a: "正常输出"
    ok = dictator._process_item(_item(1, "audio", is_final=True))
    assert ok is True
    assert D._sent == ["正常输出"]
    assert dictator.pending is None


def test_空转写不更新(dictator, monkeypatch):
    monkeypatch.setattr(D, "send_unicode_text", lambda t, d: D._sent.append(t))
    D._sent = []
    dictator.model.transcribe = lambda a: ""
    dictator._process_item(_item(1, "audio"))
    assert D._sent == []
    assert 1 not in dictator.session_outputs
