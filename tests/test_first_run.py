"""首次启动判定测试"""
import pytest

from dictation import is_first_run


def test_is_first_run_无缓存目录(tmp_path, monkeypatch):
    monkeypatch.setattr("dictation.BASE", str(tmp_path))
    assert is_first_run() is True


def test_is_first_run_空缓存目录(tmp_path, monkeypatch):
    (tmp_path / "ov_cache").mkdir()
    monkeypatch.setattr("dictation.BASE", str(tmp_path))
    assert is_first_run() is True


def test_is_first_run_有缓存文件(tmp_path, monkeypatch):
    cache = tmp_path / "ov_cache"
    cache.mkdir()
    (cache / "123.blob").write_bytes(b"x")
    monkeypatch.setattr("dictation.BASE", str(tmp_path))
    assert is_first_run() is False
