"""共享测试 fixture: 构造 Dictator 测试实例 (不加载模型/NPU)"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import dictation as D


@pytest.fixture
def cfg():
    return dict(D.DEFAULT_CONFIG)


@pytest.fixture
def dictator(cfg):
    """构造 Dictator (懒加载模型, 不启动线程)"""
    d = D.Dictator(cfg)
    d._focus_ok = lambda: True   # 默认焦点正常 (窗口保护测试单独覆盖)
    d.model.transcribe = lambda audio: ""   # 默认空转写, 各测试覆盖
    return d
