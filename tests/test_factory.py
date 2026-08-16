"""模型工厂测试"""
import pytest

from asr import create_model, REGISTRY


def test_注册表包含核心模型():
    assert "sense-voice-npu" in REGISTRY


def test_未知模型报错():
    with pytest.raises(ValueError):
        create_model("不存在的模型")


def test_工厂返回正确类型():
    m = create_model("sense-voice-npu")
    assert m.name == "sense-voice-npu"
    assert m.loaded is False  # 懒加载


def test_工厂返回独立实例():
    a = create_model("sense-voice-npu")
    b = create_model("sense-voice-npu")
    assert a is not b
