"""最小编辑逻辑测试"""
import pytest

from conftest import dictator  # noqa: F401
from dictation import Dictator


@pytest.mark.parametrize("prev,new,expect_del,expect_append", [
    ("你好", "你好", 0, ""),            # 相同 -> 不动
    ("你好", "你好世界", 0, "世界"),     # 追加 -> 只加
    ("你好世界", "你好", 2, ""),         # 缩短 -> 只删
    ("生择还是毁灭", "生存还是毁灭", 5, "存还是毁灭"),  # 中间改字 -> 保留前缀
    ("", "你好", 0, "你好"),            # 空 -> 全加
    ("你好", "", 2, ""),                # 删空
])
def test_minimal_edit(prev, new, expect_del, expect_append):
    n_del, append = Dictator._minimal_edit(prev, new)
    assert n_del == expect_del
    assert append == expect_append


def test_minimal_edit_前缀稳定():
    """长句只改尾部时, 前缀完全不动 (删 0)"""
    prev = "生存还是毁灭 这是个问题"
    new = "生存还是毁灭 这是个问题 追加内容"
    n_del, append = Dictator._minimal_edit(prev, new)
    assert n_del == 0
    assert append == " 追加内容"
