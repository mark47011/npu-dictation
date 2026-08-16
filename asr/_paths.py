"""项目根目录解析 (独立模块避免循环导入)"""
import os
import sys


def base_dir() -> str:
    """项目根目录: 源码运行=包上级; PyInstaller 打包=exe 所在目录 (模型/配置随 exe)"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
