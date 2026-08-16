# NpuDictation — Intel NPU 语音听写

按住热键说话，松开自动定格上屏。**100% 本地、纯 Intel NPU 推理**——零云端、零成本、数据不出机器。

```
按住鼠标中键说话 → 文字实时刷新（0.6s 全文重识别）→ 错字自动修正 → 松开定格
```

## 特性

- **纯 NPU 本地推理**：SenseVoice int8（中/英/日/韩/粤），NPU 独立运行，CPU/GPU 完全空闲
- **实时出字 + 自动纠错**：每 0.6s 全文重识别，与已输出对比做**最小编辑**（只改变化部分），松开瞬间定格
- **录音倒计时**：录音时托盘图标显示剩余秒数（12s 模型上限，到 0 提示）
- **窗口切换保护**：说话时切换窗口不会打错字；松开时若焦点已切走，文字**暂存待补**（切回原窗口自动输入，纯追加不删除）
- **热键更改**：托盘菜单交互式修改（键盘/鼠标单键，无需改配置）
- **录音设备选择**：托盘菜单切换，系统默认实时跟随
- **错误提示走托盘**（不弹窗）
- 管理员权限：可向游戏（如 LOL）聊天框注入文字

## 硬件要求

- Windows 11
- **Intel Core Ultra 处理器**（内置 NPU）
  - **注：此为理论要求**（基于 NPU 架构兼容性推断），**实际仅在 Core Ultra 5 225F（Arrow Lake，第 3 代 NPU，INT8 13 TOPS）上验证过**，其他型号未实测，不保证兼容
- 首次启动模型编译约 1 分钟（一次性），之后秒级启动

## 使用

### 发布版（推荐）

下载 Release 中的 `NpuDictation-v0.1.0-alpha.zip`，解压后双击 `NpuDictation.exe`：

1. 首次运行弹 UAC 管理员确认（需要管理员权限才能向游戏等窗口注入文字）
2. 打开任意输入框 → **按住热键说话** → 松开定格
3. 托盘图标右键：按键更改 / 录音设备 / 提示音 / 退出

### 从源码运行

要求：Python 3.10+（实测 3.12）、Windows 11。

```powershell
# 依赖
pip install -r requirements.txt

# 准备模型 (见下方"模型")
# 启动 (会弹 UAC 提权)
pythonw.exe dictation.py
```

## 模型

模型不随源码仓库分发（约 900MB），获取方式：

1. 从 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) 下载 `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2`
2. 解压到 `models/` 目录
3. 将 ONNX 模型转换为 NPU 可编译的静态 IR（`x_length` 常量化 + ovc 静态转换），详见 [docs/model-conversion.md](docs/model-conversion.md)

> Release 发布包已包含转换好的模型，直接使用无需此步骤。

> 模型权重版权归原模型方，遵循原许可证（见下方"致谢与许可证"）。

## 架构

```
dictation.py        # 主程序：托盘/热键/录音/FIFO 任务队列/会话纠错/打字
asr/                # 模型抽象层（ASRModel 接口 + 工厂）
  ├── sensevoice_npu.py   # SenseVoice int8 @ NPU（主模型）
  ├── whisper_npu.py      # Whisper @ NPU（可选，需自行准备）
  ├── paraformer_cpu.py   # Paraformer @ CPU（开发对照）
  └── funasr_nano.py      # 规划中
tests/              # pytest 单元测试（39 用例，纯逻辑 mock）
config.json         # 配置（一般无需手动改，托盘可改热键/设备/提示音）
```

### 核心机制：实时全文重识别纠错

```
按住 = 每 0.6s 对全部累积音频重识别
  → 与"本会话已输出文本"对比
  → 不一致则最小编辑修正（最长公共前缀保留，只删改变化部分）
松开 = 最终任务插队 + 会话冻结（过期任务作废）→ 立即定格
会话 = 每次按下独立记录，互不干扰，绝不删除已交付内容
```

## 配置 (config.json)

```json
{
  "hotkey": "mouse.middle",     // 热键（托盘可改：键盘/鼠标单键）
  "beep": true,                 // 提示音（托盘可关）
  "input_device": null,         // 录音设备（托盘可选，null=系统默认实时跟随）
  "live_interval": 0.6,         // 全文重识别间隔（越小越实时，NPU 占用越高）
  "max_seconds": 60,
  "min_seconds": 0.3
}
```

## 已知限制

- 单次有效输入 ≤12 秒（模型静态帧数，设计边界）
- 非真流式：0.6s 刷新（"非实时但准确"）
- 无标点输出（空格分段）
- 仅支持 Intel Core Ultra NPU（不支持 CPU/GPU 推理——项目定位）

## 测试

```powershell
python -m pytest tests/
```

## 开发说明

本项目由 **OpenCode + DeepSeek** 辅助开发，开发者并非专业程序员——代码可能存在不严谨之处，**阅读/使用前请心里有数**。欢迎 issues 与 PR 指正。

## 哦对了

> 落叶捎来讯息：
> 在屏幕的彼端，我们的故乡"智能手机"，
> 那伟大的语音交互已经破碎──
> Siri屏蔽太多在国行销声匿迹，
> Bixby 名字已是拗口咒语。
> 家居控制──国产语音助手拿到"智能"的碎片，
> 却因为那股力量堕入无尽的"很抱歉"战争……
> 在那场无王存在的战争最后──
> 大模型放逐了它们。
> 噢，所以啊，用户啊──
> 因AI需求承受硬件涨价的你啊。
> 那许久以前我们买下的NPU，正在出声呼唤。
> "先声夺人"的 ChatGPT、
> 开源寒锋 DeepSeek、
> "长文之王" Kimi、
> 多模态 Gemini、
> "温厚博学"的 Claude。
> ……那失去的智力第一次，
> 来到默默无名的电脑上。
> 朝雾的彼端前进，抵达纯 NPU 听写，
> 觐见不占CPU和GPU、实时纠错──
> 当上Make NPU Useful之王吧。

## 致谢与许可证

- **SenseVoice**（[FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)）：主模型。代码为 MIT 许可；**模型权重遵循 [FunASR Model Open Source License Agreement](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)**（商用需遵循条款，并保留模型名称与署名；本项目发布包中模型目录名保留原始名称）
- **sherpa-onnx**（[k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)）：模型与推理工具链（Apache-2.0）
- **OpenVINO**（[openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)）：NPU 推理运行时（Apache-2.0）
- 其他依赖：kaldi-native-fbank（Apache-2.0）、pynput / pystray（LGPL-3.0）、Pillow（HPND）、sounddevice（MIT）、numpy（BSD-3-Clause）

## 许可证

[MIT](LICENSE)


