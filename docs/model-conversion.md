# 模型转换指南（ONNX → NPU 静态 IR）

SenseVoice 官方 ONNX 模型含动态维度（`x_length` 等），NPU 编译器（vpux）只接受完全静态模型。本指南说明转换步骤。

## 原理

1. ONNX 层面把 `x_length` 输入替换为常量（200 = 模型固定帧数，≈12s 音频）
2. 用 OpenVINO 模型转换器（ovc）转换为静态 IR
3. NPU 编译静态 IR（首次 ~60s，之后缓存秒级）

## 步骤

### 1. 下载模型

从 [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) 下载：

```
sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
```

解压到 `models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/`

### 2. 替换 x_length 为常量

```python
import onnx
from onnx import numpy_helper
import numpy as np

SRC = "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx"
DST = "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/sensevoice_fixed200.onnx"
FIXED = 200  # 静态帧数 (LFR 后), 对应 ~12s 原始音频

m = onnx.load(SRC)
const = numpy_helper.from_array(np.array([FIXED], dtype=np.int32), name="x_length_const")
new_inputs = [i for i in m.graph.input if i.name != "x_length"]
del m.graph.input[:]
m.graph.input.extend(new_inputs)
m.graph.initializer.append(const)

def rename_in_nodes(nodes):
    for node in nodes:
        for j, inp in enumerate(node.input):
            if inp == "x_length":
                node.input[j] = "x_length_const"

rename_in_nodes(m.graph.node)
for node in m.graph.node:
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.GRAPH:
            rename_in_nodes(attr.g.node)
        elif attr.type == onnx.AttributeProto.GRAPHS:
            for g in attr.graphs:
                rename_in_nodes(g.node)

onnx.save(m, DST)
```

### 3. 静态转换 (ovc)

```powershell
python -m openvino.tools.ovc models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/sensevoice_fixed200.onnx `
  --input "x[1,200,560],language[1],text_norm[1]" `
  --output_model models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/sensevoice_fixed200.xml
```

### 4. 验证

```powershell
python -c "import openvino as ov; m=ov.Core().read_model(r'models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/sensevoice_fixed200.xml'); dyn=[op.get_friendly_name() for op in m.get_ops() for out in op.outputs() if out.get_partial_shape().is_dynamic]; print(f'dynamic outputs: {len(dyn)}')"
# 期望: dynamic outputs: 0 (完全静态才可 NPU 编译)
```

## 参数说明

| 参数 | 值 | 说明 |
|---|---|---|
| FIXED | 200 | LFR 后静态帧数；200×6×10ms = 12s 原始音频（模型上限） |
| x shape | [1,200,560] | 560 = 80 维 fbank × 7 (LFR window) |
| language / text_norm | 标量 | 语言 id (0=中文) / ITN (14=开) |

> 增大 FIXED 可提高音频上限（如 400=24s），但 NPU 编译时间和推理耗时同步增长。
