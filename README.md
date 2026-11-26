# oneocrLinux

把 Windows 版 OneOCR（微软截图工具自带的离线 OCR 引擎）迁移到 Linux 的**纯 ONNX Runtime** 实现。

运行时只需要 `onnxruntime + numpy + pillow`，不依赖 Wine、不依赖任何 Windows DLL。

```bash
oneocr-linux samples/sample.png
```

```
# samples/sample.png (1400x520, 6 lines)
????? Invoice 2026-11-26	[39.8, 68.8, 752.6, 112.3]
????????????	[44.2, 152.5, 271.7, 191.1]
???? 12,345.67	[43.6, 217.6, 306.4, 251.4]
Order No. A1B2-C3D4-E5F6	[38.0, 273.2, 589.1, 311.8]
?????????????????	[42.6, 342.8, 356.0, 381.5]
support@example.com +86 755 1234 5678	[27.6, 403.1, 866.8, 446.6]
```

> 输出里的 `?` 不是编码问题：原版 `oneocr.dll` 在同一台机器上对同样的图片也输出 `?`，
> 因为它把这台机器上没装对应 OCR 语言包的中文统一退回成 `?`。详见[已知限制](#已知限制)。

## 原理

Windows 版由三个闭源件组成：

| 文件 | 内容 |
| --- | --- |
| `oneocr.dll` | 闭源引擎，25 个 C 导出，静态 CRT + 静态 ICU，纯 CPU |
| `oneocr.onemodel` | 加密模型容器（58 MB，熵 ≈ 8.0），只有 `oneocr.dll` + 内置 key 能解 |
| `onnxruntime.dll` | 微软官方 ORT 1.19.0 x64 CPU |

`oneocr.dll` 静态导入 `onnxruntime.dll` **序号 1 = `OrtGetApiBase`**，推理时把解密后的
子模型通过 `OrtApi::CreateSessionFromArray` 交给 ORT。

`tools/proxy/` 里是一个 175 KB 的垫片 `onnxruntime.dll`：

1. 导出 `OrtGetApiBase @1`，加载真身 `onnxruntime_real.dll`；
2. 克隆真身的 `OrtApi` 函数表，替换 `CreateSessionFromArray*`；
3. 每次调用前把 `model_data` 原样落盘。

于是 `oneocr.onemodel` 解出来的 34 个明文 ONNX 就全部拿到了（`models/raw/`），
其中 12 个是主链路模型，其余 22 个是含自定义算子的辅助分类器。

## 模型清单（`models/onnx/`）

| 文件 | 来源 | 说明 |
| --- | --- | --- |
| `detector.onnx` | `model_001` | SegLink 文本检测，FPN2/3/4，hori+vert，纯 ORT 可跑 |
| `script_classifier.onnx` | `model_002` | 10 类书写系统分类（纯 ORT 可跑，当前未接入） |
| `rec_32632.onnx` | `model_005` | 默认识别头，覆盖 U+0020–U+7F97 |
| `rec_221 / 415 / 237 / 244 / 201 / 548 / 179 / 199` | `model_004/006–012` | 其余语言识别头 |

未纳入主链路（含 `com.microsoft.oneocr::OneOCRFeatureExtract` 自定义算子，标准 ORT 无法加载）：

* `model_003` 行布局分类器
* `model_013–023`、`model_024–034` 两级 21 维特征二分类器

这 22 个模型的 `feature/config` 是一段 4492 字节的浮点权重表，`OneOCRFeatureExtract`
把 21 维输入映射到 50 维再送 3 层 MLP。不影响主链路出字，只影响版面/语言后处理。

## 关键参数（全部由实测反推）

| 环节 | 参数 |
| --- | --- |
| 检测输入 | `[1,3,H,W]`，长边 1152、短边向上取整到 32 的倍数；**0–255 原始像素，不归一化** |
| 检测 `im_info` | `[H, W, 1.0]` |
| 检测输出 | `scores/bbox_deltas/link_scores` × `{hori,vert}` × `fpn{2,3,4}`，已过 sigmoid |
| 行分组 | `score > 0.5` 的 cell，8 邻域膨胀 1 格后取连通域（行内 link ≈ 1.0） |
| 识别输入 | `[1,3,60,W]`，`W` 按宽高比缩放并对齐到 8；**除以 255 归一化到 0–1** |
| `seq_lengths` | `max(1, W/8)` |
| 词表 | `chr(index + 32)`，**CTC blank = 最后一类**（32632 头即 32631） |

词表映射的验证方式：自造 `Hello 123` 图，识别头输出
`40 69 76 76 79 17 18 19`，恰好等于 `H e l l o 1 2 3` 的 ASCII 码减 32。

## 安装

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
```

模型文件不入库（约 58 MB）。在 Windows 上生成一次：

```powershell
# 需要 VS 2022/2026 的 x64 工具链
powershell -File tools\proxy\build.ps1
copy <原项目>\native\oneocr.dll        tools\runtime\
copy <原项目>\native\oneocr.onemodel   tools\runtime\
copy <原项目>\native\onnxruntime.dll   tools\runtime\onnxruntime_real.dll
python tools\extract_models.py --image samples\sample.png --out models\raw
```

然后把 `models/raw` 里的 12 个主链路模型按上表改名放进 `models/onnx/`。

## 部署到服务器

```bash
# 本机打包
tar -czf oneocr-linux.tar.gz oneocr_linux models/onnx pyproject.toml samples README.md

# 服务器
scp oneocr-linux.tar.gz user@host:/opt/
ssh user@host
cd /opt && tar -xzf oneocr-linux.tar.gz && cd oneocrLinux
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
oneocr-linux samples/sample.png
```

`deploy/install.sh` 封装了同样的步骤。

## 已知限制

1. **中文出 `?`**：本机（以及原版 DLL）对中文只能输出 `?`。原因是这台 Windows 没有安装
   中文 OCR 语言包时，引擎把所有非 ASCII 结果退回成 `?`；这不是移植引入的回归。
   若需要中文，需要先在 Windows 上确认原版能输出中文，再重新提取对应识别头。
2. **辅助分类器缺失**：行布局、21 维特征二分类共 22 个模型依赖 `OneOCRFeatureExtract`，
   需要按[路线 B](docs/porting-report.md) 用 C++ 复刻算子后才能启用。
3. **语言自动选择未接入**：`script_classifier.onnx` 可以跑，但 10 个 script_id 到 9 个识别头
   的对应关系还没有标定，当前固定使用最大的 32632 头。
4. **只支持横排文本行**：竖排依赖 `scores_vert_*` 与 `bbox_deltas_vert_*`，后处理尚未实现。
5. **检测框精度**：连通域近似，和原版框相比约有 4–8 px 偏差，不影响识别。

## 目录

```
oneocr_linux/        运行时（charset / detector / recognizer / pipeline / cli）
tools/proxy/         垫片 onnxruntime.dll 源码与构建脚本
tools/*.py           模型提取、参考输出采集、内存扫描等一次性工具
models/onnx/         主链路模型（不入库）
samples/             测试图与参考输出
tests/               单元测试
deploy/              服务器部署脚本
docs/                移植分析报告
```

## 许可

代码 MIT。模型权重来自微软 OneOCR，版权归微软所有，仅供研究/兼容性验证使用。