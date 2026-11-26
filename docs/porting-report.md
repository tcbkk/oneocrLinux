# OneOCR 移植评估与实现报告

日期：2026-11-26 ｜ 对象：`D:\PROGRAM\CID_SYSTEM\SnippingToolOcr`（.NET Framework 4.7.2 + P/Invoke）

## 1. 结论

**可以移植，且已完成路线 A（模型解密 + 纯 ONNX Runtime）的可用实现。**

* 34 个加密子模型全部提取为明文 ONNX（`oneocr.onemodel` 58 MB → 34 个文件）；
* 主链路（检测 + 识别）**不依赖任何 Windows DLL**，在 Ubuntu 26.04 / x64 / CPU 上
  以 onnxruntime 1.32 跑通，输出与 Windows 原版逐行一致；
* 单页耗时 **0.40 s**（1400×520，2 核），启动加载约 0.5 s；
* 22 个辅助分类器仍依赖 `OneOCRFeatureExtract` 自定义算子，需路线 B 才能启用；
* 中文在这台机器上原版就输出 `?`（缺 OCR 语言包），移植版保持同一行为，不是回归。

## 2. 三个闭源件到底是什么

| 文件 | 大小 | 性质 | 关键证据 |
| --- | --- | --- | --- |
| `oneocr.dll` | 42.5 MB | PE32+ 闭源引擎，25 个 C 导出 | 编译路径 `C:\__w\1\s\CoreEngine\Native\CustomOps\...` |
| `oneocr.onemodel` | 58.4 MB | 加密模型容器 | 前 1 MB 熵 7.9998 bit/byte，256 种字节值全覆盖，无 `ONNX` 特征串 |
| `onnxruntime.dll` | 13.1 MB | 官方 ORT 1.19.0 x64 CPU | `ORT Build Info: git-commit-id=a873ee211` |

### 2.1 oneocr.dll 的依赖面

静态导入只有 4 个：

| DLL | 函数数 | 用途 |
| --- | --- | --- |
| `KERNEL32.dll` | 183 | 线程池 / SRW 锁 / 文件 / 堆，含 `SetThreadpoolTimer`、`GetNumaHighestNodeNumber` |
| `bcrypt.dll` | 12 | CNG 解密 + 校验 `oneocr.onemodel` |
| `dbghelp.dll` | 5 | 崩溃栈符号化 |
| `onnxruntime.dll` | 1（序号 1） | `OrtGetApiBase`，稳定 C ABI |

运行时动态加载：`combase.dll`（WinRT 初始化 / 包身份）、`advapi32.dll`、可选 `OpenCL.dll`
与 `QnnHtp.dll`。另有 14 个 `api-ms-win-*` API set（磁盘上无实体文件）。

**明确不依赖**：VC++ 运行库（静态 CRT）、系统 ICU（静态 ICU 71，`icudt71_dat` 内嵌）、
DirectX / DirectML / D3D（全文件无相关字符串 → 纯 CPU 推理）、`Windows.Media` / `Windows.Graphics`。

### 2.2 模型保护方式

* `oneocr.onemodel` 是加密容器，密钥由 `CreateOcrPipeline(modelPath, key, ...)` 传入，
  项目里的 key 是 24 个字符 + 6 个引号 + `4`；
* 解密后按子模型切片，逐个送给 ORT 的 `CreateSessionFromArray`；
* 引擎把 12 个 ONNX 自定义算子编进了 DLL（`com.microsoft.oneocr` 域），
  其中 11 个是后处理类算子，**只有 `OneOCRFeatureExtract` 真正出现在模型图里**。

## 3. 提取方法（路线 A 的核心）

`oneocr.dll` 只按**序号 1** 导入 `onnxruntime.dll`，因此可以放一个同名的垫片 DLL 在
DLL 搜索路径前面：

1. 垫片导出 `OrtGetApiBase @1`，内部 `LoadLibrary("onnxruntime_real.dll")` 加载真身；
2. 调用真身 `GetApi(19)` 拿到 `OrtApi` 函数表，**整表 memcpy 一份**；
3. 把副本里的 `CreateSessionFromArray` 和
   `CreateSessionFromArrayWithPrepackedWeightsContainer` 换成自己的钩子；
4. 钩子先把 `model_data` 落盘，再转调原函数；
5. 把副本的 `OrtApiBase`（`GetApi` / `GetVersionString`）返回给 `oneocr.dll`。

垫片 175 KB，静态 CRT，不需要任何第三方库。构建脚本 `tools/proxy/build.ps1`。

一次 OCR 调用即可触发全部 34 次 `CreateSessionFromArray`，得到：

```
model_001  11.5 MB  → detector（SegLink，FPN2/3/4，hori+vert）
model_002   3.5 MB  → 10 类书写系统分类器
model_003   0.9 MB  → 行布局分类器（含 OneOCRFeatureExtract）
model_004  13.4 MB  → 识别头
model_005  13.4 MB  → 识别头 32632 类
model_006–012       → 识别头 415/237/244/201/548/179/199 类
model_013–034       → 22 个 21 维特征二分类器（含 OneOCRFeatureExtract）
```

## 4. 移植后的实现

`oneocr_linux/` 只依赖 `onnxruntime + numpy + pillow`：

| 模块 | 职责 |
| --- | --- |
| `charset.py` | 词表映射 `chr(index + 32)`、CTC 贪心解码 |
| `detector.py` | 预处理（长边 1152 / 32 对齐 / 0–255）、SegLink 前向、连通域行分组 |
| `recognizer.py` | 裁剪缩放到高 60、`/255`、`seq_lengths = W/8`、CTC |
| `pipeline.py` | 引擎封装、坐标还原、排序 |
| `cli.py` | `oneocr-linux IMAGE` |

### 4.1 反推出来的关键参数

| 参数 | 值 | 怎么确定的 |
| --- | --- | --- |
| 检测输入尺寸 | 长边 1152，短边向上取整到 32 的倍数 | 钩子抓到 `[1,3,448,1152]`，原图 1400×520 |
| 检测归一化 | 无，0–255 | 钩子抓到张量 min/max = 0/255 |
| `im_info` | `[H, W, 1.0]` | 钩子抓到 `[448, 1152, 1.0]` |
| 识别输入 | `[1,3,60,W]`，`/255` 到 0–1 | 钩子抓到的 crop 张量 |
| `seq_lengths` | `max(1, W/8)` | 模型宽度总步长 8，输出 T = W/8 |
| 词表 | `chr(index + 32)` | 自造 `Hello 123` → id `40 69 76 76 79 17 18 19` = ASCII−32 |
| CTC blank | 最后一类（32632 头 = 32631） | id 序列中稳定出现在字符之间 |

### 4.2 检测后处理

检测头输出 19 个张量（3 层 × 6 路 + 1 个空 `feature_p2`），已过 sigmoid。实测：

* 文本行内 `link_scores` 全部 ≈ 1.0，行间没有连接；
* 所以可以直接对 `score > 0.5` 的 cell 做 8 邻域膨胀 + 连通域，取外接框即为行框；
* 与原版 `GetOcrLineBoundingBox` 相比偏差 4–8 px，只影响框的松紧，不影响识别。

原版真正的后处理（`SeglinkProposals` / `XYSeglinkProposals`）在 DLL 里，做了更细的
segment 合并和四角回归；当前实现用连通域近似，`bbox_deltas` 暂未使用。

### 4.3 实测结果

同一张 1400×520 图，Windows 原版与 Linux 移植版：

| 行 | 原版 `oneocr.dll` | `oneocr-linux` |
| --- | --- | --- |
| 1 | `????? Invoice 2026-11-26` | `????? Invoice 2026-11-26` |
| 2 | `????????????` | `????????????` |
| 3 | `???? 12,345.67` | `???? 12,345.67` |
| 4 | `Order No. A1B2-C3D4-E5F6` | `Order No. A1B2-C3D4-E5F6` |
| 5 | `?????????????????` | `?????????????????` |
| 6 | `support@example.com +86 755 1234 5678` | `support@example.com +86 755 1234 5678` |

服务器：Ubuntu 26.04 / x86_64 / 2 vCPU / 3.7 GB，Python 3.14.4，onnxruntime 1.32.0。
5 次平均 **0.401 s/页**，最快 0.389 s（不含进程启动与模型加载）。

## 5. 为什么中文是 `?`

* 原版 `oneocr.dll` 对中文行返回的就是字节 `0x3F`（`?`），不是 UTF-8/GBK 编码问题；
* 直接把同一行裁图喂给提取出的识别头，argmax 同样是 31（`chr(31+32)` = `?`）；
* 说明这台机器缺少对应语言的 OCR 模型/语言包，引擎把非 ASCII 结果统一退回 `?`；
* 因此 Linux 版复刻的是**同一行为**，不是移植缺陷。要出中文，需要先在 Windows 上
  确认原版能出中文（安装对应 OCR 语言包），再重新提取模型。

## 6. 剩余工作

| 优先级 | 事项 | 说明 |
| --- | --- | --- |
| 高 | 中文语言包 | 确认原版出中文后重新提取；词表映射 `chr(i+32)` 应同样成立 |
| 高 | `OneOCRFeatureExtract` | 用 C++ 复刻并注册为 ORT custom op，启用 22 个辅助分类器 |
| 中 | SegLink 精确后处理 | 用 `bbox_deltas` 做四角回归，替换连通域近似 |
| 中 | script → recognizer 标定 | 跑 `script_classifier` 的 10 类输出，建立到 9 个识别头的映射 |
| 中 | 竖排文本 | 使用 `scores_vert_*` / `bbox_deltas_vert_*` |
| 低 | 彩色通道顺序 | 灰度图无法区分 RGB/BGR，需用彩色样本验证 |

## 7. 路线对比（原始评估，仍有效）

| 路线 | 可行性 | 代价 | 结论 |
| --- | --- | --- | --- |
| A. 解密 + 纯 ONNX | 高 | 需一次性 Windows 提取；后处理要重写 | **已实现** |
| B. C++ 复刻自定义算子 | 中 | 12 个算子逆向 + 编译 .so | 只差 `OneOCRFeatureExtract` |
| C. Wine 跑原版 | 低 | WinRT/combase 残缺、GDI+ | 不推荐 |
| D. Windows 侧包成服务 | 高 | 多一台 Windows + 网络跳转 | 改动最小，精度不变 |
| E. 换 PaddleOCR/RapidOCR | 高 | 精度与语言覆盖不同 | 想要中文时更实际 |