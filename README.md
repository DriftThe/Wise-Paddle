# Wise-Paddle

基于 PaddleOCR 的智能文档 OCR 服务，将图片或 PDF 中的版面区域自动识别并转化为结构化 Markdown 文本。

## 架构概览

Wise-Paddle 的处理流水线采用 **版面检测 → 过滤裁剪 → 视觉语言识别** 三段式设计：

```
PIL.Image（任意尺寸）
    │
    ▼
LayoutDetector（PP-DocLayoutV3）
    │ boxes / labels / scores
    ▼
BoxFilter（NMS + 面积/分数阈值 + unclip 扩框）
    │ 保留的 LayoutBox
    ▼
RegionCropper（numpy 切片裁剪）
    │
    ▼
VLPredictor（PaddleOCR-VL-1.6，按 label 分桶 batch 推理）
    │
    ▼
每张裁剪区域 → 一段 Markdown
```

**并发模型**：`PipelinePool`（asyncio.Queue）管理 N 个 `OCRPipeline` 实例；`BatchScheduler` 跨用户聚合 page 任务，按 max-min fairness 拼批后送给 pipeline 执行，layout detection 和 VL 推理均可 batch 加速。

## 快速开始

### 1. 环境准备

**硬件要求**：需要 GPU（推荐 CUDA 显存 ≥ 6GB），CPU 模式也可运行但速度极慢。

**Python 依赖**：

```bash
pip install -r requirements.txt
```
**Torch**：
请根据Torch官方文档选择合适的下载方式,`requirements.txt`中默认为cuda13.2版本
> 注意：`transformers` 需要 5.x 版本以支持 `AutoModelForImageTextToText`；项目内部包含了对 RoPE init 的兼容性补丁。

### 2. 下载模型

将以下两个模型放置到 `./model/` 目录下：

| 模型 | 用途 | 默认路径 |
|------|------|----------|
| PP-DocLayoutV3 | 版面区域检测（text / table / figure / title 等） | `./model/PP-DocLayoutV3` |
| PaddleOCR-VL-1.6 | 视觉语言识别（区域内容 → Markdown） | `./model/PaddleOCR-VL-1.6` |

模型可以从 PaddleOCR 官方仓库下载，需确保目录下包含 `config.json`、模型权重文件及 `preprocessor_config.json` 等完整文件。

### 3. 配置参数（.env）

项目所有可调参数通过 `.env` 文件驱动（不存在时则使用默认值）。

### 4. 启动服务

```bash
cd Wise-Paddle
uvicorn app:app --host 127.0.0.1 --port 8000
```

或直接运行：

```bash
python app.py
```

启动后访问 `http://127.0.0.1:8000` 即可使用内置的 Web UI（暗色主题单页应用）。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 前端 UI 页面 |
| `GET` | `/health` | 健康检查 + scheduler/pool 状态 |
| `POST` | `/ocr/upload` | 上传图片（multipart form），返回 OCR 结果 |
| `POST` | `/ocr/base64` | 上传图片（base64 JSON），返回 OCR 结果 |
| `POST` | `/ocr/pdf` | 上传 PDF，异步处理，返回 user_id + 轮询 URL |
| `GET` | `/api/ocr/pdf-status/{uid}` | 轮询 PDF 处理进度 |
| `POST` | `/api/cancel/{user_id}` | 取消某次 upload 的所有 pending + in-flight 任务 |
| `POST` | `/alive` | 前端心跳，刷新 voucher 保活倒计时 |

### 单图 OCR 示例

**上传文件方式**：

```bash
curl -X POST http://127.0.0.1:8000/ocr/upload \
  -F "file=@document.png"
```

**Base64 方式(未验证)**：

```bash
curl -X POST http://127.0.0.1:8000/ocr/base64 \
  -H "Content-Type: application/json" \
  -d '{"payload": "<base64编码的图片数据>"}'
```

**响应结构**：

```json
{
  "success": true,
  "request_id": "abc123",
  "status": "done",
  "elapsed_seconds": 2.35,
  "queue_wait_seconds": 0.12,
  "pages": [
    {
      "page_index": 0,
      "width": 1200,
      "height": 800,
      "image_b64": "<原图PNG base64>"
    }
  ],
  "regions": [
    {
      "page_index": 0,
      "box_index": 0,
      "label": "text",
      "score": 0.92,
      "rect": [100, 50, 600, 200],
      "markdown": "识别出的文本内容..."
    }
  ]
}
```

### PDF OCR 示例

```bash
# 1. 上传 PDF
curl -X POST http://127.0.0.1:8000/ocr/pdf \
  -F "file=@report.pdf"

# 响应（HTTP 202）:
# {"request_id": "xyz789", "status": "processing", "poll_url": "/api/ocr/pdf-status/xyz789"}

# 2. 轮询进度
curl http://127.0.0.1:8000/api/ocr/pdf-status/xyz789
```

### 取消任务

前端点击"移除"或页面关闭时，通过 `sendBeacon` / `fetch` 调用：

```bash
curl -X POST http://127.0.0.1:8000/api/cancel/abc123
```

服务端会取消该 `user_id` 的所有 pending + in-flight 工作（单图同步响应返回 `status=cancelled`，PDF 进度标记 `cancelled=true`）。会话级保活由 `/alive` 心跳维护，超时后服务端按 voucher 自动清理整批任务。

## 二次开发指南

### 项目结构

```
Wise-Paddle/
├── app.py              # FastAPI 服务入口（路由、配置、PDF 异步处理）
├── core_pipeline.py    # OCR 流水线核心（LayoutDetector / BoxFilter / RegionCropper / VLPredictor / OCRPipeline / PipelinePool / BatchScheduler）
├── static/
│   └── index.html      # 前端单页 UI（暗色主题，Canvas 交互）
├── model/              # 模型文件目录（需自行下载）
│   ├── PP-DocLayoutV3/
│   └── PaddleOCR-VL-1.6/
├── text_result/        # OCR 输出目录（每请求独立子目录）
└── logs/               # 日志目录（按日期 server-YYYY-MM-DD.log）
```

### 核心组件说明

#### 1. LayoutDetector（版面检测）

- 基于 PP-DocLayoutV3（`AutoModelForObjectDetection`），内部 processor 自动 resize 到 800×800
- 大图（长边 > 1600px）自动缩放防止显存溢出
- 输出：每页一组 `(box, label, score)` 三元组
- **扩展点**：可替换为其他版面检测模型（如 LayoutLMv3），只需保证输出格式兼容 `list[LayoutBox]`

#### 2. BoxFilter（框过滤 + NMS + 扩框）

- 纯 numpy 实现 NMS，无额外依赖
- `unclip_ratio` 和 `expand_pixels` 在 NMS **之后**扩框，给 VL 更多上下文——这是影响 OCR 准确率的关键参数
- **扩展点**：可自定义过滤逻辑（如按 label 类型差异化管理阈值），继承或替换 `BoxFilter.filter()` 方法

#### 3. VLPredictor（视觉语言识别）

- 基于 PaddleOCR-VL-1.6（`AutoModelForImageTextToText`）
- 按 label 分桶 batch 推理：同 label 的裁剪图共享 prompt，减少 forward 次数
- `DEFAULT_PROMPTS` 定义了每种 label 对应的 VL prompt（如 `text → "OCR:"`，`table → "Table Recognition:"`）
- **扩展点**：
  - 自定义 prompt：通过 `VLPredictor.__init__` 的 `prompts` 参数覆盖默认 prompt
  - 替换 VL 模型：可换为其他多模态模型（如 Qwen2-VL），需保证兼容 `AutoModelForImageTextToText` 和 `apply_chat_template` 接口
  - 调整 `max_forward_batch` 控制显存与速度的平衡

#### 4. OCRPipeline（流水线编排）

- 将上述三步串成一条主干：`detect → filter → crop → vl_recognize → 落盘`
- `process_page()` 处理单页，`process_batch()` 跨用户 batch 处理（layout 整批一次、VL 跨图按 label 聚合）
- 输出落盘到 `text_result/<request_id>/page_X_box_Y.md`
- **扩展点**：可在 `process_page()` / `process_batch()` 中插入自定义后处理步骤（如 Markdown 格式校正、结构化信息抽取）

#### 5. BatchScheduler（公平调度器）

- max-min fairness 策略：每个 user 独立 pending 队列，按已完成数排序，completed 少的优先分批
- 长任务（多页 PDF）不会独占 batch，新来的短任务立即被分配
- Worker 协程循环：等至少 1 个 job → 凑 batch（flush_ms 超时或满 max_batch）→ 借 pipeline → 跑批 → 分发结果
- **扩展点**：可修改 `_take_fair_batch()` 实现不同调度策略（如优先级权重、付费用户优先等）

#### 6. PipelinePool（Pipeline 池）

- `asyncio.Queue(maxsize=N)` 实现，`lease()` 上下文管理器借出/归还
- 每个槽位是一个完整的 OCRPipeline 实例（包含 layout + VL 两个模型）
- **扩展点**：可改为动态扩缩容（根据负载增减 pipeline 数），或引入优先级队列

### 常见二次开发场景

#### 场景 A：替换版面检测模型

1. 准备新模型的 HuggingFace 格式文件，放入 `./model/<your-model>/`
2. 设置 `.env` 中 `LAYOUT_MODEL_PATH=./model/<your-model>`
3. 确保 `AutoModelForObjectDetection.from_pretrained()` + `AutoImageProcessor.from_pretrained()` 能正确加载
4. 检查 `id2label` 映射是否与 PaddleOCR-VL 的 label 体系兼容

#### 场景 B：自定义 VL Prompt

在 `_make_pipeline()` 中传入自定义 prompts：

```python
custom_prompts = {
    "text": "请将图中文字逐行转录为纯文本：",
    "table": "请将表格转为 Markdown 表格格式：",
    "formula": "请将公式转为 LaTeX 格式：",
}
return OCRPipeline(
    ...,
    vl_prompts=custom_prompts,  # 需在 OCRPipeline.__init__ 传入 VLPredictor
)
```

#### 场景 C：添加后处理步骤

在 `OCRPipeline.process_page()` 或 `process_batch()` 的结果返回前，插入自定义处理：

```python
# 示例：对 table 类型的 markdown 做 HTML 格式修正
for region in regions:
    if region.label == "table":
        region.markdown = fix_table_markdown(region.markdown)
```

#### 场景 D：集成到现有系统

将 Wise-Paddle 作为微服务嵌入：

```python
# 外部系统调用示例
import httpx

async def ocr_image(image_bytes: bytes) -> dict:
    resp = await httpx.post(
        "http://wise-paddle:8000/ocr/upload",
        files={"file": ("doc.png", image_bytes)},
    )
    return resp.json()
```

或直接 import pipeline 模块：

```python
from core_pipeline import OCRPipeline, make_default_pipeline

pipeline = make_default_pipeline()
result = pipeline.process_page(pil_image, page_index=0)
print(result.markdown)
```

#### 场景 E：修改调度策略

`BatchScheduler._take_fair_batch()` 是纯逻辑方法，可直接替换为加权优先级：

```python
def _take_priority_batch(self, max_n: int) -> list[Job]:
    # 按用户优先级权重排序，而非 max-min fairness
    ...
```

### 性能调优建议

| 目标 | 调整方向 | 关键参数 |
|------|----------|----------|
| 降低显存占用 | 减少并行数 / 降低精度 | `PIPELINE_POOL_SIZE`, `VL_MAX_FORWARD_BATCH`, `DTYPE=float16` |
| 提升吞吐 | 增大 batch / 缩短 flush | `BATCH_MAX`, `BATCH_FLUSH_MS`, `PIPELINE_POOL_SIZE` |
| 提升 OCR 准确率 | 扩框 + 调 VL 参数 | `BOX_UNCLIP_RATIO`, `BOX_EXPAND_PIXELS`, `VL_REPETITION_PENALTY` |
| 处理大图 / 高清 PDF | 提高 DPI / 放宽像素限制 | `PDF_DPI`, `VL_MAX_PIXELS`, `VL_MIN_PIXELS` |

### 注意事项

- **RoPE 兼容性**：`core_pipeline.py` 启动时自动 patch `transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"]`，确保 transformers 5.x 与老模型兼容。如升级 transformers 版本后出现加载异常，请检查此 patch 是否生效。
- **VL 模型加载**：必须使用 `AutoModelForImageTextToText`（而非 `AutoModelForCausalLM`），否则模型无法看到图像输入，会纯靠文本编造。
- **Left Padding**：VL 推理 batch 模式需要 left-padding（`tokenizer.padding_side = "left"`），代码已自动设置，请勿覆盖。
- **资源释放**：前端通过 `/alive` 心跳保活，voucher 倒计时归零后服务端自动取消该会话全部任务；手动取消走 `/api/cancel/{user_id}`。后台标签页会被浏览器节流，请保持 `ALIVE_INITIAL_TTL` 足够大（默认 120）。

## 许可证

MIT License © 2026 DriftThe
