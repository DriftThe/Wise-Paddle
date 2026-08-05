# Wise-Paddle 代码执行流详解

> ⚠️ **STALE — 这是 v0.4.0 的快照（重构前 baseline）**
> 当前代码（v0.6.0）与本文差异较大：心跳已改为 `/alive` voucher 机制（`_kick_dead_user_loop` + `scheduler.cancel_voucher`），主动取消走 `/api/cancel/{user_id}`，`/api/heartbeat`、`/api/release` 均已不存在；所有可调参数为 `.env` 驱动。
> **本文档仅作历史参考**。当前代码以 `app.py` / `core_pipeline.py` / `README.md` / `.env.example` 为准。

> **历史版本（v0.4.0）**：`core_pipeline.py` (1102 行) / `demo.py` (878 行) / `static/index.html` (~52KB)
> 原始目的：作为重构 baseline，每段代码都标注了源文件行号。

---

## 0. 顶层一句话

```
客户端 (HTML/JS) ──HTTP──> FastAPI (demo.py)
                           │
                           ├── 单图:  POST /ocr/upload | /ocr/base64   → 同步等结果
                           ├── 多页:  POST /ocr/pdf                    → 202 立即返回, 后台跑
                           ├── 心跳:  POST /api/heartbeat             → 续约
                           └── 释放:  POST /api/release               → 关 tab 主动 cancel
                                          │
                                          ▼
                                  BatchScheduler (跨用户拼批)
                                          │
                                          ▼
                                  PipelinePool (asyncio.Queue)
                                   │              │
                                   ▼              ▼
                              pipeline 1      pipeline 2     ← 同一进程, 各自加载一份模型
                                   │              │
                                   ▼              ▼
                              OCRPipeline.process_batch(jobs)
                                          │
                                          ▼
                          Layout detect → Box filter → Crop → VL forward (按 label 分桶)
                                          │
                                          ▼
                          text_result/{request_id}/page_X_box_Y_*.md
```

---

## 1. 模块依赖图

```
core_pipeline.py
├── _patch_rope_default()         # transformers 5.x 兼容
├── LayoutBox / RegionResult / PageResult / Job   # dataclass
├── LayoutDetector                 # PP-DocLayoutV3
├── BoxFilter                      # NMS + 面积 + 分数
├── RegionCropper                  # numpy 切片
├── VLPredictor                    # PaddleOCR-VL-1.6 (AutoModelForImageTextToText)
├── OCRPipeline                    # 把上面串起来
│   ├── process_page()             # 单图入口 (保留, 但 demo.py 不再走它)
│   └── process_batch()            # 批入口 (跨用户共享, 实际使用)
├── PipelinePool                   # asyncio.Queue 池, lease 上下文
├── BatchScheduler                 # 攒批 + 借 pipeline + 分发 + 心跳/释放
└── make_default_pipeline() / 常量

demo.py
├── _DailyFileHandler              # 按日期切日志
├── _cleanup_old_logs()            # 启动时清旧日志
├── _setup_logging()
├── lifespan()                     # 启动 BatchScheduler + pool + 守护 task
├── FastAPI app + 路由
│   ├── GET  /                     # 前端 SPA
│   ├── GET  /health
│   ├── POST /ocr/upload           # multipart 单图
│   ├── POST /ocr/base64           # JSON base64 单图
│   ├── POST /ocr/pdf              # multipart PDF → 202 立即返回
│   ├── GET  /api/ocr/pdf-status/{user_id}    # 轮询
│   ├── POST /api/heartbeat        # 续约
│   └── POST /api/release          # 主动取消
├── _process_pages()               # 核心 helper: 提交 + gather
├── _run_pdf_batch()               # PDF 后台 task
├── _pdf_progress_cleanup_loop()   # PDF 进度 10min 自动清
└── if __name__ == "__main__":     # uvicorn.run
```

---

## 2. `core_pipeline.py` 详解

### 2.1 兼容性补丁 (50-82 行)

```python
def _patch_rope_default() -> None:
    import transformers.modeling_rope_utils as r
    if "default" in r.ROPE_INIT_FUNCTIONS:
        return
    def _compute_default_rope_parameters(config, ...):
        base = config.rope_theta
        ...
    r.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

_patch_rope_default()  # 模块导入时立即执行
```

**为什么需要**：PaddleOCR-VL-1.6 是按老版 transformers 写的。transformers 5.x 把 `ROPE_INIT_FUNCTIONS["default"]` 入口干掉了，加载模型时会抛 `KeyError: 'default'`。我们手动补一个最低可用的实现：返回 `(inv_freq, attention_scaling)`。

**注意**：这个 patch 是**模块级副作用**——只要 import `core_pipeline` 就生效。重构时要保证 patch 在 `LayoutDetector` / `VLPredictor` 之前跑（目前是 top-level 顺序，OK）。

---

### 2.2 数据类 (87-143 行)

```python
@dataclass
class LayoutBox:
    xyxy: np.ndarray       # [x1, y1, x2, y2] 原图坐标系
    label_id: int
    label_name: str
    score: float
    # @property int_rect → 四舍五入到 int tuple
    # @property area → (x2-x1) * (y2-y1)

@dataclass
class RegionResult:
    page_index: int
    box_index: int
    label: str
    score: float
    rect: tuple[int, int, int, int]
    crop_shape: tuple[int, int]
    markdown: str
    md_path: Optional[Path] = None

@dataclass
class PageResult:
    page_index: int
    width: int
    height: int
    elapsed_seconds: float
    regions: list[RegionResult]
    # @property markdown → 拼 regions 的 markdown 串

@dataclass
class Job:
    request_id: str
    page_index: int
    image: Image.Image
    fut: asyncio.Future[PageResult]  # 用 Future 把 "提交" 和 "等结果" 解耦
```

**关键设计**：
- `Job.fut` 是 `asyncio.Future`——`scheduler.submit()` 同步返回 `Job`，worker 跑完调 `set_result()`，调用方 `await job.fut`。这让 `BatchScheduler` 不用 await 就能在 API handler 里同步 submit N 个 job 再一次性 `gather`。
- `Job.image` 是 PIL `Image`——已经是 `RGB` 模式（`demo.py` 在 submit 前 `.convert("RGB")` 强转）。

---

### 2.3 `LayoutDetector` (149-224 行)

```python
class LayoutDetector:
    def __init__(self, model_path, device, score_threshold=0.5, dtype=torch.float32):
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForObjectDetection.from_pretrained(model_path, dtype=dtype)\
                       .to(device).eval()
        self._max_long_side = 1600

    def _maybe_downscale(self, image): ...  # 长边 >1600 → 缩到 1600
    def detect(self, images: Sequence[Image.Image]) -> list[list[LayoutBox]]:
        # 1) 全部缩到 ≤1600 长边
        # 2) processor 自动 resize → tensor
        # 3) target_sizes = 原图尺寸, 让 post_process 反算回原图坐标
        # 4) post_process_object_detection(threshold=score_threshold) 一次初筛
        # 5) 对每个 box clamp 到 [0, w] / [0, h], 构造 LayoutBox
        # 6) 返回 list[list[LayoutBox]]  (per-page 的 box 列表)
```

**为什么是 `Sequence[Image.Image]` 一次性吃一批**：
- `AutoModelForObjectDetection` 内部对 N 张图做 batched forward，一次 GPU call 出 N 个结果。比循环 1 张 1 张快 2-3x。
- `max_long_side=1600` 是经验值：layout detect 在 800x800 resize 后跑推理，原图 ≥1600 不再额外信息；再大只是浪费显存和推理时间。

**Score threshold (默认 0.5)**：
- `post_process_object_detection` 里已经过滤一次 0.5 以下的低分框
- `BoxFilter` 还会再做一遍 0.5 → 双保险

---

### 2.4 `BoxFilter` (230-263 行)

```python
class BoxFilter:
    def __init__(self, iou_threshold=0.5, min_area=16*16, min_score=0.5):
        ...

    def _iou(self, a, b) -> float:
        # 标准 IoU = inter / union

    def filter(self, boxes: list[LayoutBox]) -> list[LayoutBox]:
        # 1) 按 score 倒序
        # 2) 跳过 score < min_score 或 area < min_area
        # 3) NMS: 如果跟已 keep 的 box IoU > iou_threshold, 跳过
        # 4) 否则 keep
```

**为什么还要 NMS**：`post_process_object_detection` 内部用的 NMS IoU 阈值（一般 0.11）太宽松，会留一堆重叠框。`BoxFilter` 用 0.5 严格去重——同一块区域只留一个最高分框。

**复杂度**：`O(N²)` 但 N 一般 < 50，无所谓。

---

### 2.5 `RegionCropper` (269-284 行)

```python
class RegionCropper:
    def crop(self, image, box) -> np.ndarray:
        # PIL → RGB numpy (H, W, 3) uint8
        # 拿 box.int_rect (四舍五入到 int)
        # clamp 到 [0, w]/[0, h]
        # numpy 切片 arr[y1:y2, x1:x2]
        # .copy() 切断对原图的引用, 防止后面 PIL 转 RGB 触发 copy-on-write
        # 退化: x2<=x1 或 y2<=y1 → 返回 1x1 黑图
```

**返回 numpy uint8 (H, W, 3)**：后面 `Image.fromarray(arr)` 转回 PIL，喂给 VL。

---

### 2.6 `VLPredictor` (290-456 行)

#### 2.6.1 初始化 (313-358 行)

```python
class VLPredictor:
    DEFAULT_PROMPTS = {
        "text": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart": "Chart Recognition:",
        "image": "OCR:",
        ...  # 其它都退化到 "OCR:"
        "_default": "OCR:",
    }

    def __init__(self, model_path, device, prompts=None,
                 dtype=torch.bfloat16, max_new_tokens=256,
                 max_pixels=1280*28*28, max_forward_batch=10,
                 attn_impl="sdpa"):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"  # 关键: decoder-only 必须左 padding
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, attn_implementation=attn_impl
        ).to(device).eval()
        # EOS = </s> (id=2), PAD = <unk> (id=0) — PaddleOCR-VL-1.6 的 generation_config
        self.eos_token_id = tok.convert_tokens_to_ids("</s>") or 2
        self.pad_token_id = pad or unk or 0
        # PaddleOCR-VL 1.6 的 image_processor.size 是 SizeDict,
        # 不能改 .min_pixels / .max_pixels 属性, 只能走 images_kwargs={"size": {...}}
        self.shortest_edge = 112896
        self.max_pixels = max_pixels
```

**几个关键决策**：

| 决策 | 为什么 |
|---|---|
| `AutoModelForImageTextToText` 而不是 `AutoModelForCausalLM` | 老入口类 `CausalLM` 不认 multimodal config，processor 拿不到 image token，结果 model 看不到图全靠编造。`ImageTextToText` 是官方入口。 |
| `padding_side = "left"` | decoder-only generate 时必须左 padding，否则 batch 里不同长度的序列右 padding 会污染 EOS 位置 |
| `repetition_penalty=1.15` | 防 batch 推理时陷入重复循环（官方没传，但加上稳） |
| `do_sample=False` | 确定性输出，方便调试 + 复现 |
| `images_kwargs={"size": {...}}` | v1.6 的 image_processor 没有 `min_pixels`/`max_pixels` 属性，必须走 `size` 字典覆盖 |

#### 2.6.2 `_build_messages` / `_forward_once` (363-417 行)

```python
def _build_messages(self, images, label):
    # 给每张图造一个 conversation (apply_chat_template 需要 list of conversations)
    return [[{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": self._prompt_for(label)},
    ]}] for img in images]

def _forward_once(self, images, label) -> list[str]:
    # 1) 构 conversations
    # 2) processor.apply_chat_template(
    #        conversations, add_generation_prompt=True,
    #        tokenize=True, return_dict=True, return_tensors="pt",
    #        padding=True,
    #        images_kwargs={"size": {"shortest_edge": 112896, "longest_edge": 1280*28*28}}
    #    ) → {input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw, ...}
    # 3) 日志: [VL fwd=N label=X] seq_len=... image_tokens=... 
    #    (image_tokens 用 mm_token_type_ids == 1 的个数统计, 验证 model 真的看到了图)
    # 4) model.generate(..., max_new_tokens=256, do_sample=False, eos/pad, use_cache, repetition_penalty=1.15)
    # 5) 切片 out[:, input_len:] → 只保留生成部分
    # 6) processor.batch_decode → 字符串列表
```

**`mm_token_type_ids`**：apply_chat_template 在 `return_dict=True` 时会返回这个字段，shape 跟 input_ids 一样，1 = image token，0 = text token。日志里统计 image_tokens > 0 是为了**确认 model 真的看到图**——这个 debug 习惯救过我好几次。

#### 2.6.3 `recognize_batch` (419-437 行)

```python
def recognize_batch(self, images, label) -> list[str]:
    if not images: return []
    cap = self.max_forward_batch
    if len(images) <= cap:
        return self._forward_once(images, label)
    # 拆 sub-batch, 一次 forward 最多 cap 张 (防 VRAM 爆)
    out = []
    for start in range(0, len(images), cap):
        chunk = list(images[start:start + cap])
        out.extend(self._forward_once(chunk, label))
    return out
```

**`max_forward_batch=4`** (OCRPipeline 构造时传过来)：单次 VL forward 最多 4 张图。RTX 4060 8GB 跑 4 张是安全阈值。`process_batch` 跨用户同 label 桶后可能一次有 10+ 张，这里负责拆。

#### 2.6.4 `recognize_grouped` (439-456 行)

```python
def recognize_grouped(self, items: list[tuple[Image.Image, str]]) -> list[str]:
    # 1) 按 label 分桶: buckets[label] = [(orig_idx, img), ...]
    # 2) 每个 bucket 调 recognize_batch (内部再拆 sub-batch)
    # 3) 按 orig_idx 回填 results
    # 4) 返回跟原 items 顺序对齐的字符串列表
```

**注意**：`recognize_grouped` 只在 `OCRPipeline.process_page` (单图路径) 用。`process_batch` (跨用户批路径) 是手动分桶 + 调 `recognize_batch`，逻辑等价但中间多暴露了 bucket dict。

---

### 2.7 `OCRPipeline` (462-652 行)

#### 2.7.1 构造 (465-490 行)

```python
class OCRPipeline:
    def __init__(self, layout_model_path, vl_model_path, device, output_dir,
                 box_filter=None, score_threshold=0.5, max_new_tokens=256,
                 max_regions=100, vl_max_forward_batch=4):
        self.layout = LayoutDetector(layout_model_path, device, score_threshold)
        self.filter = box_filter or BoxFilter()
        self.cropper = RegionCropper()
        self.vl = VLPredictor(vl_model_path, device,
                              max_new_tokens=max_new_tokens,
                              max_forward_batch=vl_max_forward_batch)
        self.max_regions = max_regions  # 单页最多送进 VL 的 region 数
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
```

**`max_regions=100`**：单页允许的 region 上限。PP-DocLayoutV3 单页一般出 5-30 个框，100 几乎不限。`process_page` / `process_batch` 都会 `kept[:max_regions]` 截断。

#### 2.7.2 `_ensure_output_dir` (492-494 行)

```python
def _ensure_output_dir(self):
    self.output_dir.mkdir(parents=True, exist_ok=True)
```

**为什么每次都 mkdir**：外部 `mavis-trash` 可能把 `text_result/` 干掉，下次跑批 `req_dir.mkdir(parents=True)` 会因为父目录不存在而失败。每批处理前都 `mkdir(parents=True)` 是便宜的保险。

#### 2.7.3 `process_page` (单图路径, 496-556 行)

```python
def process_page(self, image, page_index=0, request_id=None) -> PageResult:
    # 1) layout detect
    layouts = self.layout.detect([image])[0]
    # 2) filter + 截断
    kept = self.filter.filter(layouts)[:self.max_regions]
    # 3) crop
    crops = []
    for box in kept:
        arr = self.cropper.crop(image, box)
        if arr.size == 0 or arr.shape[0] < 2 or arr.shape[1] < 2: continue
        crops.append((Image.fromarray(arr), box))
    # 4) VL group-by-label
    items = [(img, box.label_name) for img, box in crops]
    markdowns = self.vl.recognize_grouped(items)
    # 5) 落盘 + 构造 PageResult
    if request_id is None:
        request_id = uuid.uuid4().hex[:8]
    req_dir = self.output_dir / request_id
    req_dir.mkdir(parents=True, exist_ok=True)
    regions = []
    for (img, box), md in zip(crops, markdowns):
        md_path = req_dir / f"page_{page_index}_box_{box.label_id}_{box.label_name}.md"
        md_path = self._unique_path(md_path)  # 同 (page,label) 重名自动 _1 _2 ...
        md_path.write_text(md or "", encoding="utf-8")
        regions.append(RegionResult(...))
    return PageResult(page_index, w, h, elapsed, regions)
```

**注意**：`process_page` 现在 demo.py **不再直接调用**——保留它是因为 (a) 老测试可能用到 (b) 单图调试方便。生产路径走的是 `process_batch`。

#### 2.7.4 `process_batch` (批路径, 569-652 行) ⭐ 核心

```python
def process_batch(self, jobs: list[Job]) -> list[PageResult]:
    if not jobs: return []
    self._ensure_output_dir()
    # 防御: request_id 缺失会污染根目录
    for j in jobs:
        if not j.request_id:
            raise ValueError("Job.request_id must be set; got empty/None")

    # ── 1) layout detect: 整批一次, 内部 batch
    images_rgb = [j.image.convert("RGB") for j in jobs]
    layouts_per_page = self.layout.detect(images_rgb)  # list[list[LayoutBox]]

    # ── 2) 每图独立 filter + crop
    crops_per_page = []
    for image, layouts in zip(images_rgb, layouts_per_page):
        kept = self.filter.filter(layouts)[:self.max_regions]
        crops = []
        for box in kept:
            arr = self.cropper.crop(image, box)
            if arr.size == 0 or arr.shape[0] < 2 or arr.shape[1] < 2: continue
            crops.append((Image.fromarray(arr), box))
        crops_per_page.append(crops)

    # ── 3) 跨图按 label 桶聚合 (同一 bucket 一次 VL forward)
    # bucket: label_name -> [(page_idx, crop_idx, crop_img), ...]
    buckets = {}
    for p_idx, crops in enumerate(crops_per_page):
        for c_idx, (img, box) in enumerate(crops):
            buckets.setdefault(box.label_name, []).append((p_idx, c_idx, img))

    # ── 4) 每个 bucket 调一次 VL batch
    md_lookup = {}
    for label, items in buckets.items():
        imgs = [it[2] for it in items]
        texts = self.vl.recognize_batch(imgs, label)  # 内部再拆 sub-batch
        for (p_idx, c_idx, _), md in zip(items, texts):
            md_lookup[(p_idx, c_idx)] = md

    # ── 5) 写文件 + 构造 PageResult
    results = []
    for p_idx, (job, crops) in enumerate(zip(jobs, crops_per_page)):
        image = images_rgb[p_idx]
        req_dir = self.output_dir / job.request_id
        req_dir.mkdir(parents=True, exist_ok=True)
        regions = []
        for c_idx, (img, box) in enumerate(crops):
            md = md_lookup.get((p_idx, c_idx), "")
            md_path = req_dir / f"page_{job.page_index}_box_{box.label_id}_{box.label_name}.md"
            md_path = self._unique_path(md_path)
            md_path.write_text(md, encoding="utf-8")
            regions.append(RegionResult(...))
        results.append(PageResult(
            page_index=job.page_index,
            width=image.width, height=image.height,
            elapsed_seconds=time.perf_counter() - st,  # 整批 wall time (粗略)
            regions=regions,
        ))
    return results
```

**关键点**：
- **`elapsed_seconds` 是整批 wall time**，不是单 page 的——`process_batch` 内部只测了一次 `time.perf_counter()`。每个 PageResult 都拿到同一个 batch elapsed，**不准确**（前几个 page 应该比后几个 page 时间长）。如果前端要 per-page elapsed 需要改。
- **跨用户共享 label 桶**——A 用户的 text crop 和 B 用户的 text crop 在同一个 `text` bucket 里被一次 VL forward 跑完。这才是 `BatchScheduler` 攒批真正的价值。
- **落盘路径 `text_result/{request_id}/page_X_box_Y_*.md`**——`request_id` 隔离不同请求的输出，跨用户不冲突。

---

### 2.8 `PipelinePool` (658-701 行)

```python
class PipelinePool:
    def __init__(self, size, factory):
        self.size = size
        self._queue: asyncio.Queue[OCRPipeline] = asyncio.Queue(maxsize=size)

    async def init(self):
        # lifespan 启动时调一次, 预热所有 pipeline
        for i in range(self.size):
            logger.info("building pipeline %d/%d", i + 1, self.size)
            pipe = self._factory()
            await self._queue.put(pipe)

    @asynccontextmanager
    async def lease(self):
        # 1) 从 queue 拿一个 pipeline (阻塞等)
        # 2) yield 给调用方
        # 3) finally 归还
        pipe = await self._queue.get()  # ← 这里会等, 直到有空闲
        try:
            yield pipe
        finally:
            await self._queue.put(pipe)
```

**为什么用 `asyncio.Queue(maxsize=size)` 而不是 `asyncio.Semaphore`**：
- 池子里的 `OCRPipeline` 是**有状态对象**（加载了模型、占着显存），不能"复制"，必须**独占**
- Semaphore 只能控制并发数，不能保证同一个 pipeline 不被两个协程同时拿到
- Queue 是真正的"对象池"：放进去 2 个 pipeline 实体，谁拿到谁跑，跑完还回去

**`lease` 日志**：`lease: waited 0.000s, free=1/2` / `release: free=2/2`——这个是排查"卡在哪一步"的关键线索。

---

### 2.9 `BatchScheduler` (707-1009 行) ⭐⭐⭐ 整套系统的大脑

#### 2.9.1 状态字段 (734-764 行)

```python
self.pool = pool
self.max_batch = 5                  # 一次最多拼几张
self.flush_seconds = 0.25            # 凑批等待 250ms
self.n_workers = 2                   # 跟 pool 大小一样, 一一对应

# per-user 独立队列
self._user_pending: dict[str, deque[Job]] = {}    # 每个 user 一个 FIFO
self._user_order: list[str] = []                    # user 首次出现顺序 (断 ties)
self._user_completed: dict[str, int] = {}          # 每个 user 累计完成数 (用于 fair 排序)
self._pending_count = 0                            # 总 pending 数 (O(1) 读)

self._lock = asyncio.Lock()       # 保护上面 3 个字段
self._wakeup = asyncio.Event()    # 有新 job / release 时唤醒 worker
self._closed = False
self._worker_tasks: list[Task] = []

# 心跳 / 释放
self.heartbeat_timeout = 30.0       # user 30s 没心跳 → 自动 release
self.cleanup_interval = 5.0         # 5s 检查一次
self._user_heartbeat: dict[str, float] = {}  # user_id → 上次心跳 monotonic 时间
self._released: set[str] = set()     # 已主动 release 但 in-flight 还没跑完的
self._cleanup_task: Optional[Task] = None
```

#### 2.9.2 `start` / `close` (766-797 行)

```python
async def start(self):
    for i in range(self.n_workers):
        self._worker_tasks.append(asyncio.create_task(self._worker_loop(i)))
    self._cleanup_task = asyncio.create_task(self._cleanup_loop())

async def close(self):
    self._closed = True
    self._wakeup.set()               # 唤醒所有 worker 让它们看到 _closed
    self._cleanup_task.cancel()
    for t in self._worker_tasks: t.cancel()
    await self._cancel_all_pending("scheduler closed")
```

#### 2.9.3 `submit` (799-814 行) —— 同步入口

```python
def submit(self, request_id, page_index, image) -> Job:
    fut = loop.create_future()
    job = Job(request_id, page_index, image, fut)
    # 首次出现的 user 加进 _user_order
    if request_id not in self._user_pending:
        self._user_pending[request_id] = collections.deque()
        self._user_order.append(request_id)
    self._user_pending[request_id].append(job)
    self._pending_count += 1
    # 第一次见到这个 user 就记一次心跳
    self._user_heartbeat[request_id] = time.monotonic()
    self._wakeup.set()                # 唤醒 worker
    return job                        # ⚠️ 同步返回, 不 await
```

**关键点**：`submit` 是**同步**的（不 `async def`），但内部碰的是 asyncio 同步状态 (`deque`, `dict`)，**没有 await**。这个很重要——它意味着 `submit` 必须从 asyncio 协程里调（不是任意线程），且整个调用序列 (`append`/`_wakeup.set`) 是原子的，因为当前协程是唯一写者。

#### 2.9.4 `heartbeat` / `release_user` (819-855 行)

```python
def heartbeat(self, request_id):
    self._user_heartbeat[request_id] = time.monotonic()
    self._released.discard(request_id)  # 如果被释放过又被认领, 清掉标记

def release_user(self, request_id, reason="tab closed") -> int:
    return self._cancel_user_pending(request_id, reason)

def _cancel_user_pending(self, user, reason) -> int:
    q = self._user_pending.get(user)
    n = 0
    if q:
        while q:
            job = q.pop()
            if not job.fut.done():
                job.fut.cancel(reason)   # ⚠️ 这里让 await job.fut 的协程抛 CancelledError
            n += 1
        del self._user_pending[user]
        self._pending_count -= n
    # 无论是否有 pending, 都从 heartbeat 表移除 + 记 released
    # 否则 health 还会以为 user 活跃
    self._user_heartbeat.pop(user, None)
    self._released.add(user)
    self._wakeup.set()                    # 唤醒 worker
    return n
```

**坑**：
- **老 bug**：`_cancel_user_pending` 早期实现是 `if not q: early return + 只加 _released`，导致 `heartbeat` 表不 pop，active_users 永远不归零。修法：把 `pop` 移到方法前部，任何路径都先 pop heartbeat（已修）。
- **释放 in-flight 任务**：`_cancel_user_pending` 只取消**还在 `_user_pending` 里的 job**。如果 worker 已经 `_take_fair_batch` 抢走但还没跑完，那种 job 没法取消——等它跑完 `set_result`，但调用方已经在 `gather` 里 `CancelledError` 抛出，所以 result 没人接（Python GC 自然处理）。

#### 2.9.5 `_cleanup_loop` (862-882 行)

```python
async def _cleanup_loop(self):
    while not self._closed:
        await asyncio.sleep(self.cleanup_interval)
        now = time.monotonic()
        expired = []
        # ⚠️ 必须 list() 快照, dict 边遍历边改会 RuntimeError
        for u, last in list(self._user_heartbeat.items()):
            if u in self._released: continue
            if now - last > self.heartbeat_timeout:
                expired.append(u)
        for u in expired:
            self._cancel_user_pending(u, reason=f"heartbeat timeout ({self.heartbeat_timeout}s)")
```

**坑**：必须 `list(self._user_heartbeat.items())` 快照，因为 `_cancel_user_pending` 内部会 `pop` 这个 dict，遍历中改 size 会 `RuntimeError: dictionary changed size during iteration`。

#### 2.9.6 `_take_fair_batch` (884-937 行) ⭐ max-min fair 调度

```python
def _take_fair_batch(self, max_n) -> list[Job]:
    # 调用者必须持 self._lock
    if not self._user_pending or max_n <= 0:
        return []

    order_idx = {u: i for i, u in enumerate(self._user_order)}
    def _sort_key(u):
        return (self._user_completed.get(u, 0), order_idx.get(u, 0))

    # 第一轮: 每个 user 各拿 1 张
    users = sorted(
        [u for u in self._user_order if self._user_pending.get(u)],
        key=_sort_key,
    )
    for u in users:
        q = self._user_pending[u]
        if not q: continue
        result.append(q.popleft())
        if len(result) >= max_n: break

    # 第二轮: 填满到 max_n
    while len(result) < max_n:
        users = sorted(
            [u for u in self._user_order if self._user_pending.get(u)],
            key=_sort_key,
        )
        if not users: break
        picked = False
        for u in users:
            q = self._user_pending[u]
            if not q: continue
            result.append(q.popleft())
            picked = True
            if len(result) >= max_n: break
        if not picked: break

    # 清理空 user 队列 (但保留 _user_order 顺序历史)
    for u in list(self._user_pending.keys()):
        if not self._user_pending[u]:
            del self._user_pending[u]

    self._pending_count -= len(result)
    return result
```

**max-min fair 解释**：
- 排序 key = `(已完成数, 首次出现顺序)`
- 第一轮保证每个 user 至少拿到 1 张 → 公平起点
- 第二轮按 key 继续拿 → 防止"已完成多"的 user 继续抢

**举例**：A 一次 submit 90 页 PDF，B 5s 后 submit 1 张图
- 第 1 批：1 张 A + 1 张 B = 2 张（各 1 张）
- 第 2 批：A 已完成 1，B 已完成 1 → key 都是 (1, 0)，按 user_order 顺序拿，A 优先 → 拿 1 张 A = 1 张
- 第 3 批：现在 A (1,0) vs B (1,1) → A 优先 → 拿 1 张 A
- ... 这样 A 平均每 2 批拿 1 张，**B 每 2 批拿 1 张**，新 B 用户不被老 A 独吞

#### 2.9.7 `_worker_loop` (939-1009 行) ⭐ 调度主循环

```python
async def _worker_loop(self, wid):
    while not self._closed:
        # ── 1) 等到至少 1 个 job
        while self._pending_count == 0 and not self._closed:
            self._wakeup.clear()
            await self._wakeup.wait()
        if self._closed: break
        if self._pending_count == 0: continue

        # ── 2) 抢 1 个 job 当头 (fair 选)
        async with self._lock:
            if self._pending_count == 0: continue
            batch = self._take_fair_batch(1)
            if not batch: continue
            if self._pending_count == 0:
                self._wakeup.clear()      # 没 job 了, 清掉 wakeup, 回 1) 等

        # ── 3) 凑 batch (max_batch 或 flush_seconds 超时; 每轮都 fair 选)
        deadline = time.perf_counter() + self.flush_seconds
        while len(batch) < self.max_batch and self._pending_count > 0:
            remaining = deadline - time.perf_counter()
            if remaining <= 0: break
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            async with self._lock:
                more = self._take_fair_batch(self.max_batch - len(batch))
                batch.extend(more)
                if self._pending_count == 0:
                    self._wakeup.clear()
            if len(batch) >= self.max_batch: break

        # ── 4) 借 pipeline + 跑批 + 分发
        t0 = time.perf_counter()
        async with self.pool.lease() as pipe:
            wait = time.perf_counter() - t0
            users_in_batch = sorted({j.request_id for j in batch})
            user_in_batch_count = {}
            for j in batch:
                user_in_batch_count[j.request_id] = user_in_batch_count.get(j.request_id, 0) + 1
            logger.info(
                "scheduler[%d] lease: waited %.3fs, free=%d/%d, batch=%d (%d user %s), per_user=%s",
                wid, wait, self.pool.qsize(), self.pool.size,
                len(batch), len(users_in_batch), users_in_batch,
                user_in_batch_count,
            )
            from starlette.concurrency import run_in_threadpool
            results = await run_in_threadpool(pipe.process_batch, batch)
        dt = time.perf_counter() - t0

        # ── 5) 更新每个 user 的已完成数
        for u, c in user_in_batch_count.items():
            self._user_completed[u] = self._user_completed.get(u, 0) + c

        # ── 6) 分发结果
        for job, page in zip(batch, results):
            if not job.fut.done():
                job.fut.set_result(page)
        logger.info("scheduler[%d] done: batch=%d in %.2fs, ...")
```

**为什么 worker 是 2 个 (= pool size)**：
- 每个 worker 持有一个 pipeline 跑批
- 2 个 worker 可以同时跑 2 个 batch（但要等 `lease` 借到 pipeline）
- 实际是 "N 个 worker 抢 N 个 pipeline"，worker 数 = pool size 是最优

**`_wakeup.wait()` + `wait_for(timeout=...)` 模式**：
- `_wakeup` 是 `asyncio.Event`
- worker 阻塞在 `await self._wakeup.wait()` 上
- `submit` / `_cancel_user_pending` 会 `_wakeup.set()` 唤醒
- 凑批循环里用 `wait_for(timeout=remaining)` 实现"最多等 250ms"——既能被新 job 唤醒，也能 flush 到时

**`run_in_threadpool(pipe.process_batch, batch)`**：
- `process_batch` 是**同步函数**（不 `async def`），因为里面有 GPU forward（`torch.no_grad` + `model.generate`）
- `run_in_threadpool` 把同步函数扔到 anyio 线程池，不阻塞 event loop
- **重要**：N 个 worker 跑在 N 个线程上，**不同 worker 的 `process_batch` 真的并行**（因为是线程）

#### 2.9.8 工厂 + 默认路径 (1012-1030 行)

```python
LAYOUT_MODEL_PATH = "./model/PP-DocLayoutV3"
VL_MODEL_PATH = "./model/PaddleOCR-VL-1.6"
DEFAULT_OUTPUT_DIR = Path("./text_result")

def make_default_pipeline(output_dir=DEFAULT_OUTPUT_DIR, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return OCRPipeline(
        layout_model_path=LAYOUT_MODEL_PATH,
        vl_model_path=VL_MODEL_PATH,
        device=device,
        output_dir=output_dir,
    )
```

**注意**：`demo.py` 里**没有用** `make_default_pipeline`，而是直接 import `OCRPipeline` 自己写 `_make_pipeline()` 工厂 (demo.py:179-188)。差别只是 `_make_pipeline` 强转了 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`，可合并。

---

## 3. `demo.py` 详解

### 3.1 配置 / 环境变量 (61-84 行)

```python
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
POOL_SIZE = max(1, int(os.environ.get("PIPELINE_POOL_SIZE", "2")))
BATCH_MAX = max(1, int(os.environ.get("BATCH_MAX", "5")))
BATCH_FLUSH_MS = float(os.environ.get("BATCH_FLUSH_MS", "250"))
LOG_DIR = Path(os.environ.get("LOG_DIR", "./logs"))
LOG_KEEP_DAYS = int(os.environ.get("LOG_KEEP_DAYS", "14"))
PDF_DPI = float(os.environ.get("PDF_DPI", "200"))

# 资源保护
MAX_IMAGE_UPLOAD_MB = float(os.environ.get("MAX_IMAGE_UPLOAD_MB", "32"))
MAX_PDF_UPLOAD_MB = float(os.environ.get("MAX_PDF_UPLOAD_MB", "128"))
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "200"))

# 心跳 / 清理
HEARTBEAT_TIMEOUT_S = float(os.environ.get("HEARTBEAT_TIMEOUT_S", "30"))
CLEANUP_INTERVAL_S = float(os.environ.get("CLEANUP_INTERVAL_S", "5"))

# CORS
CORS_ALLOW_ORIGINS = os.environ.get("CORS_ALLOW_ORIGINS", "http://localhost:*,http://127.0.0.1:*").split(",")

STATIC_DIR = Path(__file__).parent / "static"
```

---

### 3.2 日志 (87-171 行)

#### 3.2.1 `_DailyFileHandler` (88-122 行)

```python
class _DailyFileHandler(logging.FileHandler):
    def __init__(self, log_dir, prefix="server", encoding="utf-8"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._current_date = None
        super().__init__(self._today_path(), mode="a", encoding=encoding, delay=False)
        self._current_date = time.strftime("%Y-%m-%d")

    def _today_path(self):
        return str(self._log_dir / f"{self._prefix}-{time.strftime('%Y-%m-%d')}.log")

    def emit(self, record):
        today = time.strftime("%Y-%m-%d")
        if today != self._current_date:
            # 跨天, 关旧文件, 打开新文件
            self._current_date = today
            self.close()                        # 关 stream
            self.baseFilename = self._today_path()  # 换路径
            try:
                self.stream = self._open()
            except Exception:
                return                          # 目录被删, 这一条丢了不 crash
        super().emit(record)
```

**为什么不用 `TimedRotatingFileHandler`**：那个是 suffix 改名模式，会先有 `server.log` 然后 rotate 成 `server.log.2026-07-24`，管理麻烦。`_DailyFileHandler` 是**独立文件**模式：`server-2026-07-24.log`、`server-2026-07-25.log`，每个文件就是一天，没有 `server.log` 父文件。

**roll 触发**：`emit()` 每次检查今天日期，跨天那次自动切——不需要后台线程。

#### 3.2.2 `_cleanup_old_logs` (125-146 行)

启动时扫 `logs/server-*.log`，文件日期 > `LOG_KEEP_DAYS` 天前的删掉。

#### 3.2.3 `_setup_logging` (149-171 行)

```python
def _setup_logging():
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    formatter = logging.Formatter(fmt)
    sh = logging.StreamHandler(); sh.setFormatter(formatter); sh.setLevel(INFO)
    fh = _DailyFileHandler(LOG_DIR, prefix="server", encoding="utf-8")
    fh.setFormatter(formatter); fh.setLevel(INFO)
    root = logging.getLogger()
    root.handlers.clear()          # 避免 uvicorn 默认 handler 重复
    root.addHandler(sh); root.addHandler(fh)
    root.setLevel(INFO)

_setup_logging()                  # 模块级副作用, import 时就跑
logger = logging.getLogger("wise-paddle")
```

**注意**：`root.handlers.clear()` 会把 uvicorn 自带的 handler 干掉——uvicorn 的 access log (`INFO: 127.0.0.1:xxx - "POST /ocr/upload" 200 OK`) 走的是 uvicorn 自定义 logger (`uvicorn.access`)，不走 `root`，所以清空 root 不影响 access log（access log 也不进我们的文件 handler，只去控制台/stdout）。

---

### 3.3 `lifespan` (191-244 行) ⭐ 启动钩子

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 启动阶段
    global scheduler, pool_ref
    pool = PipelinePool(size=POOL_SIZE, factory=_make_pipeline)
    await pool.init()                              # 预热, 加载 2 份模型
    pool_ref = pool

    sched = BatchScheduler(
        pool=pool, max_batch=BATCH_MAX, flush_ms=BATCH_FLUSH_MS,
        n_workers=POOL_SIZE,
        heartbeat_timeout=HEARTBEAT_TIMEOUT_S,
        cleanup_interval=CLEANUP_INTERVAL_S,
    )
    await sched.start()                            # 启动 worker + cleanup
    scheduler = sched

    _pdf_cleanup_task = asyncio.create_task(_pdf_progress_cleanup_loop())

    _cleanup_old_logs(LOG_KEEP_DAYS)               # 启动时清旧 log

    yield                                          # ── 运行阶段

    # ── 关闭阶段
    # 1) 取消 PDF progress cleanup
    # 2) 取消所有 PDF 后台 task
    # 3) scheduler.close() 取消 worker + 取消所有 pending
    # 4) pool.close() (目前是空操作)
    # 5) 清空 _pdf_progress_store
```

**`global scheduler, pool_ref`**：FastAPI 路由 handler 用 `scheduler.xxx()` 调用，因为 handler 不能直接 import 这些实例（实例是 lifespan 内创建的），所以用模块级全局变量。

---

### 3.4 数据模型 (271-336 行)

```python
class Base64Request(BaseModel):
    payload: str                                # base64 字符串, 可带 data: URI 前缀

class OCRRegion(BaseModel):
    page_index, box_index, label, score, rect, md_path, markdown

class OCRPage(BaseModel):
    page_index, width, height, image_b64        # 原图 PNG base64

class OCRResponse(BaseModel):
    success, request_id, status, elapsed_seconds, queue_wait_seconds
    scheduler_pending, pool_size, pool_free
    pages: list[OCRPage]
    regions: list[OCRRegion]                    # 摊平, 按 page_index 关联

class HealthResponse(BaseModel):
    status, version, pool_size, pool_free
    scheduler_pending, batch_max, active_users

class HeartbeatRequest(BaseModel): user_id: str
class HeartbeatResponse(BaseModel): ok, pending, server_time
class ReleaseRequest(BaseModel):   user_id: str
class ReleaseResponse(BaseModel):  ok, cancelled
```

**`image_b64` 的设计**：服务端把原图再 PNG 编码一次 base64 返回。**好处**：前端拿到直接 `<img src="data:image/png;base64,..."/>` 画到 canvas，不用再传一次原图。**坏处**：单图 100KB 的话响应体多 130KB。生产可能要改成给前端一个 `/api/image/{request_id}/{page_idx}` URL。

---

### 3.5 工具函数 (339-413 行)

#### `_check_content_length` (340-352 行)
- 提前拒绝超大上传, 避免 `await file.read()` 把整个文件读进内存

#### `_strip_data_uri` / `_decode_b64_to_rgb` (355-371 行)
- 剥 `data:image/png;base64,` 前缀
- `cv2.imdecode` 直接吃 `np.frombuffer(raw, np.uint8)` → BGR
- `cv2.cvtColor(..., BGR2RGB)` → 给 PIL

#### `_pil_to_b64_png` (380-387 行)
- `Image.save(BytesIO, format="PNG", optimize=False)` → `base64.b64encode`

#### `_decode_pdf_to_pil_pages` (390-413 行)
```python
def _decode_pdf_to_pil_pages(raw, dpi=200.0):
    if not raw[:4] == b"%PDF": raise ValueError("不是合法 PDF 文件")
    zoom = dpi / 72.0                # PDF 坐标系 72 dpi
    mat = pymupdf.Matrix(zoom, zoom)
    pdf = pymupdf.open(stream=raw, filetype="pdf")  # ⚠️ in-memory, 不落盘
    try:
        for page_num in range(len(pdf)):
            pix = pdf[page_num].get_pixmap(matrix=mat, alpha=False)
            # pix.samples 是 RGB 字节 (alpha=False 时)
            img = PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(img)
    finally:
        pdf.close()
    return pages
```

**`PILImage.frombytes` 而不是 `numpy.array` 再 `Image.fromarray`**：少一次内存拷贝。100 页 PDF 省 50MB 中间分配。

---

### 3.6 `/` `/health` (417-433 行)

```python
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok", version="0.4.0",
        pool_size=POOL_SIZE,
        pool_free=pool_ref.qsize() if pool_ref else 0,
        scheduler_pending=scheduler.pending_size() if scheduler else 0,
        batch_max=BATCH_MAX,
        active_users=len(scheduler._user_heartbeat) if scheduler else 0,
    )
```

**`active_users = len(scheduler._user_heartbeat)`**：直接读 scheduler 内部 dict。**不算干净**——`_user_heartbeat` 是下划线开头，理想是 scheduler 暴露一个 `active_user_count()` 方法。

---

### 3.7 `_process_pages` (437-474 行) ⭐ 单图/单 page 提交核心

```python
async def _process_pages(request_id, pil_pages) -> tuple[list[PageResult], float]:
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler 尚未初始化")

    # 1) 提交时主动 heartbeat 一次, 避免 cleanup 误判
    scheduler.heartbeat(request_id)

    submit_t = time.perf_counter()
    jobs = []
    for idx, pil_img in enumerate(pil_pages):
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        job = scheduler.submit(request_id=request_id, page_index=idx, image=pil_img)
        jobs.append(job)

    # 2) 等所有 page 全部完成
    try:
        pages = await asyncio.gather(*(j.fut for j in jobs))
        queue_wait = time.perf_counter() - submit_t
        return list(pages), queue_wait
    except asyncio.CancelledError:
        # 关 tab / cleanup 释放 / server shutdown
        for j in jobs:
            if not j.fut.done():
                j.fut.cancel("cancelled by client/server")
        queue_wait = time.perf_counter() - submit_t
        logger.info("request cancelled: user=%s pages=%d", request_id, len(jobs))
        raise   # 仍然 raise, 让上层 handler 抓
```

**为什么先 `heartbeat` 再 submit**：第一次见到这个 user 时 `_user_pending[user]` 初始化 + `_user_heartbeat[user] = now`，但 `_user_heartbeat[user]` 是 `submit` 内部做的（833 行），所以理论上不需要。但**保险起见** `heartbeat()` 再加一次，避免 cleanup 协程在那 5s 里把新 user 当作"超时未心跳"清掉（虽然 user 第一次 submit 就在 `_user_heartbeat` 里登记了）。

**`asyncio.gather(*futures)`**：
- 一个 `CancelledError` 抛上来 → 所有未完成的 future 也会被 cancel
- handler 层 try/except 抓 `CancelledError` → 返回 200 + `status="cancelled"` JSON（**不**让 FastAPI 把它当 500）

---

### 3.8 `_build_ocr_response_with_images` (477-520 行)

```python
def _build_ocr_response_with_images(request_id, pil_pages, page_results, elapsed, queue_wait):
    all_regions = []
    out_pages = []
    for pil, pr in zip(pil_pages, page_results):
        b64 = _pil_to_b64_png(pil)
        out_pages.append(OCRPage(page_index=pr.page_index, width=pr.width,
                                 height=pr.height, image_b64=b64))
        for b_idx, r in enumerate(pr.regions):
            all_regions.append(OCRRegion(page_index=pr.page_index, ...))
    return OCRResponse(success=True, request_id=request_id, status="done",
                       elapsed_seconds=round(elapsed, 3), ...)
```

**响应结构**：
- `pages[]` — 每页原图 base64（前端 canvas 画）
- `regions[]` — 摊平的所有 region，按 `page_index` 关联回 `pages[i]`

---

### 3.9 单图路由 `/ocr/upload` `/ocr/base64` (523-599 行)

```python
@app.post("/ocr/upload", response_model=OCRResponse)
async def ocr_upload(request, file):
    _check_content_length(request, MAX_IMAGE_UPLOAD_MB)
    raw = await file.read()
    if len(raw) > MAX_IMAGE_UPLOAD_MB * 1024*1024:
        raise HTTPException(413, ...)
    # cv2 解码
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400, "无法解码")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = _rgb_to_pil(rgb)

    request_id = request.query_params.get("user_id") or uuid.uuid4().hex[:8]
    st = time.perf_counter()
    try:
        page_results, queue_wait = await _process_pages(request_id, [pil])
    except asyncio.CancelledError:
        return JSONResponse(200, content={...status="cancelled", success=False, ...})
    elapsed = time.perf_counter() - st
    return _build_ocr_response_with_images(request_id, [pil], page_results, elapsed, queue_wait)
```

**`request_id = request.query_params.get("user_id")`**：前端用 `?user_id=web-xxx` 传同一个 id，心跳和后续 cancel 才能对得上。如果前端没传（curl 测试），降级到 `uuid.uuid4().hex[:8]`。

**`try/except CancelledError` + `JSONResponse(200)`**：把"被取消"作为**业务状态**返回（不是异常）。前端拿到 `status="cancelled"` 知道是被主动 cancel 的，区别于 500 错误。

---

### 3.10 PDF 路由 `/ocr/pdf` + 进度轮询 (602-828 行) ⭐ 异步流式

#### 3.10.1 全局状态 (617-618 行)

```python
_pdf_progress_store: dict[str, dict] = {}    # user_id → progress dict
_pdf_progress_lock = asyncio.Lock()
```

**`PDFProgress` 字段**：

```python
{
   'total_pages': int,
   'pil_pages': list[PIL.Image],  # 保留原图给进度响应里 image_b64 用
   'pages': dict[int, dict],  # page_index → 已完成 page 的 dict
   'done': bool,
   'done_count': int,
   'started_at': float,
   'finished_at': float | None,
   'task': asyncio.T | None,
   'error': str | None,
   'cancelled': bool,
}
```

#### 3.10.2 `_make_pdf_page_dict` (621-640 行)

`PageResult` + PIL 原图 → 单 page 响应 dict（结构跟 `OCRResponse.pages[]` 单元素一致）。

#### 3.10.3 `_run_pdf_batch` (643-693 行) ⭐ PDF 后台 task

```python
async def _run_pdf_batch(user_id, pil_pages):
    progress = _pdf_progress_store[user_id]
    jobs = []
    try:
        scheduler.heartbeat(user_id)             # 防 cleanup 误判
        for idx, pil_img in enumerate(pil_pages):
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            job = scheduler.submit(request_id=user_id, page_index=idx, image=pil_img)
            jobs.append(job)

        # wait(FIRST_COMPLETED) 按完成顺序处理
        pending = {j.fut for j in jobs}
        fut_to_idx = {j.fut: j.page_index for j in jobs}
        while pending:
            done_set, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for fut in done_set:
                try:
                    page_result = await fut
                    page_idx = fut_to_idx.get(fut)
                    if page_idx is None: continue
                    page_dict = _make_pdf_page_dict(page_result, pil_pages[page_idx])
                    progress['pages'][page_idx] = page_dict         # ← 每完成一页立即写入
                    progress['done_count'] = len(progress['pages'])
                except asyncio.CancelledError:
                    progress['cancelled'] = True
                    for j in jobs:
                        if not j.fut.done():
                            j.fut.cancel("cancelled")
                    return
                except Exception as e:
                    progress.setdefault('errors', []).append({"err": str(e)})
    except Exception as e:
        progress['error'] = str(e)
    finally:
        progress['done'] = True
        progress['finished_at'] = time.time()
        logger.info("PDF batch finished: user=%s done_count=%d/%d wall=%.2fs", ...)
```

**关键点**：
- `asyncio.wait(pending, FIRST_COMPLETED)` — 每完成一页就处理一页（不是 gather 一次性等完）
- 每完成一页立即 `progress['pages'][page_idx] = page_dict` — **前端轮询能立刻看到这一页**
- `fut_to_idx` 字典是必需的——`wait` 返回的 `done_set` 里是 Future 本身，没有 page_index

#### 3.10.4 `/ocr/pdf` 路由 (713-797 行)

```python
@app.post("/ocr/pdf")
async def ocr_pdf(request, file):
    # 1) 校验 + 解析 (跟 /ocr/upload 类似)
    # 2) PDF 渲染放线程池, pymupdf 偶尔会卡
    pil_pages = await run_in_threadpool(_decode_pdf_to_pil_pages, raw, PDF_DPI)
    # 3) 页数检查
    if len(pil_pages) > MAX_PDF_PAGES: raise HTTPException(413, ...)

    user_id = request.query_params.get("user_id") or uuid.uuid4().hex[:8]

    # 4) 初始化 progress, 启动后台 task
    async with _pdf_progress_lock:
        _pdf_progress_store[user_id] = {
            'total_pages': len(pil_pages),
            'pil_pages': pil_pages,
            'pages': {}, 'done': False, 'done_count': 0,
            'started_at': time.time(), 'finished_at': None,
            'error': None, 'cancelled': False,
        }
        task = asyncio.create_task(_run_pdf_batch(user_id, pil_pages))
        _pdf_progress_store[user_id]['task'] = task

    # 5) 立即返回 202
    return JSONResponse(202, content={
        "success": True, "request_id": user_id,
        "status": "processing", "total_pages": len(pil_pages),
        "poll_url": f"/api/ocr/pdf-status/{user_id}",
        "elapsed_seconds": 0,
        ...
    })
```

**为什么返回 202**：202 Accepted = "请求已收到，正在处理"。前端拿到 `poll_url` 立刻开始轮询。

**`run_in_threadpool(_decode_pdf_to_pil_pages, raw, PDF_DPI)`**：pymupdf 偶尔会卡几秒，放线程池不阻塞 event loop。

#### 3.10.5 `/api/ocr/pdf-status/{user_id}` (800-828 行)

```python
@app.get("/api/ocr/pdf-status/{user_id}")
async def pdf_status(user_id):
    progress = _pdf_progress_store.get(user_id)
    if not progress:
        return JSONResponse(200, content={
            "user_id": user_id, "total_pages": 0, "done": True,
            "done_count": 0, "error": "progress expired or unknown user_id",
            "pages": {},
        })
    return JSONResponse(200, content={
        "user_id": user_id, "total_pages": progress['total_pages'],
        "done": progress['done'], "done_count": progress['done_count'],
        "cancelled": progress.get('cancelled', False),
        "error": progress.get('error'),
        "pages": dict(progress['pages']),
    })
```

**前端轮询策略**（在 `static/index.html`）：1.5s 一次，每次拿到新 pages 调 `fillOnePageSlot` 增量填卡片。

#### 3.10.6 `_pdf_progress_cleanup_loop` (696-710 行)

每 60s 扫一次 `done=True && finished_at > 10min` 的 progress，删掉避免内存泄漏。

---

### 3.11 心跳/释放路由 (831-863 行)

```python
@app.post("/api/heartbeat")
async def api_heartbeat(req):
    if scheduler is None: raise HTTPException(503, ...)
    scheduler.heartbeat(req.user_id)
    return HeartbeatResponse(ok=True, pending=scheduler.pending_size(), server_time=time.time())

@app.post("/api/release")
async def api_release(req):
    if scheduler is None: raise HTTPException(503, ...)
    cancelled = scheduler.release_user(req.user_id, reason="client released")
    # 取消该 user 的 PDF 后台 task (如果存在)
    progress = _pdf_progress_store.get(req.user_id)
    if progress:
        task = progress.get('task')
        if task is not None and not task.done():
            task.cancel()
        progress['cancelled'] = True
    return ReleaseResponse(ok=True, cancelled=cancelled)
```

**`/api/release` 同时取消 scheduler pending + PDF 后台 task**——一个 user 离开要彻底清掉他所有 GPU 工作。

---

### 3.12 启动入口 (866-878 行)

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "demo:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
        log_level="info",
        timeout_graceful_shutdown=int(os.environ.get("GRACEFUL_SHUTDOWN_S", "15")),
    )
```

**`timeout_graceful_shutdown=15`**：Ctrl+C 后 15s 等待正在跑的请求完成（`scheduler.close()` + 取消 pending task）。

**`reload=False`**：开发可以开 reload，但会重启进程丢失 BatchScheduler 状态。

---

## 4. 一次完整请求的执行流（自上而下）

### 4.1 单图上传 `/ocr/upload`

```
[前端]
  POST /ocr/upload?user_id=web-abc123
  body: multipart/form-data { file: <binary> }
         |
         ▼
[demo.py: ocr_upload 525]
  1. _check_content_length(413 拒超大)
  2. raw = await file.read()
  3. cv2.imdecode → BGR → cv2.cvtColor → RGB
  4. _rgb_to_pil → PIL
  5. request_id = "web-abc123" (从 query string 拿)
  6. await _process_pages(request_id, [pil])  ← 525
         |
         ▼
[demo.py: _process_pages 437]
  1. scheduler.heartbeat(request_id)  ← 防 cleanup
  2. submit_t = now
  3. scheduler.submit(request_id, 0, pil)  ← 同步, 立即返回 Job
       └→ [core_pipeline.py: BatchScheduler.submit 799]
            1. fut = loop.create_future()
            2. job = Job(...)
            3. _user_pending[user].append(job)
            4. _pending_count += 1
            5. _user_heartbeat[user] = now
            6. _wakeup.set()           ← 唤醒 worker
  4. asyncio.gather(j.fut)            ← 阻塞等
         |
         ▼
[core_pipeline.py: BatchScheduler._worker_loop 939]
  worker 在 self._wakeup.wait() 上醒来
  1. _take_fair_batch(1)             ← 抢这 1 个 job
  2. deadline = now + 0.25s           ← 凑批等待 250ms
  3. 循环 _take_fair_batch 到 max_batch=5 或 timeout
  4. async with self.pool.lease() as pipe:   ← 借 pipeline
       └→ [core_pipeline.py: PipelinePool.lease 687]
            1. pipe = await self._queue.get()   ← 阻塞等空闲
            2. yield pipe
  5. results = await run_in_threadpool(pipe.process_batch, batch)
       └→ [core_pipeline.py: OCRPipeline.process_batch 569]
            1. layout.detect(images)        ← 1 次 batch forward
            2. filter.filter + crop          ← per page
            3. buckets = group_by_label       ← 跨 user 桶
            4. for bucket: vl.recognize_batch() ← 每桶 1 次 forward
            5. write md files + return PageResult list
  6. for job, page in zip: job.fut.set_result(page)  ← 解锁 await
  7. _user_completed[user] += n
         |
         ▼
[demo.py: _process_pages 463]
  pages = await asyncio.gather(...)  ← 收到结果, 返回
         |
         ▼
[demo.py: ocr_upload 562]
  return _build_ocr_response_with_images(...)  ← OCRResponse JSON
         |
         ▼
[前端]
  解析 JSON, build placeholder card, fill regions, draw boxes on canvas
```

### 4.2 PDF 上传 `/ocr/pdf`

```
[前端]
  POST /ocr/pdf?user_id=web-xyz789
  body: multipart/form-data { file: <PDF binary> }
         |
         ▼
[demo.py: ocr_pdf 713]
  1. _check_content_length(MAX_PDF_UPLOAD_MB=128)
  2. raw = await file.read()
  3. pil_pages = await run_in_threadpool(_decode_pdf_to_pil_pages, raw, 200dpi)  ← 740
  4. len(pil_pages) > 200? raise 413
  5. user_id = "web-xyz789"
  6. async with _pdf_progress_lock:
       _pdf_progress_store[user_id] = { total_pages, pil_pages, pages={}, ... }
       task = asyncio.create_task(_run_pdf_batch(user_id, pil_pages))  ← 778
  7. return JSONResponse(202, { request_id, status="processing", poll_url, total_pages, ... })
         |
         ▼
[前端]
  收到 202, 立刻 build placeholder card with page chips
  setInterval(() => fetch('/api/ocr/pdf-status/web-xyz789'), 1500)  ← 轮询
         |
         ▼
[demo.py: _run_pdf_batch 643] (后台 task)
  for idx, pil in enumerate(pil_pages):
      job = scheduler.submit(user_id, idx, pil)     ← 同步入队
  pending = {j.fut for j in jobs}
  while pending:
      done_set, pending = await asyncio.wait(pending, FIRST_COMPLETED)
      for fut in done_set:
          page_result = await fut                  ← 等这一页完成
          page_dict = _make_pdf_page_dict(page_result, pil_pages[page_idx])
          progress['pages'][page_idx] = page_dict  ← 立即写, 前端轮询能立刻看到
          progress['done_count'] += 1
  progress['done'] = True
  progress['finished_at'] = time.time()
         |
         ▼
[前端轮询]
  1.5s 一次, 拿 progress['pages'] 增量 fill 进 UI:
  - 新页: fillOnePageSlot 画图 + 填 region 卡片 + wire canvas
  - 全部 done: clear interval, markAllDone
```

### 4.3 心跳 / 释放流程

```
[前端]
  setInterval(() => fetch('/api/heartbeat', {user_id}), 10000)  ← 每 10s 一次
  
  pagehide / beforeunload 触发:
  navigator.sendBeacon('/api/release', {user_id})               ← 关 tab
         |
         ▼
[demo.py: api_heartbeat 832]
  scheduler.heartbeat(user_id)  ← 续约 _user_heartbeat[user] = monotonic()
  return { ok, pending, server_time }
         |
         ▼
[core_pipeline.py: BatchScheduler._cleanup_loop 862] (后台, 每 5s)
  for u, last in list(_user_heartbeat.items()):
      if now - last > 30: _cancel_user_pending(u, "heartbeat timeout")
         |
         ▼
[demo.py: api_release 849]
  scheduler.release_user(user_id)      ← 取消 scheduler pending jobs
  progress = _pdf_progress_store.get(user_id)
  if progress:
      progress['task'].cancel()        ← 取消 PDF 后台 task
      progress['cancelled'] = True
  return { ok, cancelled: N }
```

---

## 5. 关键设计决策（为什么这么做）

### 5.1 单进程 + PipelinePool(2) vs 多 uvicorn worker

| 概念 | 数量 | 显存 | 作用 |
|---|---|---|---|
| uvicorn process | 1 | Python 解释器 1 份 | HTTP 服务 |
| PipelinePool.size | 2 | 模型权重 2 份 ≈3.8GB | 同一进程内 2 路并发 |
| BatchScheduler.worker | 2 | 协程 | 攒批 + 借 pipeline + 跑批 |

**为什么不要 `uvicorn --workers 2`**：每个 worker 是独立 Python 解释器，会把 layout+VL 各加载 1 份，2 个 worker = 4 份模型 ≈7.6GB，8GB 显卡 OOM。

**为什么需要 2 份模型**：GPU 推理是串行的，1 个 pipeline 同时只能跑 1 个 batch。 加载 2 份 = 2 个 GPU context，A 在跑 layout 时 B 已经在另一个 context 跑 VL，**真正同时占 GPU**。 实测吞吐提升 ~1.5-1.8x。

### 5.2 max-min fair 调度

- **问题**：长任务（90 页 PDF）会占满 batch，新小任务（1 张图）一直等
- **解法**：`_take_fair_batch` 按 `(已完成数, 首次出现顺序)` 排序
  - 第一轮：每个 user 各拿 1 张 → 公平起点
  - 第二轮：completed 少的 user 继续拿 → 防止"富者愈富"
- **实测**：A 90 页 PDF + B 5s 后 1 张图 → B 在 A 还在跑第 2 批时就完成

### 5.3 心跳 / 释放 / cleanup 三件套

| 组件 | 谁 | 作用 |
|---|---|---|
| `heartbeat(user_id)` | 前端定时器 (10s) | 续约 `_user_heartbeat[user]` |
| `release_user(user_id)` | 前端 pagehide (sendBeacon) | 立即取消该 user 的所有 pending + PDF task |
| `_cleanup_loop` (5s 检查) | 后台协程 | 30s 没心跳的 user 自动 release |
| 客户端 cancel (关 tab) | `_process_pages` 的 `asyncio.gather` | 抛 `CancelledError` |
| 服务端 cancel (cleanup 释放) | `release_user → job.fut.cancel()` | 抛 `CancelledError` |

**`CancelledError` → 200 cancelled JSON (不是 500)**：FastAPI 默认把 `CancelledError` 当异常返回 500，handler 层 `try/except` 抓它转成业务状态。

### 5.4 PDF 异步流式 (202 + poll)

- **为什么不全 gather 等完再返回**：90 页 PDF 跑 5 分钟，HTTP timeout 会断
- **为什么不用 SSE/WebSocket**：简单 1.5s 轮询够用，且前端逻辑简单
- **`pil_pages` 保留在 progress dict**：进度响应里 `image_b64` 也要给前端，前端有它就有了（不需要再向 server 拿）

### 5.5 VL forward 拆 sub-batch (`max_forward_batch=4`)

- 8GB 显卡单次 VL forward 最多 4 张是安全阈值
- 跨用户 label 桶后可能一次 10+ 张 → `recognize_batch` 内部 `range(0, N, 4)` 拆 sub-batch

### 5.6 layout detect 批 + VL 按 label 分桶

- **layout detect**：单次 batched forward 跑 N 张图（同尺寸范围）
- **VL**：不能跨 label 拼 batch（不同 label 用不同 prompt），所以**先按 label 分桶再 batch**
- **两层批**：`N 张图 → 1 次 layout → 跨图按 label 桶 → 每桶 1 次 VL`

### 5.7 文件落盘隔离 `text_result/{request_id}/`

- 防止不同请求的 markdown 互相覆盖
- `request_id` 可以是 `web-abc123`（前端心跳用）也可以是 `uuid4().hex[:8]`
- **`_unique_path` 处理同 (page, label_name) 重名** → 自动加 `_1 _2 ...`

### 5.8 日志按日期独立文件

- `TimedRotatingFileHandler` 的 suffix 模式会有 `server.log` + `server.log.YYYY-MM-DD`，管理麻烦
- `_DailyFileHandler` 是**独立文件**模式：`server-2026-07-24.log`、`server-2026-07-25.log`
- roll 在 `emit()` 第一次发现跨天时懒触发，不需要后台线程
- 启动时 `_cleanup_old_logs(LOG_KEEP_DAYS)` 清超过 14 天的

---

## 6. 配置项全表

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `HOST` | `0.0.0.0` | uvicorn bind 地址 |
| `PORT` | `8000` | uvicorn bind 端口 |
| `PIPELINE_POOL_SIZE` | `2` | PipelinePool 大小, 同一进程内并发路数 |
| `BATCH_MAX` | `5` | 一次最多拼几张图 |
| `BATCH_FLUSH_MS` | `250` | 凑批等待超时 (ms) |
| `PDF_DPI` | `200` | PDF 渲染清晰度 (PDF 坐标系 72dpi) |
| `MAX_IMAGE_UPLOAD_MB` | `32` | 单图上传上限 |
| `MAX_PDF_UPLOAD_MB` | `128` | PDF 上传上限 |
| `MAX_PDF_PAGES` | `200` | PDF 页数上限 |
| `HEARTBEAT_TIMEOUT_S` | `30` | 心跳超时 (s), 超过自动 release |
| `CLEANUP_INTERVAL_S` | `5` | cleanup 协程检查间隔 (s) |
| `LOG_DIR` | `./logs` | 日志目录 |
| `LOG_KEEP_DAYS` | `14` | 旧日志自动清 (启动时扫) |
| `CORS_ALLOW_ORIGINS` | `http://localhost:*,http://127.0.0.1:*` | CORS 白名单, 逗号分隔 |
| `GRACEFUL_SHUTDOWN_S` | `15` | Ctrl+C 后等待正在跑的请求完成时间 |
| `OUTPUT_DIR` | `./text_result` | OCR 产物根目录 |

---

## 7. 重构时的关注点（潜在问题 / 改进方向）

> 这部分是给"重构"留的——当前实现能跑，但有一些"如果重构我会改的地方"：

1. **`active_users = len(scheduler._user_heartbeat)`** (demo.py:432)
   - 直接读 scheduler 内部下划线字段
   - 改：scheduler 暴露 `active_user_count()` 方法

2. **`OCRPipeline.process_batch` 的 `elapsed_seconds` 是整批 wall time** (core_pipeline.py:648)
   - 每个 PageResult 都拿到同一个 batch elapsed，**不准确**
   - 改：每个 page 自己计 elapsed（`time.perf_counter() - st` at PageResult 构造点）

3. **`process_page` 和 `process_batch` 部分逻辑重复** (core_pipeline.py:496 vs 569)
   - filter+crop+write 三段代码各写了一遍
   - 改：抽 `_process_one_page(job, image, layouts) -> PageResult` 私有方法，`process_page` / `process_batch` 都调它

4. **`_take_fair_batch` 两轮循环可以合并** (core_pipeline.py:898-929)
   - 第一轮 + 第二轮其实可以一次循环按 fair 顺序填到 max_n
   - 改：单循环，但每轮按 sorted key 取一个 user 的一个 job

5. **`_make_pipeline` 工厂放在 demo.py 没必要** (demo.py:179)
   - `make_default_pipeline` 已经在 `core_pipeline.py` 里有
   - 改：把 device 选择也搬过去，demo.py 直接 import `make_default_pipeline`

6. **CORS 配置写死的 `http://localhost:*` / `http://127.0.0.1:*`**
   - 通配符 `*` 在 CORS 标准里其实不合法（`Access-Control-Allow-Origin: *` 不能带 credentials）
   - 改：默认值改成 `["http://localhost:8000", "http://127.0.0.1:8000"]`，需要通配符让用户显式设

7. **PDF 进度轮询是 1.5s 固定间隔**
   - 前 10 页 (快) 显得浪费，后 80 页 (慢) 显得不够快
   - 改：adaptive backoff (前 5s 1s 一次, 5-30s 2s 一次, 30s+ 3s 一次)，或者切 SSE

8. **`text_result` 目录里 markdown 文件名没带 user_id 但落盘在 user_id 子目录**
   - `text_result/{user_id}/page_0_box_14_text.md` —— OK, 但如果某 user 多次 submit 同 page_index 会冲突
   - 已有 `_unique_path` 加 `_1 _2 ...` 兜底
   - 改：可以加 `request_uuid` 进一步隔离

9. **uvicorn access log 不进 `_DailyFileHandler`**
   - 控制台有，但文件没有
   - 改：单独给 `uvicorn.access` logger 也加 `_DailyFileHandler`

10. **PDF 进度清理是 10min 写死** (demo.py:706)
    - 长 PDF 用户拉完最后一次状态可能要等更久
    - 改：暴露成环境变量 `PDF_PROGRESS_TTL_S=600`

---

## 8. 总结：模块依赖图

```
                            ┌──────────────┐
                            │ static/      │
                            │ index.html   │
                            └──────┬───────┘
                                   │ HTTP
                                   ▼
       ┌──────────────────────────────────────────────────┐
       │ demo.py                                            │
       │                                                    │
       │  FastAPI app + lifespan                            │
       │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
       │  │ /ocr/upload  │  │ /ocr/pdf     │  │ /api/*   │ │
       │  │ /ocr/base64  │  │ + background │  │ hb/rel   │ │
       │  └──────┬───────┘  └──────┬───────┘  └─────┬────┘ │
       │         │                 │                │      │
       │         └────────┬────────┘                │      │
       │                  ▼                          │      │
       │         ┌──────────────────┐               │      │
       │         │ _process_pages   │               │      │
       │         │ scheduler.submit │◀──────────────┘      │
       │         │ gather(j.fut)    │                      │
       │         └────────┬─────────┘                      │
       └──────────────────┼────────────────────────────────┘
                          │ submit/await fut
                          ▼
       ┌──────────────────────────────────────────────────┐
       │ core_pipeline.py                                   │
       │                                                    │
       │  BatchScheduler (n_workers=2)                     │
       │  ├─ per-user queue (max-min fair)                  │
       │  ├─ heartbeat/release/cleanup                     │
       │  └─ _worker_loop × 2                               │
       │       │                                            │
       │       │ lease pipeline                             │
       │       ▼                                            │
       │  PipelinePool (asyncio.Queue, size=2)              │
       │  ├─ OCRPipeline #1 (cuda:0, full model)            │
       │  └─ OCRPipeline #2 (cuda:0, full model)            │
       │       │                                            │
       │       │ process_batch(jobs)                        │
       │       ▼                                            │
       │  ┌─────────────┐ ┌────────┐ ┌──────────┐          │
       │  │ Layout      │ │ Box    │ │ Region   │          │
       │  │ Detector    │→│ Filter │→│ Cropper  │          │
       │  │ (batched)   │ │ (NMS)  │ │ (numpy)  │          │
       │  └─────────────┘ └────────┘ └────┬─────┘          │
       │                                  │                 │
       │                                  ▼                 │
       │                            ┌──────────┐           │
       │                            │ VL       │           │
       │                            │ Predictor│           │
       │                            │ (label   │           │
       │                            │ bucketed │           │
       │                            │ batched) │           │
       │                            └────┬─────┘           │
       │                                 │                 │
       │                                 ▼                 │
       │                    text_result/{user_id}/         │
       │                    page_X_box_Y_*.md              │
       └──────────────────────────────────────────────────┘
```

---

## 9. 修订记录

| 日期 | 改动 | 作者 |
|---|---|---|
| 2026-07-25 | 初版: 记录重构前 baseline | Mavis |
