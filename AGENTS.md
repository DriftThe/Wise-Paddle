# AGENTS.md — Wise-Paddle

> 面向 AI 编码代理（以及所有贡献者）的仓库指南。读这个文件能让你快速理解
> 项目怎么跑、代码怎么组织、改动时该注意什么。
> 用户文档见 [README.md](README.md)；架构说明见 [ARCHITECTURE.md](ARCHITECTURE.md)。

---

## 1. 项目是什么

Wise-Paddle 是一个基于 PaddleOCR 的**文档 OCR 服务**：把图片 / PDF 中的版面
区域自动识别出来，再转成结构化 Markdown。核心处理流水线：

```
PIL.Image → LayoutDetector(PP-DocLayoutV3) → BoxFilter(NMS+扩框) → RegionCropper
          → VLPredictor(PaddleOCR-VL-1.6, 按 label 分桶 batch 推理) → Markdown
```

服务形态是 FastAPI（`app.py`）+ 单页前端（`static/index.html` + `static/app.js`）。
推理并发由 `core_pipeline.py` 的 `PipelinePool` + `BatchScheduler` 支撑。

## 2. 快速命令

| 目的 | 命令 |
|------|------|
| 启动服务 | `.\.venv\Scripts\python.exe app.py`（等价 `uvicorn app:app --host 127.0.0.1 --port 8000`） |
| 运行单测 | `.\.venv\Scripts\python.exe -m unittest tests.test_validation -v` |
| 运行全部单测 | `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` |
| 语法检查 | `.\.venv\Scripts\python.exe -m py_compile app.py core_pipeline.py` |

> 依赖在 `.venv`（Python 3.12）。模型需要 GPU（推荐 CUDA ≥ 6GB），CPU 也能跑但极慢。
> 模型权重**不随仓库分发**，需自行下载到 `./model/`（见 README §2）。

## 3. 仓库结构

```
Wise-Paddle/
├── app.py               # FastAPI 服务入口：路由、配置、上传/取消/心跳、PDF 后台任务
├── core_pipeline.py     # 核心流水线：LayoutDetector / BoxFilter / RegionCropper /
│                        #   VLPredictor / OCRPipeline / PipelinePool / BatchScheduler
├── static/
│   ├── index.html       # 前端单页 UI（暗色主题，Canvas 画检测框）
│   └── app.js           # 前端逻辑（上传、心跳保活、进度轮询、结果渲染）
├── tests/
│   └── test_validation.py  # 单元测试（校验/取消/公平性/像素预算等，无模型依赖）
├── model/               # 模型权重目录（不入库）
├── text_result/         # OCR 产物（按 request_id 分目录）
├── logs/                # 按日期滚动日志 server-YYYY-MM-DD.log
├── .env                 # 运行配置（不入库，模板见 .env.example）
└── AGENTS.md / README.md / ARCHITECTURE.md / CODE_REVIEW.md / TODO.md
```

## 4. 核心架构速览（改代码前必读）

### 4.1 数据类（core_pipeline.py）

- `LayoutBox` — 单个版面框：`xyxy`(float)、`label_id/name`、`score`
- `RegionResult` — 单个裁剪区域的最终结果（含 `markdown` 与落盘路径 `md_path`）
- `PageResult` — 一页的结果；`cancelled=True` 表示该页被取消过滤，`regions` 为空
- `Job` — 一个待处理 page：`request_id` + `page_index` + `image` + `fut`(Future) +
  `voucher_id` + `generation`

### 4.2 OCRPipeline（core_pipeline.py）

- `process_page(image, page_index, request_id)` — 单图端到端（调试/旧调用方用）
- `process_batch(jobs, cancelled_vouchers, cancelled_requests)` — **生产主路径**，
  跨用户共享 layout 检测 + 按 label 聚合的 VL 推理；`md_lookup[(page_in_batch,
  crop_idx)]` 定位 markdown，最后落盘到 `text_result/{request_id}/` 下
- `_crop_page` / `_write_regions` — 过滤裁剪 / 写盘 + 构造 `RegionResult`

### 4.3 BatchScheduler（core_pipeline.py）

- max-min fairness：每个 user 独立 pending 队列，按“已完成数”升序拼批
- 取消维度有两套：
  - `cancel_voucher(voucher_id)` — 会话级（用户离线/心跳超时），清整批
  - `cancel_request(request_id)` — 请求级（前端点 remove），只清这一条
- `generation` 机制：每次上传申请新代数，避免旧取消标记误伤新上传
- worker 用 `run_in_threadpool(pipe.process_batch, ...)` 跑推理，并用
  `asyncio.shield` 保证 worker 被取消时 pipeline 在线程跑完前不被提前归还

### 4.4 app.py 关键机制

- 全部可调参数由 `.env` 驱动（`Config` dataclass），空值回退到代码默认值
- `_process_pages()` 提交单图/PDF 页 → `asyncio.gather(job.fut)` 等结果
- PDF 走异步进度：`/ocr/pdf` 返回 202 + `user_id`，`/api/ocr/pdf-status/{user_id}`
  轮询；每完成一页更新 `_pdf_progress_store`
- 心跳保活：前端 `/alive` 刷新 voucher TTL，`_kick_dead_user_loop` 归零即
  `scheduler.cancel_voucher` 取消任务
- 取消/进度接口有 voucher 归属校验（`_check_voucher_guard`），防猜 ID

## 5. 工程约定

### 5.1 代码风格

- 类型标注：新代码尽量带完整类型（`list[...]` / `dict[...]` / `Optional`）
- 注释：关键坑（RoPE 补丁、left-padding、threadpool 边界）必须写“为什么”
- 日志：用模块级 `logger = logging.getLogger("wise-paddle...")`，不要 `print`
- 文档字符串：公开类/方法写 docstring；内部 helper 至少一句话注释

### 5.2 并发与线程安全（容易踩的坑）

- `BatchScheduler` 内部状态由 `asyncio.Lock`（`self._lock`）保护；**所有访问
  `_user_pending` / `_cancelled_*` 的路径都要持锁**
- `process_batch` 在**线程池线程**里跑（`run_in_threadpool`），因此它不能触碰
  asyncio 对象；取消集合（`cancelled_vouchers` / `cancelled_requests`）以参数
  快照传入，只读
- `PipelinePool.lease()` 是 `asynccontextmanager`，保证“谁借谁还”；worker 被
  取消时不能提前归还 pipeline（见 `_run_batch` 的 `asyncio.shield`）
- 不要在 `await` 之间做跨协程共享状态的“读-改-写”而不加锁

### 5.3 安全基线（改动涉及上传/路径/ID 时对照）

- `user_id` / `voucher_id` 必须走 `_sanitize_user_id` / `_sanitize_voucher_id`
  （白名单正则，防路径穿越 / 超长 / 控制字符）
- 写盘路径必须经 `OCRPipeline._safe_request_dir` 做 `resolve()` + `is_relative_to`
  校验（纵深防御）
- 上传大小/像素预算：`_read_upload_with_limit`、`_check_pixel_budget`、
  `_decode_raw_image`（先读头部尺寸再解码）——新增上传入口必须复用这些
- 日志里插用户可控字符串（文件名、ID）前用 `_safe_filename` / 已 sanitize 的 ID

### 5.4 新增/修改配置

1. 在 `.env.example` 加条目（带单位/区间/默认值/调参影响注释）
2. 在 `Config` dataclass 加字段（用 `_env_str/int/float/bool` 读）
3. 在 `_make_pipeline()`（若影响模型/推理）透传；启动日志打印关键值

## 6. 测试

- 测试无模型依赖（不加载 torch 模型），可离线跑：`unittest tests.test_validation`
- 新增逻辑（尤其调度/取消/校验/资源限制）应补单测；取消与代数相关的用例参考
  `TestCancelJobsAndGenerations`、`TestVoucherGuard`、`TestPruneIdleUsers`
- 需要 GPU/模型的端到端行为不在单测范围，改动 `process_batch` 后至少保证现有
  单测全绿

## 7. 常见改动场景

| 你想做什么 | 改哪里 |
|-----------|--------|
| 换 layout 模型 | `LayoutDetector.__init__`（保证输出 `list[LayoutBox]`） |
| 换 VL 模型 | `VLPredictor.__init__` / `_forward_once`（保证 `AutoModelForImageTextToText` 接口） |
| 调 OCR 精度 | `.env`：`BOX_UNCLIP_RATIO` / `BOX_EXPAND_PIXELS` / `VL_*` |
| 改调度策略 | `BatchScheduler._take_fair_batch`（纯逻辑，可替换） |
| 加取消/心跳逻辑 | `app.py` 的 voucher 相关函数 + `BatchScheduler.cancel_*` |
| 加后处理 | `OCRPipeline.process_batch` 的写盘段 / `_write_regions` |

## 8. 注意事项 / 已知限制

- **RoPE 兼容性**：`core_pipeline.py` 导入时自动 patch
  `transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"]`，这是模块级
  副作用，改动 transformers 相关代码时别删
- **VL 必须用 `AutoModelForImageTextToText`**（不是 CausalLM），否则模型看不到图
- **VL batch 必须 left-padding**（代码已设置 `padding_side="left"`，勿覆盖）
- 心跳 TTL（`ALIVE_INITIAL_TTL`）默认 120，过小会在用户切后台标签页时误杀任务
- `text_result/` 可能被外部工具清掉，`OCRPipeline` 每次写盘前都会 `mkdir`
