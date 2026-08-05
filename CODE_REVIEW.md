# Wise-Paddle 代码审查报告

> 审查日期：2026-08-05
>
> 审查范围：`app.py`（后端服务）、`core_pipeline.py`（OCR 流水线）、`static/app.js`、`static/index.html`（前端）、`requirements.txt`、`.env.example`、`.gitignore`，并交叉核对 `README.md` / `ARCHITECTURE.md` / `USAGE.md`
>
> 结论：整体工程质量较好——分层清晰、并发与取消机制设计用心、前端 XSS 防护到位。但存在 **1 个可直接利用的任意路径写文件漏洞** 和若干 **内存/资源耗尽风险**，上线前建议优先修复"高"优先级问题。

---

## 一、问题汇总

| 编号 | 级别 | 问题 | 位置 |
|------|------|------|------|
| H-01 | 高 | `user_id` 未校验 → 路径穿越，可越界写文件 | app.py:748/792/998，core_pipeline.py:716 |
| H-02 | 高 | PDF 页数上限在完整渲染之后才检查 → 内存耗尽 | app.py:983/992 |
| H-03 | 高 | `/ocr/base64` 无大小限制 + 图像解压炸弹 | app.py:785-791 |
| H-04 | 高 | 后台标签页心跳被节流 → 长时间任务被误取消 | app.js:41/210，app.py:139-142 |
| H-05 | 高 | scheduler 的 user 台账无限增长 → 内存泄漏 | core_pipeline.py:992-993/1063/1384 |
| M-01 | 中 | CORS 默认配置实际不生效（端口通配符不支持） | app.py:170-171 |
| M-02 | 中 | 响应内嵌全尺寸 base64 图；PNG 编码阻塞事件循环 | app.py:682/932 |
| M-03 | 中 | 文档与代码不一致（`/api/release` 已不存在） | README.md:85/153/306 |
| M-04 | 中 | 取消/进度接口无鉴权，用户 ID 客户端可控 | app.py:1041/1073/1123 |
| M-05 | 中 | 全局异常处理器向客户端泄露内部错误 | app.py:469-477 |
| M-06 | 中 | `_kick_dead_user_once` TTL 双检逻辑缺陷 | app.py:1170-1182 |
| M-07 | 中 | 同一请求并发写目录时 `_unique_path` 存在竞态 | core_pipeline.py:700-705 |
| M-08 | 中 | 前端 PDF 预览无页数上限 | app.js:256 |
| M-09 | 中 | opencv-python 与 opencv-contrib-python 重复依赖 | requirements.txt:58-59 |
| M-10 | 中 | `.env` 被 Git 跟踪 | .gitignore |
| M-11 | 中 | 前端超时/清空队列不清理服务端任务 | app.js:472 |
| M-12 | 中 | PDF 轮询每次都全量返回已完成页 base64 | app.py:1057-1069 |
| M-13 | 中 | PDF 进度 store 同时持有 PIL 页 + base64，内存翻倍 | app.py:1009/932 |
| L-01 | 低 | 版本号不一致（后端 0.5.0 vs 前端 0.6.0） | app.py:219 |
| L-02 | 低 | `Image.BILINEAR` 弃用 | core_pipeline.py:224 |
| L-03 | 低 | MD 文件名含模型 label，未消毒 | core_pipeline.py:700 |
| L-04 | 低 | 响应泄露服务端 `md_path` 绝对路径 | app.py:710/938 |
| L-05 | 低 | 日志注入（`file.filename` 直接拼接） | app.py:751 |
| L-06 | 低 | 无任何自动化测试 | 项目根 |
| L-07 | 低 | `server.log` 等运行产物未忽略 | .gitignore |
| L-08 | 低 | `/alive` 接受空 voucher | app.py:1130 |
| L-09 | 低 | 单页 `elapsed_seconds` 实为整批耗时，语义偏差 | core_pipeline.py:842 |
| L-10 | 低 | 每个 region 裁剪都重新 `convert("RGB")` 整图 | core_pipeline.py:387 |
| L-11 | 低 | 取消追踪 GC 日志中 voucher 数恒为 0，误导 | core_pipeline.py:1197 |
| L-12 | 低 | 密码加密 PDF 解析失败返回 500 而非 400 | app.py:986-988 |
| L-13 | 低 | EXIF 方向不处理，手机竖拍图可能横置 | app.py:744 |

---

## 二、高优先级问题

### H-01 `user_id` 未校验 → 路径穿越，可越界写文件

**位置**：`app.py:748`、`app.py:792`、`app.py:998`；`core_pipeline.py:716`、`core_pipeline.py:698-702`

三个上传入口都直接取客户端 query 参数作为输出目录名：

```python
request_id = request.query_params.get("user_id") or uuid.uuid4().hex[:8]
```

随后 `core_pipeline.py` 用 `self.output_dir / job.request_id` 拼路径，并执行 `mkdir` + `write_text`。

**已验证**：`Path("text_result") / "../../evil"` 解析后落在项目目录之外。攻击者提交 `user_id=../../some/dir` 即可在服务账户可写范围内任意创建目录并写入 `.md` 文件（内容为模型生成的文本，可控性有限，但目录/文件名可任意指定，可覆盖文件、污染其他用户目录）。

**附带风险**：该 ID 同时用于 `/api/cancel/{user_id}` 与 `/api/ocr/pdf-status/{user_id}`（见 M-04），攻击者也可用他人 ID 取消任务或读取进度。

**建议**：
1. 服务端生成 ID（`uuid4().hex`），不信任客户端传入；
2. 若必须接受客户端 ID，校验 `^[A-Za-z0-9_-]{1,64}$`，拒绝 `/`、`\`、`.`、空串；
3. 写盘前 `resolved = (output_dir / request_id).resolve()`，并断言 `resolved.is_relative_to(output_dir.resolve())`。

---

### H-02 PDF 页数上限在完整渲染之后才检查 → 内存耗尽

**位置**：`app.py:579-600`（`_decode_pdf_to_pil_pages` 渲染全部页面）、`app.py:983`（先解码）、`app.py:992`（后检查页数）

```python
pil_pages = await run_in_threadpool(_decode_pdf_to_pil_pages, raw, CFG.pdf_dpi)
# ... 全部页面已在内存中 ...
if len(pil_pages) > CFG.max_pdf_pages:   # 太迟了
```

一个 1000 页的 PDF 会被**完整光栅化**后才被 413 拒绝。按 200 DPI A4 估算，每页约 1654×2339×3 ≈ 11.6 MB，100 页即 1.2 GB 峰值。这是无需认证的内存耗尽攻击面。

**建议**：
1. 打开 PDF 后先取 `len(pdf)`，超限立刻 413；
2. 解码时对总像素（`width×height×pages`）设置硬上限；
3. 考虑逐页解码、逐页入队，边解码边释放。

---

### H-03 `/ocr/base64` 无大小限制 + 图像解压炸弹

**位置**：`app.py:785-791`（`ocr_base64`）、`app.py:555-562`（`_decode_b64_to_rgb`）

与 `/ocr/upload`（有 `Content-Length` + 文件大小双重校验）不同，`/ocr/base64` 路径：

- 没有 `_check_content_length`；
- 没有 payload 长度上限（任意大的 base64 字符串都会被完整解码）；
- 没有解码后的像素尺寸上限——一张几十 KB 的 PNG 可解码为 20000×20000 的超大数组（解压炸弹），`cv2.imdecode` 与后续 `_pil_to_b64_png` 会分配数 GB 内存。

`/ocr/upload` 虽限 32 MB 文件，但同样没有像素尺寸上限，解压炸弹仍然成立。

**建议**：统一对 base64 长度、解码后 `width×height` 设置上限（如 `MAX_IMAGE_PIXELS`），超限返回 413/400。

---

### H-04 后台标签页心跳被节流 → 长时间任务被误取消

**位置**：`app.js:41`（心跳间隔 5s）、`app.js:210`；`app.py:139-142`（TTL=7 秒、每秒 tick）

浏览器会对后台/隐藏标签页的 `setInterval` 做节流（Chrome 对隐藏页可达 1 次/分钟）。用户处理长 PDF 时切到其他标签页，心跳停止，7 秒后 `_kick_dead_user_once` 调 `scheduler.cancel_voucher` 把整批任务取消——**用户只是切了个标签页，任务就被杀了**。这对 PDF 场景（用户几乎必然切走等待）是功能性缺陷。

**建议**：
1. TTL 提高到 30~60 秒（`ALIVE_INITIAL_TTL`）；
2. 前端改用 `document.visibilitychange` + `fetch(keepalive: true)` 或 `navigator.sendBeacon` 心跳，不依赖 `setInterval` 精度；
3. 或者对已提交的任务，心跳超时只停止"新提交"而不强杀 in-flight 工作。

---

### H-05 scheduler 的 user 台账无限增长 → 内存泄漏

**位置**：`core_pipeline.py:992-993`（`_user_order` / `_user_completed`）、`:1063`（submit 时 append）、`:1281-1284`（只删 pending 队列、保留顺序历史）、`:1384`（completed 计数只增不减）

每个 upload 都生成新的 `web-xxxx` 随机 ID（`app.js:446`），`submit()` 会把每个新 ID 永久加入 `_user_order` 列表，`_run_batch` 又永久累加 `_user_completed`。`_maybe_gc_cancelled_tracking`（`:1170-1200`）只清理 `_cancelled_requests`，不清理这两处。长跑服务（哪怕只有少量活跃用户）会无限增长这两个结构，每次公平排序还要遍历全部历史用户。

**建议**：用户队列清空且超过空闲窗口（如 10 分钟）后，从 `_user_order` / `_user_completed` 中移除；或用 LRU/定期压缩；排序只针对活跃用户。

---

## 三、中优先级问题

### M-01 CORS 默认配置实际不生效

`app.py:170-171` 默认 `http://localhost:*,http://127.0.0.1:*`。但已安装的 Starlette（`starlette/middleware/cors.py:105`）只做精确字符串匹配：`return origin in self.allow_origins`，除 `"*"` 外不支持任何通配符/端口模式。因此 `http://localhost:8000` 这类真实来源**不会匹配**，跨端口访问前端时浏览器会拦截请求。默认值与注释宣称的行为不符。

**建议**：改用 `allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?"`，或写死明确 origin 列表；不要写带 `*` 的模式。

---

### M-02 响应内嵌全尺寸 base64；PNG 编码阻塞事件循环

**位置**：`app.py:682`、`app.py:932`（`_pil_to_b64_png` 同步调用）、`app.py:570-576`

1. PNG 编码是 CPU 密集操作，在 async 事件循环内同步执行，一张大图即可让整个服务"卡死"所有其他请求；PDF 每页编码同理。
2. 单图响应把**原始全尺寸** PNG base64 塞进 JSON：32 MB 上传可能产生 100 MB+ 响应；200 页 PDF 的进度 store 内存成倍膨胀。

**建议**：编码放入 `run_in_threadpool`；响应前对图片做归一化（限制最长边）；考虑返回文件 URL 而非 base64，或支持增量/分页取图。

---

### M-03 文档与代码严重不一致

`README.md:85/153/306`、`ARCHITECTURE.md:4/20/73/1242` 均描述 `POST /api/release`（关页面主动释放），但 `app.py` 中**不存在该路由**（现状是 `/alive` + `/api/cancel/{user_id}`）。`ARCHITECTURE.md:4` 甚至声称"彻底删除了心跳机制"，而当前代码恰恰新增了 alive voucher 心跳。照文档调用会得到 404。

**建议**：以代码为准重写 README/ARCHITECTURE 的接口章节；或把 `/api/release` 作为 `/api/cancel` 的兼容别名实现。

---

### M-04 取消/进度接口无鉴权，用户 ID 客户端可控

`/api/cancel/{user_id}`（app.py:1073）、`/api/ocr/pdf-status/{user_id}`（app.py:1041）、`/alive`（app.py:1123）均无认证。`user_id` 由客户端自选（H-01），意味着任何人都能：取消他人的任务、读取他人 OCR 进度（含页面图片）、冒充他人 ID 上传导致输出目录内容被混写。

**建议**：至少要求取消/查询时携带服务端签发的 `voucher_id` 且必须匹配；正式部署加会话/Token 鉴权。

---

### M-05 全局异常处理器泄露内部错误信息

`app.py:469-477` 把 `type(exc).__name__: exc` 原样返回给客户端，可能泄露文件路径、模型内部错误等敏感信息。

**建议**：返回固定 `500 internal error`，详情只进服务端日志。

---

### M-06 `_kick_dead_user_once` 的 TTL 双检逻辑缺陷

**位置**：`app.py:1168-1182`

```python
new_ttl = ttl - 1                 # 快照值减一
async with _alive_lock:
    current = _alive_users.get(voucher)
    ...
    new_ttl = min(current, new_ttl)   # 取"更小值"，快照期间被 /alive 重置的 TTL 也会被减
```

若快照 TTL=1，随后 `/alive` 把 TTL 重置为 7，本 tick 仍得到 `min(7, 0)=0` → 把**刚续期的活跃用户**踢下线。双检的意图被 `min` 取小逻辑破坏了。

**建议**：先比较 `current == snapshot_ttl` 再递减；否则不动作（说明已被 /alive 重置）。

---

### M-07 同一请求并发写目录时 `_unique_path` 存在竞态

**位置**：`core_pipeline.py:700-705`（check-then-write）、`:1310/1327`（多 worker 可从同一用户队列取不同页）

两个 worker 可能把同一 `request_id` 的不同页分到不同 batch，同一时刻向同一 `req_dir` 写同名 `page_X_box_Y_label.md`。`_unique_path` 是"先检查后写入"，线程间存在 TOCTOU 竞态，可能互相覆盖或产生意外后缀。

**建议**：文件名加入 `uuid4().hex` 短后缀（内容不依赖路径唯一性）；或对同一 request 的写入串行化。

---

### M-08 前端 PDF 预览无页数上限

`app.js:256` 对 `pdf.numPages` 全量逐页渲染。上千页 PDF 会让浏览器先卡死，随后才被服务端 413 拒绝。

**建议**：预览只渲染前 N 页（如 20 页），并提示"超过服务端上限的 PDF 将无法处理"。

---

### M-09 opencv 依赖重复

`requirements.txt:58-59` 同时安装 `opencv-contrib-python` 与 `opencv-python`，两者都提供 `cv2` 包，属重复/冲突依赖。保留 contrib（超集）即可。

---

### M-10 `.env` 被 Git 跟踪

`git ls-files` 显示 `.env` 已入库，`.gitignore` 未包含 `.env`。当前内容仅配置项、无明文密钥，但一旦后续加入 API Key 就会随仓库泄露。

**建议**：`git rm --cached .env`，`.gitignore` 增加 `.env`，保留 `.env.example`。

---

### M-11 前端超时/清空队列不清理服务端任务

`app.js:472` 15 分钟超时 `abort` 后仅把卡片标为 error，服务端任务继续占用 GPU；`clearQueueBtn` 清空队列时也不对 processing 项发 `/api/cancel`。

**建议**：超时/清空/`pagehide` 时统一调 `/api/cancel/{user_id}`（fire-and-forget）。

---

### M-12 PDF 轮询每次都全量返回已完成页 base64

`app.py:1057-1069` 每次轮询返回 `pages` 全量（含每页 `image_b64`），前端每 1.5s 拉一次（app.js:43）。100 页 PDF 每次轮询都重复传输几十 MB base64。

**建议**：支持 `?since=<page_index>` 增量参数，或把页面图像与元数据接口分离。

---

### M-13 PDF 进度 store 内存双重持有

`app.py:1009` 保存整份 `pil_pages`，同时 `_make_pdf_page_dict`（app.py:932）为每页生成 base64 存进 `pages`，TTL 600 秒内两份大对象常驻。200 页 200 DPI PDF 可达数 GB 峰值。

**建议**：完成后立即释放 `pil_pages`（当前在 `finally` 里清空 `progress["pil_pages"]`，但 base64 仍全量保留），或对总内存设上限、改流式取图。

---

## 四、低优先级 / 改进建议

| 编号 | 问题 | 位置 | 说明 |
|------|------|------|------|
| L-01 | 版本号不一致 | app.py:219 | `_app_version="0.5.0"`，前端头注释 v0.6.0 |
| L-02 | `Image.BILINEAR` 弃用 | core_pipeline.py:224 | 新版 Pillow 建议 `Image.Resampling.BILINEAR` |
| L-03 | MD 文件名未消毒 | core_pipeline.py:700 | `label_name` 直接拼文件名，遇特殊字符可能出问题；建议白名单替换 |
| L-04 | 响应泄露 `md_path` 绝对路径 | app.py:710/938 | 客户端不需要服务端路径，可去掉或改为相对路径 |
| L-05 | 日志注入 | app.py:751 | `file.filename` 可含换行/控制字符；建议转义或截断 |
| L-06 | 无自动化测试 | 项目根 | 调度/取消/公平性是核心逻辑，建议至少补 scheduler 与取消链路的单测 |
| L-07 | `server.log` 未忽略 | .gitignore | 运行产物不应入库；建议与 `logs/` 一起忽略 |
| L-08 | `/alive` 接受空 voucher | app.py:1130 | 空串也会进 `_alive_users`；建议拒绝空串 |
| L-09 | 单页耗时语义偏差 | core_pipeline.py:842 | `PageResult.elapsed_seconds` 用的是整批 wall time，每页相同 |
| L-10 | 重复整图转 RGB | core_pipeline.py:387 | 每 region 都 `image.convert("RGB")` 一次；应每页转换一次再切片 |
| L-11 | GC 日志误导 | core_pipeline.py:1197 | `_cancelled_vouchers` 从不删除，`gc_v` 恒为 0 |
| L-12 | 加密 PDF 返回 500 | app.py:986-988 | 密码保护 PDF 应返回 400 + 友好提示 |
| L-13 | EXIF 方向不处理 | app.py:744 | 手机竖拍图用 `cv2.imdecode` 会丢方向信息，建议读 EXIF 或改用 PIL 解码 |

---

## 五、值得肯定的设计

1. **流水线分层清晰**：LayoutDetector → BoxFilter → RegionCropper → VLPredictor 职责单一，参数全部可配，注释（尤其 doclayout 边界、RoPE 补丁、left-padding 等"坑"）质量很高。
2. **并发与取消设计用心**：max-min fair 拼批 + PipelinePool + 两层取消过滤（入口 / crop 后）思路正确，worker 异常通过 `job.fut.set_exception` 隔离，不会整队死亡。
3. **配置集中化**：`.env` 驱动的 `Config` + 详尽 `.env.example`，调参与代码解耦。
4. **前端安全基线好**：统一的 `escapeHtml`/`escapeAttr`、CSP、`textContent` 优先，注入面控制到位。
5. **运维配套**：按日滚动日志、启动清理旧日志、进度/文本结果 TTL 清理、优雅停机链路完整。

---

## 六、审查文件清单

- `app.py`（1208 行）
- `core_pipeline.py`（1415 行）
- `static/app.js`（1233 行）
- `static/index.html`（886 行）
- `requirements.txt`
- `.env.example` / `.gitignore`
- `README.md` / `USAGE.md` / `ARCHITECTURE.md`（交叉核对）
