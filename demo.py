"""
Wise-Paddle uvicorn 服务（批处理调度版 + PDF 多页 + 用户活跃检测）

设计：
- 单 pipeline 一次处理一批（最多 BATCH_MAX 张图）
- BatchScheduler 跨用户聚合 page 任务送给 pipeline
- layout detection 和 VL 推理都能 batch 起来
- 单 pipeline 实例能打满 GPU 吞吐
- 每个 request 有 user_id (request_id)，前端每 10s 发心跳，关 tab 主动 release；
  后台 cleanup 协程心跳超时自动释放 pending 任务，避免长任务卡死。

HTTP 接口：
- GET  /                          —— 前端 UI（单页 HTML）
- GET  /health                    —— 健康 + scheduler/pool 状态
- POST /ocr/upload                —— 上传图片（multipart），返回 1 个 OCRPage
- POST /ocr/base64                —— base64 JSON body，返回 1 个 OCRPage
- POST /ocr/pdf                   —— 上传 PDF，转 N 页 PNG 入队，返回 N 个 OCRPage
- POST /api/heartbeat             —— 续约 user 活跃状态，body: {"user_id": "..."}
- POST /api/release               —— 主动释放 user 所有 pending，body: {"user_id": "..."}

响应里 image_b64 是原图 PNG，前端拿到可以直接画到 canvas 上叠检测框。
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import logging.handlers
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from core_pipeline import (
    DEFAULT_OUTPUT_DIR,
    LAYOUT_MODEL_PATH,
    VL_MODEL_PATH,
    OCRPipeline,
    PipelinePool,
    BatchScheduler,
    Job,
)

load_dotenv()

# ─── 配置 ───────────────────────────────────────────────────────
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
POOL_SIZE = max(1, int(os.environ.get("PIPELINE_POOL_SIZE", "2")))
BATCH_MAX = max(1, int(os.environ.get("BATCH_MAX", "5")))
BATCH_FLUSH_MS = float(os.environ.get("BATCH_FLUSH_MS", "250"))
LOG_DIR = Path(os.environ.get("LOG_DIR", "./logs"))
LOG_KEEP_DAYS = int(os.environ.get("LOG_KEEP_DAYS", "14"))
PDF_DPI = float(os.environ.get("PDF_DPI", "200"))  # PDF 渲染清晰度

# 资源保护上限
MAX_IMAGE_UPLOAD_MB = float(os.environ.get("MAX_IMAGE_UPLOAD_MB", "32"))
MAX_PDF_UPLOAD_MB = float(os.environ.get("MAX_PDF_UPLOAD_MB", "128"))
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "200"))

# 前端心跳 / 后台清理
HEARTBEAT_TIMEOUT_S = float(os.environ.get("HEARTBEAT_TIMEOUT_S", "30"))
CLEANUP_INTERVAL_S = float(os.environ.get("CLEANUP_INTERVAL_S", "5"))

# CORS（默认允许本地所有端口；部署到生产改环境变量 CORS_ALLOW_ORIGINS）
CORS_ALLOW_ORIGINS = os.environ.get(
    "CORS_ALLOW_ORIGINS", "http://localhost:*,http://127.0.0.1:*"
).split(",")

STATIC_DIR = Path(__file__).parent / "static"


# ─── 日志（按日期写 logs/server-YYYY-MM-DD.log） ───────────────
def _setup_logging() -> None:
    """配置根 logger：控制台 + 按天滚动的文件 handler。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    formatter = logging.Formatter(fmt)

    # 控制台
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)

    # 按天滚动文件
    log_file = LOG_DIR / "server.log"
    fh = logging.handlers.TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=LOG_KEEP_DAYS,
        encoding="utf-8",
        utc=False,
    )
    # suffix 改成 YYYY-MM-DD 而不是默认 YYYY-MM-DD_N
    fh.suffix = "%Y-%m-%d"
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    root = logging.getLogger()
    root.handlers.clear()  # 避免 uvicorn 默认 handler 重复
    root.addHandler(sh)
    root.addHandler(fh)
    root.setLevel(logging.INFO)


_setup_logging()
logger = logging.getLogger("wise-paddle")

# ─── Scheduler 生命周期 ────────────────────────────────────────
scheduler: Optional[BatchScheduler] = None
pool_ref: Optional[PipelinePool] = None


def _make_pipeline() -> OCRPipeline:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return OCRPipeline(
        layout_model_path=LAYOUT_MODEL_PATH,
        vl_model_path=VL_MODEL_PATH,
        device=device,
        output_dir=OUTPUT_DIR,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler, pool_ref
    logger.info(
        "Building pipeline pool: %d pipeline(s), batch_max=%d, flush=%dms, pdf_dpi=%.0f",
        POOL_SIZE, BATCH_MAX, int(BATCH_FLUSH_MS), PDF_DPI,
    )
    pool = PipelinePool(size=POOL_SIZE, factory=_make_pipeline)
    await pool.init()
    pool_ref = pool

    sched = BatchScheduler(
        pool=pool,
        max_batch=BATCH_MAX,
        flush_ms=BATCH_FLUSH_MS,
        n_workers=POOL_SIZE,  # 1 worker per pipeline slot
        heartbeat_timeout=HEARTBEAT_TIMEOUT_S,
        cleanup_interval=CLEANUP_INTERVAL_S,
    )
    await sched.start()
    scheduler = sched
    logger.info(
        "Service ready: pool=%d, batch_max=%d, hb_timeout=%.0fs, cleanup=%.0fs",
        POOL_SIZE, BATCH_MAX, HEARTBEAT_TIMEOUT_S, CLEANUP_INTERVAL_S,
    )
    try:
        yield
    finally:
        logger.info("Shutting down ...")
        if scheduler is not None:
            await scheduler.close()
        if pool is not None:
            await pool.close()
        scheduler = None
        pool_ref = None


app = FastAPI(title="Wise-Paddle Service", version="0.4.0", lifespan=lifespan)

# CORS（前端通常跟后端同源，但支持跨域调试）
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── 全局错误处理：未捕获的异常 → 500 + JSON，避免半截 HTML ──────────
@app.exception_handler(Exception)
async def _unhandled_exc_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"internal error: {type(exc).__name__}: {exc}"},
    )


# ─── Request / Response 模型 ─────────────────────────────────────
class Base64Request(BaseModel):
    payload: str = Field(..., description="图片 base64 字符串，可带 data URI 前缀")


class OCRRegion(BaseModel):
    page_index: int = 0
    box_index: int = 0
    label: str
    score: float
    rect: tuple[int, int, int, int]  # x1, y1, x2, y2
    md_path: Optional[str] = None
    markdown: str


class OCRPage(BaseModel):
    page_index: int
    width: int
    height: int
    image_b64: str = Field(..., description="原图 PNG base64；前端直接画到 canvas")


class OCRResponse(BaseModel):
    success: bool
    request_id: str
    status: str = "done"  # done | pending
    elapsed_seconds: float
    queue_wait_seconds: float
    scheduler_pending: int
    pool_size: int
    pool_free: int
    pages: list[OCRPage]
    regions: list[OCRRegion] = Field(
        default_factory=list,
        description="所有 page 摊平的 regions；按 page_index 分组可对应到 pages[i]",
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    pool_size: int
    pool_free: int
    scheduler_pending: int
    batch_max: int
    active_users: int  # 仍在心跳的 user 数（pending + 已无 pending 但未 release）


class HeartbeatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64,
                          description="前端会话 ID（一般是首次请求的 request_id）")


class HeartbeatResponse(BaseModel):
    ok: bool
    pending: int
    server_time: float


class ReleaseRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)


class ReleaseResponse(BaseModel):
    ok: bool
    cancelled: int


# ─── 工具函数 ────────────────────────────────────────────────────
def _check_content_length(request: Request, max_mb: float) -> None:
    """拦 Content-Length 超限的上传，避免全部读进内存。"""
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            size = int(cl)
        except ValueError:
            return
        if size > max_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"上传文件超过 {max_mb:.0f}MB 限制（Content-Length={size} bytes）",
            )


def _strip_data_uri(payload: str) -> str:
    if not payload.startswith("data:"):
        return payload
    comma_idx = payload.find(",")
    if comma_idx == -1:
        raise ValueError("Invalid data URI: missing comma")
    return payload[comma_idx + 1:]


def _decode_b64_to_rgb(payload: str) -> np.ndarray:
    b64 = _strip_data_uri(payload)
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法将 base64 解码为图片")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _rgb_to_pil(rgb: np.ndarray) -> "PILImage":
    from PIL import Image as PILImage

    return PILImage.fromarray(rgb)


def _pil_to_b64_png(pil_img) -> str:
    """PIL.Image → png base64。"""
    buf = io.BytesIO()
    # PDF 出来的图本来就是 RGB，PNG 编码前确认模式
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_pdf_to_pil_pages(raw: bytes, dpi: float = 200.0) -> list:
    """PDF bytes → list[PIL.Image]，每页一张 RGB 图。in-memory，不落盘。"""
    import pymupdf  # 软依赖

    if not raw[:4] == b"%PDF":
        raise ValueError("不是合法 PDF 文件")

    zoom = dpi / 72.0  # PDF 坐标系 72 dpi
    mat = pymupdf.Matrix(zoom, zoom)
    pages: list = []
    pdf = pymupdf.open(stream=raw, filetype="pdf")
    try:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            # pix.samples 是 RGB 字节（alpha=False 时是 RGB 而不是 RGBA）
            # 直接构造 PIL Image 避免 numpy 中转
            from PIL import Image as PILImage

            img = PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(img)
    finally:
        pdf.close()
    return pages


# ─── 路由 ────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def index():
    """前端 UI"""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="0.4.0",
        pool_size=POOL_SIZE,
        pool_free=pool_ref.qsize() if pool_ref else 0,
        scheduler_pending=scheduler.pending_size() if scheduler else 0,
        batch_max=BATCH_MAX,
        active_users=len(scheduler._user_heartbeat) if scheduler else 0,
    )


# ─── 核心：把 N 个 PIL page 提交到 scheduler，等所有完成 ────────────
async def _process_pages(
        request_id: str,
        pil_pages: list,
) -> tuple[list, float]:
    """把一组 PIL 图全部提交给 BatchScheduler，等所有 page 完成后合并结果。

    返回 (list[PageResult], queue_wait_seconds)
    """
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler 尚未初始化")

    # 提交时主动 heartbeat 一次，避免 cleanup 误判
    scheduler.heartbeat(request_id)

    submit_t = time.perf_counter()
    # 一次性提交所有 page；BatchScheduler 会自动跨 page 凑 batch
    jobs: list[Job] = []
    for idx, pil_img in enumerate(pil_pages):
        # 必须确保 RGB，否则后面 layout detect 报 mode 错
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        job = scheduler.submit(request_id=request_id, page_index=idx, image=pil_img)
        jobs.append(job)

    # 等所有 page 全部完成（asyncio.gather 跨 N 个 Future）
    try:
        pages = await asyncio.gather(*(j.fut for j in jobs))
        queue_wait = time.perf_counter() - submit_t
        return list(pages), queue_wait
    except asyncio.CancelledError:
        # 上一层取消（关 tab / cleanup 协程 heartbeat timeout 释放 / server shutdown）
        # —— 把本地 jobs 也取消, 但不让 CancelledError 冒到 FastAPI（那会变 500）
        for j in jobs:
            if not j.fut.done():
                j.fut.cancel("cancelled by client/server")
        queue_wait = time.perf_counter() - submit_t
        logger.info("request cancelled: user=%s pages=%d", request_id, len(jobs))
        raise  # 仍然 raise 让上层处理（handler 会 catch 并返回 200+cancelled）


def _build_ocr_response_with_images(
        request_id: str,
        pil_pages: list,
        page_results: list,
        elapsed: float,
        queue_wait: float,
) -> OCRResponse:
    """组装 OCRResponse：每页含原图 b64 + 摊平所有 regions（带 page_index）。"""
    all_regions: list[OCRRegion] = []
    out_pages: list[OCRPage] = []
    for pil, pr in zip(pil_pages, page_results):
        b64 = _pil_to_b64_png(pil)
        out_pages.append(
            OCRPage(
                page_index=pr.page_index,
                width=pr.width,
                height=pr.height,
                image_b64=b64,
            )
        )
        for b_idx, r in enumerate(pr.regions):
            all_regions.append(
                OCRRegion(
                    page_index=pr.page_index,
                    box_index=b_idx,
                    label=r.label,
                    score=round(r.score, 4),
                    rect=r.rect,
                    md_path=str(r.md_path) if r.md_path else None,
                    markdown=r.markdown,
                )
            )
    return OCRResponse(
        success=True,
        request_id=request_id,
        status="done",
        elapsed_seconds=round(elapsed, 3),
        queue_wait_seconds=round(queue_wait, 3),
        scheduler_pending=scheduler.pending_size() if scheduler else 0,
        pool_size=POOL_SIZE,
        pool_free=pool_ref.qsize() if pool_ref else 0,
        pages=out_pages,
        regions=all_regions,
    )


# ─── 路由：单图 ──────────────────────────────────────────────────
@app.post("/ocr/upload", response_model=OCRResponse)
async def ocr_upload(request: Request, file: UploadFile = File(...)):
    _check_content_length(request, MAX_IMAGE_UPLOAD_MB)
    raw = await file.read()
    if len(raw) > MAX_IMAGE_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"图片超过 {MAX_IMAGE_UPLOAD_MB:.0f}MB 限制"
        )
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码上传的图片")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = _rgb_to_pil(rgb)

    # 前端心跳用的 user_id 优先, 没有再生成
    request_id = request.query_params.get("user_id") or uuid.uuid4().hex[:8]
    st = time.perf_counter()
    try:
        page_results, queue_wait = await _process_pages(request_id, [pil])
    except asyncio.CancelledError:
        # client disconnect / heartbeat timeout / server shutdown
        # —— 返回 200 + status=cancelled, 让前端知道是被主动取消的（不是 bug）
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "request_id": request_id,
                "status": "cancelled",
                "elapsed_seconds": round(time.perf_counter() - st, 3),
                "queue_wait_seconds": 0.0,
                "scheduler_pending": scheduler.pending_size() if scheduler else 0,
                "pool_size": POOL_SIZE,
                "pool_free": pool_ref.qsize() if pool_ref else 0,
                "pages": [],
                "regions": [],
            },
        )
    elapsed = time.perf_counter() - st
    return _build_ocr_response_with_images(
        request_id, [pil], page_results, elapsed, queue_wait
    )


@app.post("/ocr/base64", response_model=OCRResponse)
async def ocr_base64(req: Base64Request, request: Request):
    try:
        rgb = _decode_b64_to_rgb(req.payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    pil = _rgb_to_pil(rgb)

    request_id = request.query_params.get("user_id") or uuid.uuid4().hex[:8]
    st = time.perf_counter()
    try:
        page_results, queue_wait = await _process_pages(request_id, [pil])
    except asyncio.CancelledError:
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "request_id": request_id,
                "status": "cancelled",
                "elapsed_seconds": round(time.perf_counter() - st, 3),
                "queue_wait_seconds": 0.0,
                "scheduler_pending": scheduler.pending_size() if scheduler else 0,
                "pool_size": POOL_SIZE,
                "pool_free": pool_ref.qsize() if pool_ref else 0,
                "pages": [],
                "regions": [],
            },
        )
    elapsed = time.perf_counter() - st
    return _build_ocr_response_with_images(
        request_id, [pil], page_results, elapsed, queue_wait
    )


# ─── 路由：PDF 多页 ──────────────────────────────────────────────
@app.post("/ocr/pdf", response_model=OCRResponse)
async def ocr_pdf(request: Request, file: UploadFile = File(...)):
    """PDF → in-memory 转 N 张 PNG → 拆 N 个 Job 提交 BatchScheduler → 全部完成。

    返回 pages 列表（长度 = PDF 页数），regions 列表（所有 page 摊平，每条带 page_index）。
    """
    _check_content_length(request, MAX_PDF_UPLOAD_MB)
    raw = await file.read()
    if len(raw) > MAX_PDF_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"PDF 超过 {MAX_PDF_UPLOAD_MB:.0f}MB 限制"
        )
    if not raw[:4] == b"%PDF":
        raise HTTPException(status_code=400, detail="不是合法 PDF 文件")

    # PDF 渲染放线程池里，pymupdf 偶尔会卡
    try:
        pil_pages = await run_in_threadpool(_decode_pdf_to_pil_pages, raw, PDF_DPI)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("PDF decode failed")
        raise HTTPException(status_code=500, detail=f"PDF 解析失败: {e}")

    if not pil_pages:
        raise HTTPException(status_code=400, detail="PDF 没有页面")
    if len(pil_pages) > MAX_PDF_PAGES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF 页数 {len(pil_pages)} 超过 {MAX_PDF_PAGES} 页上限",
        )

    logger.info(
        "PDF decode: file=%s size=%d pages=%d (dpi=%.0f)",
        file.filename, len(raw), len(pil_pages), PDF_DPI,
    )

    request_id = request.query_params.get("user_id") or uuid.uuid4().hex[:8]
    st = time.perf_counter()
    try:
        page_results, queue_wait = await _process_pages(request_id, pil_pages)
    except asyncio.CancelledError:
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "request_id": request_id,
                "status": "cancelled",
                "elapsed_seconds": round(time.perf_counter() - st, 3),
                "queue_wait_seconds": 0.0,
                "scheduler_pending": scheduler.pending_size() if scheduler else 0,
                "pool_size": POOL_SIZE,
                "pool_free": pool_ref.qsize() if pool_ref else 0,
                "pages": [],
                "regions": [],
            },
        )
    elapsed = time.perf_counter() - st

    return _build_ocr_response_with_images(
        request_id, pil_pages, page_results, elapsed, queue_wait
    )


# ─── 路由：心跳 / 释放（前端用） ─────────────────────────────────
@app.post("/api/heartbeat", response_model=HeartbeatResponse)
async def api_heartbeat(req: HeartbeatRequest):
    """前端每 10s 发一次心跳，续约 user 活跃状态。

    后台 cleanup 协程每 5s 检查一次，超过 HEARTBEAT_TIMEOUT_S 没心跳的 user
    会自动释放（pending job 取消）。
    """
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler 尚未初始化")
    scheduler.heartbeat(req.user_id)
    return HeartbeatResponse(
        ok=True,
        pending=scheduler.pending_size(),
        server_time=time.time(),
    )


@app.post("/api/release", response_model=ReleaseResponse)
async def api_release(req: ReleaseRequest):
    """前端在 pagehide / beforeunload 时调用，把该 user 的所有 pending job 取消。"""
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler 尚未初始化")
    cancelled = scheduler.release_user(req.user_id, reason="client released")
    return ReleaseResponse(ok=True, cancelled=cancelled)


# ─── 启动入口 ────────────────────────────────────────────────────
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
