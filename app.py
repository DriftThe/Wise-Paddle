"""
Wise-Paddle uvicorn 服务（生产落地版：.env 驱动 + 显式 cleanup，零心跳）

设计要点：
- 单 pipeline 一次处理一批（最多 BATCH_MAX 张图）
- BatchScheduler 跨用户聚合 page 任务送给 pipeline
- layout detection 和 VL 推理都能 batch 起来
- 每个 request 唯一 user_id (request_id)；客户端通过 /api/release 显式释放
  （前端在 pagehide / beforeunload 时 sendBeacon 触发）
- 服务端在 HTTP handler 收到 CancelledError 时也调用 release_user() 兜底
  —— 替代了之前的心跳+cleanup_loop 机制

HTTP 接口：
- GET  /                          —— 前端 UI（单页 HTML）
- GET  /health                    —— 健康 + scheduler/pool 状态
- POST /ocr/upload                —— 上传图片（multipart），返回 1 个 OCRPage
- POST /ocr/base64                —— base64 JSON body，返回 1 个 OCRPage
- POST /ocr/pdf                   —— 上传 PDF，转 N 页 PNG 入队，返回 N 个 OCRPage
- GET  /api/ocr/pdf-status/{uid}  —— 轮询 PDF 进度
- POST /api/release               —— 主动释放 user 所有 pending，body: {"user_id": "..."}

所有可调参数从 .env 读取（见 .env.example），影响：
- 显存：PIPELINE_POOL_SIZE, BATCH_MAX, VL_MAX_PIXELS, VL_MIN_PIXELS, ATTN_IMPL, DTYPE
- 速度：BATCH_FLUSH_MS, VL_MAX_FORWARD_BATCH, MAX_NEW_TOKENS
- 精度：LAYOUT_SCORE_THRESHOLD, BOX_IOU_THRESHOLD, BOX_MIN_AREA, BOX_MIN_SCORE,
        BOX_UNCLIP_RATIO（关键，doclayout 边界偏紧时调大）, BOX_EXPAND_PIXELS,
        VL_REPETITION_PENALTY, VL_DO_SAMPLE
- 内存/CPU：OMP_NUM_THREADS（可放 .env）
- 资源保护：MAX_IMAGE_UPLOAD_MB, MAX_PDF_UPLOAD_MB, MAX_PDF_PAGES
- 日志：LOG_DIR, LOG_KEEP_DAYS
- 清理：PDF_PROGRESS_TTL_S, TEXT_RESULT_KEEP_DAYS
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
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

# ─── 加载 .env ────────────────────────────────────────────────────
# 放在所有 os.environ.get 之前；缺 .env 不报错（生产环境常用 systemd 注入 env）
load_dotenv(override=False)


# ─── 配置：所有可调参数集中此处 ───────────────────────────────────
def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v is not None and v != "" else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    return v.lower() in ("1", "true", "yes", "on", "y", "t")


@dataclass
class Config:
    # ---- HTTP server ----
    host: str = _env_str("HOST", "127.0.0.1")
    port: int = _env_int("PORT", 8000)
    cors_allow_origins: list[str] = None  # type: ignore  # set in __post_init__
    graceful_shutdown_s: int = _env_int("GRACEFUL_SHUTDOWN_S", 15)

    # ---- 模型路径 ----
    layout_model_path: str = _env_str("LAYOUT_MODEL_PATH", LAYOUT_MODEL_PATH)
    vl_model_path: str = _env_str("VL_MODEL_PATH", VL_MODEL_PATH)
    output_dir: Path = None  # type: ignore  # set in __post_init__

    # ---- 调度器 ----
    pool_size: int = _env_int("PIPELINE_POOL_SIZE", 2)
    batch_max: int = _env_int("BATCH_MAX", 5)
    batch_flush_ms: float = _env_float("BATCH_FLUSH_MS", 250.0)

    # ---- 精度（LayoutDetector）----
    layout_score_threshold: float = _env_float("LAYOUT_SCORE_THRESHOLD", 0.5)

    # ---- 精度（BoxFilter / unclip）----
    # doclayout 框偏紧时，把 BOX_UNCLIP_RATIO 调到 0.05~0.10，VL 能看到更多上下文，准确率明显提升
    box_iou_threshold: float = _env_float("BOX_IOU_THRESHOLD", 0.5)
    box_min_area: float = _env_float("BOX_MIN_BOX_AREA", 16 * 16)
    box_min_score: float = _env_float("BOX_MIN_SCORE", 0.5)
    box_unclip_ratio: float = _env_float("BOX_UNCLIP_RATIO", 0.05)
    box_expand_pixels: float = _env_float("BOX_EXPAND_PIXELS", 4.0)

    # ---- 精度（VL）----
    vl_min_pixels: int = _env_int("VL_MIN_PIXELS", 112896)         # shortest_edge
    vl_max_pixels: int = _env_int("VL_MAX_PIXELS", 1280 * 28 * 28)  # longest_edge (≈ 1MP)
    vl_repetition_penalty: float = _env_float("VL_REPETITION_PENALTY", 1.15)
    vl_do_sample: bool = _env_bool("VL_DO_SAMPLE", False)

    # ---- 速度/显存（VL）----
    vl_max_forward_batch: int = _env_int("VL_MAX_FORWARD_BATCH", 4)  # 单次 VL forward image 上限
    vl_max_new_tokens: int = _env_int("VL_MAX_NEW_TOKENS", 256)
    vl_attn_impl: str = _env_str("ATTN_IMPL", "sdpa")  # sdpa | eager
    vl_dtype: str = _env_str("DTYPE", "bfloat16")      # bfloat16 | float16 | float32

    # ---- 速度（pipeline）----
    max_regions: int = _env_int("MAX_REGIONS", 100)  # 单页最多送 VL 的 region 数

    # ---- 资源保护 ----
    max_image_upload_mb: float = _env_float("MAX_IMAGE_UPLOAD_MB", 32.0)
    max_pdf_upload_mb: float = _env_float("MAX_PDF_UPLOAD_MB", 128.0)
    max_pdf_pages: int = _env_int("MAX_PDF_PAGES", 200)
    pdf_dpi: float = _env_float("PDF_DPI", 200.0)

    # ---- 日志 ----
    log_dir: Path = None  # type: ignore  # set in __post_init__
    log_keep_days: int = _env_int("LOG_KEEP_DAYS", 14)
    log_level: str = _env_str("LOG_LEVEL", "INFO")  # DEBUG | INFO | WARNING

    # ---- 清理 ----
    pdf_progress_ttl_s: int = _env_int("PDF_PROGRESS_TTL_S", 600)  # _pdf_progress_store 过期时间
    text_result_keep_days: int = _env_int("TEXT_RESULT_KEEP_DAYS", 0)  # 0 = 不清
    cleanup_interval_s: int = _env_int("CLEANUP_INTERVAL_S", 300)  # 5min 扫一次

    def __post_init__(self):
        if self.cors_allow_origins is None:
            self.cors_allow_origins = _env_str(
                "CORS_ALLOW_ORIGINS", "http://localhost:*,http://127.0.0.1:*"
            ).split(",")
        if self.output_dir is None:
            self.output_dir = Path(_env_str("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
        if self.log_dir is None:
            self.log_dir = Path(_env_str("LOG_DIR", "./logs"))


CFG = Config()
STATIC_DIR = Path(__file__).parent / "static"


# ─── 日志（按日期写独立文件 logs/server-YYYY-MM-DD.log） ─────────
class _DailyFileHandler(logging.FileHandler):
    """按日期生成独立日志文件: logs/server-YYYY-MM-DD.log

    - 每天 0 点首次写入时自动切换到新文件
    - 不依赖 suffix 改名, 直接换 baseFilename
    - 旧文件永久保留, 由 _cleanup_old_logs 启动时按 LOG_KEEP_DAYS 删
    """
    def __init__(self, log_dir: Path, prefix: str = "server", encoding: str = "utf-8"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._prefix = prefix
        self._encoding = encoding
        self._current_date: str | None = None
        super().__init__(self._today_path(), mode="a", encoding=encoding, delay=False)
        self._current_date = time.strftime("%Y-%m-%d")

    def _today_path(self) -> str:
        return str(self._log_dir / f"{self._prefix}-{time.strftime('%Y-%m-%d')}.log")

    def emit(self, record: logging.LogRecord) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._current_date:
            self._current_date = today
            try:
                self.close()
            except Exception:
                pass
            self.baseFilename = self._today_path()
            try:
                self.stream = self._open()
            except Exception:
                return
        super().emit(record)


def _cleanup_old_logs(max_keep_days: int) -> int:
    """启动时清理 LOG_DIR 下超过 max_keep_days 天的 server-YYYY-MM-DD.log 文件。
    返回删除的文件数。"""
    if max_keep_days <= 0:
        return 0
    CFG.log_dir.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - max_keep_days * 86400
    pattern = re.compile(r"^server-(\d{4}-\d{2}-\d{2})\.log$")
    removed = 0
    for p in CFG.log_dir.glob("server-*.log"):
        m = pattern.match(p.name)
        if not m:
            continue
        try:
            file_date = time.strptime(m.group(1), "%Y-%m-%d")
            file_ts = time.mktime(file_date)
            if file_ts < cutoff:
                p.unlink(missing_ok=True)
                removed += 1
        except Exception:
            continue
    if removed:
        logger.info("CLEANUP: log cleanup removed %d old log file(s) (>%d days)",
                    removed, max_keep_days)
    return removed


def _setup_logging() -> None:
    """配置根 logger: 控制台 + 按日期独立文件。"""
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    formatter = logging.Formatter(fmt)
    level = getattr(logging, CFG.log_level.upper(), logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(level)

    fh = _DailyFileHandler(CFG.log_dir, prefix="server", encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(level)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(sh)
    root.addHandler(fh)
    root.setLevel(level)


_setup_logging()
logger = logging.getLogger("wise-paddle")


# ─── 全局服务对象（lifespan 里初始化） ────────────────────────────
scheduler: Optional[BatchScheduler] = None
pool_ref: Optional[PipelinePool] = None
_pdf_progress_store: dict[str, dict] = {}
_pdf_progress_lock = asyncio.Lock()
_pdf_cleanup_task: Optional[asyncio.Task] = None
_text_result_cleanup_task: Optional[asyncio.Task] = None
_app_version = "0.5.0"


# ─── 文本结果清理：定时删除过老的 text_result/<req_id>/ 子目录 ─────
async def _text_result_cleanup_loop() -> None:
    """每 CLEANUP_INTERVAL_S 扫一次, 删除超过 TEXT_RESULT_KEEP_DAYS 天的子目录。
    0 表示禁用。"""
    if CFG.text_result_keep_days <= 0:
        logger.info("text_result cleanup disabled (TEXT_RESULT_KEEP_DAYS=0)")
        return
    logger.info("text_result cleanup enabled: keep_days=%d interval=%ds",
                CFG.text_result_keep_days, CFG.cleanup_interval_s)
    while True:
        try:
            await asyncio.sleep(CFG.cleanup_interval_s)
        except asyncio.CancelledError:
            return
        try:
            removed = _cleanup_text_result_once()
            if removed:
                logger.info("CLEANUP: text_result removed %d stale request dir(s) (keep_days=%d)",
                            removed, CFG.text_result_keep_days)
        except Exception as e:
            logger.exception("text_result cleanup error: %s", e)


def _cleanup_text_result_once() -> int:
    if not CFG.output_dir.exists():
        return 0
    cutoff = time.time() - CFG.text_result_keep_days * 86400
    removed = 0
    for p in CFG.output_dir.iterdir():
        if not p.is_dir():
            continue
        try:
            mtime = p.stat().st_mtime
            if mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
                logger.info("CLEANUP: text_result removed dir=%s (mtime=%s, age=%.0fd)",
                            p.name, time.strftime("%Y-%m-%d", time.localtime(mtime)),
                            (time.time() - mtime) / 86400)
        except Exception as e:
            logger.warning("text_result cleanup failed for %s: %s", p.name, e)
    return removed


# ─── Pipeline 工厂 ──────────────────────────────────────────────
def _make_pipeline() -> OCRPipeline:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(CFG.vl_dtype.lower(), torch.bfloat16)

    logger.info(
        "Building pipeline: device=%s dtype=%s attn=%s unclip_ratio=%.3f "
        "vl_pixels=[%d,%d] max_fwd_batch=%d max_new_tokens=%d",
        device, CFG.vl_dtype, CFG.vl_attn_impl, CFG.box_unclip_ratio,
        CFG.vl_min_pixels, CFG.vl_max_pixels, CFG.vl_max_forward_batch, CFG.vl_max_new_tokens,
    )
    return OCRPipeline(
        layout_model_path=CFG.layout_model_path,
        vl_model_path=CFG.vl_model_path,
        device=device,
        output_dir=CFG.output_dir,
        # LayoutDetector
        score_threshold=CFG.layout_score_threshold,
        # BoxFilter
        box_iou_threshold=CFG.box_iou_threshold,
        box_min_area=CFG.box_min_area,
        box_min_score=CFG.box_min_score,
        box_unclip_ratio=CFG.box_unclip_ratio,
        box_expand_pixels=CFG.box_expand_pixels,
        # VLPredictor
        max_new_tokens=CFG.vl_max_new_tokens,
        vl_min_pixels=CFG.vl_min_pixels,
        vl_max_pixels=CFG.vl_max_pixels,
        vl_max_forward_batch=CFG.vl_max_forward_batch,
        vl_repetition_penalty=CFG.vl_repetition_penalty,
        vl_do_sample=CFG.vl_do_sample,
        # Pipeline
        max_regions=CFG.max_regions,
        dtype=dtype,
        attn_impl=CFG.vl_attn_impl,
    )


# ─── Lifespan ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler, pool_ref, _pdf_cleanup_task, _text_result_cleanup_task
    logger.info(
        "Wise-Paddle v%s starting: pool=%d batch_max=%d flush=%dms pdf_dpi=%.0f "
        "log_dir=%s log_keep=%dd",
        _app_version, CFG.pool_size, CFG.batch_max, int(CFG.batch_flush_ms),
        CFG.pdf_dpi, CFG.log_dir, CFG.log_keep_days,
    )
    # 启动时清一次旧 log
    _cleanup_old_logs(CFG.log_keep_days)

    pool = PipelinePool(size=CFG.pool_size, factory=_make_pipeline)
    await pool.init()
    pool_ref = pool

    sched = BatchScheduler(
        pool=pool,
        max_batch=CFG.batch_max,
        flush_ms=CFG.batch_flush_ms,
        n_workers=CFG.pool_size,
    )
    await sched.start()
    scheduler = sched

    # 后台守护：PDF 进度 / 文本结果 清理
    _pdf_cleanup_task = asyncio.create_task(_pdf_progress_cleanup_loop())
    _text_result_cleanup_task = asyncio.create_task(_text_result_cleanup_loop())

    logger.info("Service ready: pool=%d batch_max=%d", CFG.pool_size, CFG.batch_max)
    try:
        yield
    finally:
        logger.info("Shutting down ...")
        for t in (_pdf_cleanup_task, _text_result_cleanup_task):
            if t is not None:
                t.cancel()
        for t in (_pdf_cleanup_task, _text_result_cleanup_task):
            if t is not None:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        # 取消所有 PDF 后台跑批 task
        for uid, prog in list(_pdf_progress_store.items()):
            task = prog.get("task")
            if task is not None and not task.done():
                task.cancel()
        if scheduler is not None:
            await scheduler.close()
        if pool is not None:
            await pool.close()
        scheduler = None
        pool_ref = None
        _pdf_progress_store.clear()


app = FastAPI(title="Wise-Paddle Service", version=_app_version, lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CFG.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── 全局错误处理：未捕获的异常 → 500 + JSON ────────────────────
@app.exception_handler(Exception)
async def _unhandled_exc_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("UNHANDLED: %s %s", request.method, request.url.path)
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
    rect: tuple[int, int, int, int]
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
    status: str = "done"
    elapsed_seconds: float
    queue_wait_seconds: float
    scheduler_pending: int
    pool_size: int
    pool_free: int
    pages: list[OCRPage]
    regions: list[OCRRegion] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    version: str
    pool_size: int
    pool_free: int
    scheduler_pending: int
    batch_max: int
    active_users: int  # 仍有 pending job 的 user 数


class ReleaseRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)


class ReleaseResponse(BaseModel):
    ok: bool
    cancelled: int


# ─── 工具函数 ────────────────────────────────────────────────────
def _check_content_length(request: Request, max_mb: float) -> None:
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
    buf = io.BytesIO()
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_pdf_to_pil_pages(raw: bytes, dpi: float = 200.0) -> list:
    """PDF bytes → list[PIL.Image]，每页一张 RGB 图。in-memory，不落盘。"""
    import pymupdf
    if not raw[:4] == b"%PDF":
        raise ValueError("不是合法 PDF 文件")
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    pages: list = []
    pdf = pymupdf.open(stream=raw, filetype="pdf")
    try:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            from PIL import Image as PILImage
            img = PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(img)
    finally:
        pdf.close()
    return pages


def _release_user_silent(user_id: str, reason: str) -> int:
    """同步、安全地释放一个 user 的所有 pending + in-flight job。
    任何 handler 收到 CancelledError 时都该调这个。返回被取消的 pending 数。"""
    if scheduler is None:
        return 0
    try:
        return scheduler.release_user(user_id, reason=reason)
    except Exception as e:
        logger.exception("release_user failed for %s: %s", user_id, e)
        return 0


# ─── 核心：把 N 个 PIL page 提交到 scheduler，等所有完成 ────────────
async def _process_pages(
        request_id: str,
        pil_pages: list,
) -> tuple[list, float]:
    """把一组 PIL 图全部提交给 BatchScheduler，等所有 page 完成后合并结果。

    返回 (list[PageResult], queue_wait_seconds)
    CancelledError 兜底：释放该 user 的所有 pending，wakeup worker 重新拼批。
    """
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler 尚未初始化")

    submit_t = time.perf_counter()
    jobs: list[Job] = []
    for idx, pil_img in enumerate(pil_pages):
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        job = scheduler.submit(request_id=request_id, page_index=idx, image=pil_img)
        jobs.append(job)

    try:
        pages = await asyncio.gather(*(j.fut for j in jobs))
        queue_wait = time.perf_counter() - submit_t
        return list(pages), queue_wait
    except asyncio.CancelledError:
        # 关 tab / 网络断 / 服务关闭 都会让 CancelledError 冒上来
        # 关键：必须 release_user()，否则 fut.cancel() 只是本地层取消，
        # _user_pending 里的 Job 还在队列，worker 还是会去跑、浪费 GPU
        n = _release_user_silent(request_id, reason="cancelled by client/server")
        queue_wait = time.perf_counter() - submit_t
        logger.info("CANCEL: user=%s pages_pending_cancelled=%d queue_wait=%.2fs",
                    request_id, n, queue_wait)
        raise  # 让 handler 决定如何返回 200+cancelled


def _build_ocr_response_with_images(
        request_id: str,
        pil_pages: list,
        page_results: list,
        elapsed: float,
        queue_wait: float,
) -> OCRResponse:
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
        pool_size=CFG.pool_size,
        pool_free=pool_ref.qsize() if pool_ref else 0,
        pages=out_pages,
        regions=all_regions,
    )


# ─── 路由：根 / 健康 ──────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version=_app_version,
        pool_size=CFG.pool_size,
        pool_free=pool_ref.qsize() if pool_ref else 0,
        scheduler_pending=scheduler.pending_size() if scheduler else 0,
        batch_max=CFG.batch_max,
        active_users=scheduler.active_user_count() if scheduler else 0,
    )


# ─── 路由：单图 ──────────────────────────────────────────────────
@app.post("/ocr/upload", response_model=OCRResponse)
async def ocr_upload(request: Request, file: UploadFile = File(...)):
    _check_content_length(request, CFG.max_image_upload_mb)
    raw = await file.read()
    if len(raw) > CFG.max_image_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"图片超过 {CFG.max_image_upload_mb:.0f}MB 限制"
        )
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码上传的图片")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = _rgb_to_pil(rgb)

    request_id = request.query_params.get("user_id") or uuid.uuid4().hex[:8]
    logger.info("UPLOAD: user=%s endpoint=/ocr/upload file=%s size=%d bytes mime=%s",
                request_id, file.filename or "<unnamed>", len(raw), file.content_type or "?")
    st = time.perf_counter()
    try:
        page_results, queue_wait = await _process_pages(request_id, [pil])
    except asyncio.CancelledError:
        elapsed = round(time.perf_counter() - st, 3)
        logger.info("CANCEL: user=%s endpoint=/ocr/upload elapsed=%.2fs",
                    request_id, elapsed)
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "request_id": request_id,
                "status": "cancelled",
                "elapsed_seconds": elapsed,
                "queue_wait_seconds": 0.0,
                "scheduler_pending": scheduler.pending_size() if scheduler else 0,
                "pool_size": CFG.pool_size,
                "pool_free": pool_ref.qsize() if pool_ref else 0,
                "pages": [],
                "regions": [],
            },
        )
    elapsed = time.perf_counter() - st
    n_regions = sum(len(p.regions) for p in page_results)
    logger.info("COMPLETE: user=%s endpoint=/ocr/upload pages=1 regions=%d elapsed=%.2fs queue_wait=%.2fs",
                request_id, n_regions, elapsed, queue_wait)
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
    logger.info("UPLOAD: user=%s endpoint=/ocr/base64 size=%d bytes (decoded=%dx%d)",
                request_id, len(req.payload), rgb.shape[1], rgb.shape[0])
    st = time.perf_counter()
    try:
        page_results, queue_wait = await _process_pages(request_id, [pil])
    except asyncio.CancelledError:
        elapsed = round(time.perf_counter() - st, 3)
        logger.info("CANCEL: user=%s endpoint=/ocr/base64 elapsed=%.2fs",
                    request_id, elapsed)
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "request_id": request_id,
                "status": "cancelled",
                "elapsed_seconds": elapsed,
                "queue_wait_seconds": 0.0,
                "scheduler_pending": scheduler.pending_size() if scheduler else 0,
                "pool_size": CFG.pool_size,
                "pool_free": pool_ref.qsize() if pool_ref else 0,
                "pages": [],
                "regions": [],
            },
        )
    elapsed = time.perf_counter() - st
    n_regions = sum(len(p.regions) for p in page_results)
    logger.info("COMPLETE: user=%s endpoint=/ocr/base64 pages=1 regions=%d elapsed=%.2fs queue_wait=%.2fs",
                request_id, n_regions, elapsed, queue_wait)
    return _build_ocr_response_with_images(
        request_id, [pil], page_results, elapsed, queue_wait
    )


# ─── 路由：PDF 多页（流式 / 异步进度） ──────────────────────────────
async def _run_pdf_batch(user_id: str, pil_pages: list) -> None:
    """后台 task: 跑完 N 个 page, 每完成一页更新 _pdf_progress_store[user_id]"""
    progress = _pdf_progress_store[user_id]
    jobs: list[Job] = []
    try:
        for idx, pil_img in enumerate(pil_pages):
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            job = scheduler.submit(request_id=user_id, page_index=idx, image=pil_img)
            jobs.append(job)

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
                    if page_idx is None:
                        logger.error("PDF batch: fut not found (user=%s)", user_id)
                        continue
                    page_dict = _make_pdf_page_dict(page_result, pil_pages[page_idx])
                    progress["pages"][page_idx] = page_dict
                    progress["done_count"] = len(progress["pages"])
                except asyncio.CancelledError:
                    progress["cancelled"] = True
                    for j in jobs:
                        if not j.fut.done():
                            j.fut.cancel("cancelled")
                    logger.info("CANCEL: user=%s endpoint=/ocr/pdf pages_pending=%d",
                                user_id, len([j for j in jobs if not j.fut.done()]))
                    return
                except Exception as e:
                    logger.exception("PDF page failed: user=%s err=%s", user_id, e)
                    progress.setdefault("errors", []).append({"err": str(e)})
    except Exception as e:
        logger.exception("PDF batch crashed: user=%s", user_id)
        progress["error"] = str(e)
    finally:
        progress["done"] = True
        progress["finished_at"] = time.time()
        logger.info(
            "COMPLETE: user=%s endpoint=/ocr/pdf done=%d/%d wall=%.2fs cancelled=%s",
            user_id, progress["done_count"], progress["total_pages"],
            progress["finished_at"] - progress["started_at"],
            progress.get("cancelled", False),
        )


def _make_pdf_page_dict(page_result, pil_page) -> dict:
    return {
        "page_index": page_result.page_index,
        "width": page_result.width,
        "height": page_result.height,
        "image_b64": _pil_to_b64_png(pil_page),
        "regions": [
            {
                "page_index": page_result.page_index,
                "box_index": b_idx,
                "label": r.label,
                "score": round(r.score, 4),
                "rect": r.rect,
                "md_path": str(r.md_path) if r.md_path else None,
                "markdown": r.markdown,
            }
            for b_idx, r in enumerate(page_result.regions)
        ],
    }


async def _pdf_progress_cleanup_loop() -> None:
    """每 60s 扫一次, 删除 finished 超过 PDF_PROGRESS_TTL_S 的进度条目。"""
    logger.info("pdf_progress cleanup enabled: ttl=%ds", CFG.pdf_progress_ttl_s)
    while True:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            return
        now = time.time()
        expired: list[tuple[str, float]] = []
        for uid, p in list(_pdf_progress_store.items()):
            if p.get("done") and p.get("finished_at") and now - p["finished_at"] > CFG.pdf_progress_ttl_s:
                expired.append((uid, now - p["finished_at"]))
        for uid, age in expired:
            _pdf_progress_store.pop(uid, None)
            logger.info("CLEANUP: pdf_progress removed user=%s (age=%.0fs)", uid, age)


@app.post("/ocr/pdf")
async def ocr_pdf(request: Request, file: UploadFile = File(...)):
    """PDF → in-memory 转 N 页 PNG → 异步提交 BatchScheduler → 立即返回 user_id。

    长 PDF 不阻塞 HTTP 响应: 客户端拿到 user_id 后用 GET /api/ocr/pdf-status/{user_id}
    轮询进度, 每完成一页就拿到新数据。
    """
    _check_content_length(request, CFG.max_pdf_upload_mb)
    raw = await file.read()
    if len(raw) > CFG.max_pdf_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"PDF 超过 {CFG.max_pdf_upload_mb:.0f}MB 限制"
        )
    if not raw[:4] == b"%PDF":
        raise HTTPException(status_code=400, detail="不是合法 PDF 文件")

    try:
        pil_pages = await run_in_threadpool(_decode_pdf_to_pil_pages, raw, CFG.pdf_dpi)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("PDF decode failed")
        raise HTTPException(status_code=500, detail=f"PDF 解析失败: {e}")

    if not pil_pages:
        raise HTTPException(status_code=400, detail="PDF 没有页面")
    if len(pil_pages) > CFG.max_pdf_pages:
        raise HTTPException(
            status_code=413,
            detail=f"PDF 页数 {len(pil_pages)} 超过 {CFG.max_pdf_pages} 页上限",
        )

    user_id = request.query_params.get("user_id") or uuid.uuid4().hex[:8]
    logger.info("UPLOAD: user=%s endpoint=/ocr/pdf file=%s size=%d bytes pages=%d dpi=%.0f",
                user_id, file.filename or "<unnamed>", len(raw), len(pil_pages), CFG.pdf_dpi)

    async with _pdf_progress_lock:
        _pdf_progress_store[user_id] = {
            "total_pages": len(pil_pages),
            "pil_pages": pil_pages,
            "pages": {},
            "done": False,
            "done_count": 0,
            "started_at": time.time(),
            "finished_at": None,
            "error": None,
            "cancelled": False,
        }
        task = asyncio.create_task(_run_pdf_batch(user_id, pil_pages))
        _pdf_progress_store[user_id]["task"] = task

    return JSONResponse(
        status_code=202,
        content={
            "success": True,
            "request_id": user_id,
            "status": "processing",
            "total_pages": len(pil_pages),
            "poll_url": f"/api/ocr/pdf-status/{user_id}",
            "elapsed_seconds": 0,
            "queue_wait_seconds": 0.0,
            "scheduler_pending": scheduler.pending_size() if scheduler else 0,
            "pool_size": CFG.pool_size,
            "pool_free": pool_ref.qsize() if pool_ref else 0,
            "pages": [],
            "regions": [],
        },
    )


@app.get("/api/ocr/pdf-status/{user_id}")
async def pdf_status(user_id: str):
    progress = _pdf_progress_store.get(user_id)
    if not progress:
        return JSONResponse(
            status_code=200,
            content={
                "user_id": user_id,
                "total_pages": 0,
                "done": True,
                "done_count": 0,
                "error": "progress expired or unknown user_id",
                "pages": {},
            },
        )
    return JSONResponse(
        status_code=200,
        content={
            "user_id": user_id,
            "total_pages": progress["total_pages"],
            "done": progress["done"],
            "done_count": progress["done_count"],
            "cancelled": progress.get("cancelled", False),
            "error": progress.get("error"),
            "pages": dict(progress["pages"]),
        },
    )


# ─── 路由：释放（前端 pagehide 时 sendBeacon 调用） ─────────────
@app.post("/api/release", response_model=ReleaseResponse)
async def api_release(req: ReleaseRequest):
    """前端在 pagehide / beforeunload 时 sendBeacon 调用。

    一次性清掉该 user 的：
    - BatchScheduler 里所有 pending job（Fut 抛 CancelledError）
    - _pdf_progress_store 里对应的 PDF 后台 task
    - pdf_progress['cancelled'] 标记，让前端轮询能识别"已取消"
    """
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler 尚未初始化")
    cancelled = scheduler.release_user(req.user_id, reason="client released")
    progress = _pdf_progress_store.get(req.user_id)
    pdf_task_cancelled = False
    if progress:
        task = progress.get("task")
        if task is not None and not task.done():
            task.cancel()
            pdf_task_cancelled = True
        progress["cancelled"] = True
    logger.info("RELEASE: user=%s scheduler_cancelled=%d pdf_task_cancelled=%s",
                req.user_id, cancelled, pdf_task_cancelled)
    return ReleaseResponse(ok=True, cancelled=cancelled)


# ─── 启动入口 ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=CFG.host,
        port=CFG.port,
        reload=False,
        log_level=CFG.log_level.lower(),
        timeout_graceful_shutdown=CFG.graceful_shutdown_s,
    )
