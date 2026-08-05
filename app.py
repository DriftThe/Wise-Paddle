"""
Wise-Paddle uvicorn 服务（生产落地版：.env 驱动 + 显式 cleanup，零心跳）

设计要点：
- 单 pipeline 一次处理一批（最多 BATCH_MAX 张图）
- BatchScheduler 跨用户聚合 page 任务送给 pipeline（max-min fair）
- layout detection 和 VL 推理都能 batch 起来
- 每个 request 唯一 user_id (request_id)，用于 per-user 输出子目录隔离
- 用户在线/离线检测逻辑由调用方自行实现 —— 服务端不做自动取消

HTTP 接口：
- GET  /                          —— 前端 UI（单页 HTML）
- GET  /health                    —— 健康 + scheduler/pool 状态
- POST /ocr/upload                —— 上传图片（multipart），返回 1 个 OCRPage
- POST /ocr/base64                —— base64 JSON body，返回 1 个 OCRPage
- POST /ocr/pdf                   —— 上传 PDF，转 N 页 PNG 入队，返回 N 个 OCRPage
- GET  /api/ocr/pdf-status/{uid}  —— 轮询 PDF 进度
- POST /api/cancel/{user_id}      —— 主动取消某次 upload 的所有 pending + in-flight
- POST /alive                     —— 前端心跳，刷新 voucher 倒计时

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
from typing import Any, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage, ImageOps
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from core_pipeline import (
    DEFAULT_OUTPUT_DIR,
    LAYOUT_MODEL_PATH,
    VL_MODEL_PATH,
    BatchScheduler,
    Job,
    OCRPipeline,
    PipelinePool,
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
    """所有运行时可调参数。所有字段在 ``__post_init__`` 里完成最终类型化。"""

    # ---- HTTP server ----
    host: str = _env_str("HOST", "127.0.0.1")
    port: int = _env_int("PORT", 8000)
    cors_allow_origin_regex: str = _env_str(
        "CORS_ALLOW_ORIGIN_REGEX",
        r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    )
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
    vl_min_pixels: int = _env_int("VL_MIN_PIXELS", 112896)  # shortest_edge
    vl_max_pixels: int = _env_int("VL_MAX_PIXELS", 1280 * 28 * 28)  # longest_edge (≈ 1MP)
    vl_repetition_penalty: float = _env_float("VL_REPETITION_PENALTY", 1.15)
    vl_do_sample: bool = _env_bool("VL_DO_SAMPLE", False)

    # ---- 速度/显存（VL）----
    vl_max_forward_batch: int = _env_int("VL_MAX_FORWARD_BATCH", 4)  # 单次 VL forward image 上限
    vl_max_new_tokens: int = _env_int("VL_MAX_NEW_TOKENS", 256)
    vl_attn_impl: str = _env_str("ATTN_IMPL", "sdpa")  # sdpa | eager
    vl_dtype: str = _env_str("DTYPE", "bfloat16")  # bfloat16 | float16 | float32

    # ---- 速度（pipeline）----
    max_regions: int = _env_int("MAX_REGIONS", 100)  # 单页最多送 VL 的 region 数

    # ---- 资源保护 ----
    max_image_upload_mb: float = _env_float("MAX_IMAGE_UPLOAD_MB", 32.0)
    max_pdf_upload_mb: float = _env_float("MAX_PDF_UPLOAD_MB", 128.0)
    max_pdf_pages: int = _env_int("MAX_PDF_PAGES", 200)
    max_image_pixels: int = _env_int("MAX_IMAGE_PIXELS", 50_000_000)  # 解码后单图像素上限
    pdf_dpi: float = _env_float("PDF_DPI", 200.0)

    # ---- 日志 ----
    log_dir: Path = None  # type: ignore  # set in __post_init__
    log_keep_days: int = _env_int("LOG_KEEP_DAYS", 14)
    log_level: str = _env_str("LOG_LEVEL", "INFO")  # DEBUG | INFO | WARNING

    # ---- 清理 ----
    pdf_progress_ttl_s: int = _env_int("PDF_PROGRESS_TTL_S", 600)  # _pdf_progress_store 过期时间
    text_result_keep_days: int = _env_int("TEXT_RESULT_KEEP_DAYS", 0)  # 0 = 不清
    cleanup_interval_s: int = _env_int("CLEANUP_INTERVAL_S", 300)  # 5min 扫一次

    # ---- 心跳 ----
    alive_tick_seconds: float = _env_float("ALIVE_TICK_SECONDS", 2.0)  # KickDeadUser 扫描周期
    alive_initial_ttl: int = _env_int("ALIVE_INITIAL_TTL", 120)  # 每次 /alive 把 TTL 重置到这个值（秒数=ttl×tick）

    def __post_init__(self) -> None:
        if self.output_dir is None:
            self.output_dir = Path(_env_str("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
        if self.log_dir is None:
            self.log_dir = Path(_env_str("LOG_DIR", "./logs"))


CFG = Config()
STATIC_DIR = Path(__file__).parent / "static"


# ─── 日志（按日期写独立文件 logs/server-YYYY-MM-DD.log） ─────────
class _DailyFileHandler(logging.FileHandler):
    """按日期生成独立日志文件: ``logs/server-YYYY-MM-DD.log``

    - 每天 0 点首次写入时自动切换到新文件
    - 不依赖 suffix 改名，直接换 ``baseFilename``
    - 旧文件永久保留，由 ``_cleanup_old_logs`` 启动时按 ``LOG_KEEP_DAYS`` 删
    """

    def __init__(self, log_dir: Path, prefix: str = "server", encoding: str = "utf-8"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._prefix = prefix
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

    返回删除的文件数。
    """
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
        logger.info(
            "CLEANUP: log cleanup removed %d old log file(s) (>%d days)",
            removed, max_keep_days,
        )
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
_alive_check_task: Optional[asyncio.Task] = None
_app_version = "0.6.0"

# ─── 心跳：aliveUsers 由 _alive_lock 保护 ─────────────────────────
# voucher -> 剩余 tick 数。每次 /alive 把 TTL 重置到 CFG.alive_initial_ttl；
# KickDeadUser 每秒 -1，归零时调 scheduler.cancel_voucher 清掉整批。
_alive_users: dict[str, int] = {}
_alive_lock = asyncio.Lock()


# ─── 文本结果清理：定时删除过老的 text_result/<req_id>/ 子目录 ─────
async def _text_result_cleanup_loop() -> None:
    """每 CLEANUP_INTERVAL_S 扫一次, 删除超过 TEXT_RESULT_KEEP_DAYS 天的子目录。

    0 表示禁用。
    """
    if CFG.text_result_keep_days <= 0:
        logger.info("text_result cleanup disabled (TEXT_RESULT_KEEP_DAYS=0)")
        return
    logger.info(
        "text_result cleanup enabled: keep_days=%d interval=%ds",
        CFG.text_result_keep_days, CFG.cleanup_interval_s,
    )
    while True:
        try:
            await asyncio.sleep(CFG.cleanup_interval_s)
        except asyncio.CancelledError:
            return
        try:
            removed = _cleanup_text_result_once()
            if removed:
                logger.info(
                    "CLEANUP: text_result removed %d stale request dir(s) (keep_days=%d)",
                    removed, CFG.text_result_keep_days,
                )
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
                logger.info(
                    "CLEANUP: text_result removed dir=%s (mtime=%s, age=%.0fd)",
                    p.name, time.strftime("%Y-%m-%d", time.localtime(mtime)),
                    (time.time() - mtime) / 86400,
                )
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
    global scheduler, pool_ref, _pdf_cleanup_task, _text_result_cleanup_task, _alive_check_task
    logger.info(
        "Wise-Paddle v%s starting: pool=%d batch_max=%d flush=%dms pdf_dpi=%.0f "
        "log_dir=%s log_keep=%dd",
        _app_version, CFG.pool_size, CFG.batch_max, int(CFG.batch_flush_ms),
        CFG.pdf_dpi, CFG.log_dir, CFG.log_keep_days,
    )
    # 启动时清一次旧 log
    _cleanup_old_logs(CFG.log_keep_days)
    _alive_check_task = asyncio.create_task(_kick_dead_user_loop())
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
        # 先停心跳，避免 shutdown 期间还在 cancel_voucher
        if _alive_check_task is not None:
            _alive_check_task.cancel()
        for t in (_pdf_cleanup_task, _text_result_cleanup_task):
            if t is not None:
                t.cancel()
        for t in (_alive_check_task, _pdf_cleanup_task, _text_result_cleanup_task):
            if t is not None:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        _alive_check_task = None
        _pdf_cleanup_task = None
        _text_result_cleanup_task = None
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
    allow_origin_regex=CFG.cors_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 静态资源挂载在文件末尾 — 必须放在所有 @app.get/post 之后，
# 否则 mount 会优先匹配并吞掉 API 路由。


# ─── 全局错误处理：未捕获的异常 → 500 + JSON ────────────────────
@app.exception_handler(Exception)
async def _unhandled_exc_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("UNHANDLED: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error"},
    )


# ─── Request / Response 模型 ─────────────────────────────────────
class Base64Request(BaseModel):
    payload: str = Field(..., description="图片 base64 字符串，可带 data URI 前缀")


class AliveRequest(BaseModel):
    aliveVoucher: str


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
    # 前端 alive_check voucher；空表示未绑定
    voucher_id: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str
    pool_size: int
    pool_free: int
    scheduler_pending: int
    batch_max: int
    alive_voucher: str


# ─── 工具函数 ────────────────────────────────────────────────────
def _check_content_length(request: Request, max_mb: float) -> None:
    """Reject request early when Content-Length header exceeds ``max_mb`` MB."""
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


_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _sanitize_user_id(value: Optional[str]) -> str:
    """校验 / 生成用户侧 request_id，拒绝路径穿越等非法字符（H-01 / M-04）。"""
    if value is None or value == "":
        return uuid.uuid4().hex[:8]
    if not _USER_ID_RE.fullmatch(value):
        raise HTTPException(
            status_code=400,
            detail="user_id 只允许字母 / 数字 / 下划线 / 连字符（1-64 字符）",
        )
    return value


def _safe_filename(name: Optional[str]) -> str:
    """日志安全：转义文件名中的控制字符，防日志注入（L-05）。"""
    name = name or "<unnamed>"
    return name.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _check_pixel_budget(width: int, height: int) -> None:
    """解码后像素预算检查，防解压炸弹（H-03）。"""
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="图片尺寸无效")
    if width * height > CFG.max_image_pixels:
        raise HTTPException(
            status_code=413,
            detail=f"图片像素数超过 {CFG.max_image_pixels // 1_000_000}MP 限制",
        )


def _strip_data_uri(payload: str) -> str:
    """Strip the ``data:...;base64,`` prefix if present, return the raw b64 body."""
    if not payload.startswith("data:"):
        return payload
    comma_idx = payload.find(",")
    if comma_idx == -1:
        raise ValueError("Invalid data URI: missing comma")
    return payload[comma_idx + 1:]


def _decode_raw_image(raw: bytes) -> np.ndarray:
    """把图片字节解码为 RGB ndarray，优先走 PIL 以处理 EXIF 方向（L-13），失败回退 cv2。"""
    try:
        with PILImage.open(io.BytesIO(raw)) as pil_img:
            transposed = ImageOps.exif_transpose(pil_img)
            transposed.load()
        return np.asarray(transposed.convert("RGB"))
    except Exception:
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解码图片")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _decode_b64_to_rgb(payload: str) -> np.ndarray:
    """Decode a base64 image payload (optionally data-URI prefixed) to RGB ndarray."""
    b64 = _strip_data_uri(payload)
    # base64 文本长度上限 ≈ 原始字节上限 × 4/3，额外留一点余量
    if len(b64) > CFG.max_image_upload_mb * 1.5 * 1024 * 1024:
        raise ValueError(f"base64 载荷超过 {CFG.max_image_upload_mb:.0f}MB 限制")
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as e:
        raise ValueError("base64 解码失败") from e
    rgb = _decode_raw_image(raw)
    _check_pixel_budget(rgb.shape[1], rgb.shape[0])
    return rgb


def _rgb_to_pil(rgb: np.ndarray) -> PILImage.Image:
    return PILImage.fromarray(rgb)


def _pil_to_b64_png(pil_img: PILImage.Image) -> str:
    """Encode a PIL image to a base64 PNG string (no data-URI prefix)."""
    buf = io.BytesIO()
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _count_pdf_pages(raw: bytes) -> int:
    """只打开 PDF 元数据，返回页数；加密 PDF 抛 ValueError（H-02 / L-12）。"""
    import pymupdf

    if raw[:4] != b"%PDF":
        raise ValueError("不是合法 PDF 文件")
    try:
        pdf = pymupdf.open(stream=raw, filetype="pdf")
    except Exception as e:
        raise ValueError("无法解析 PDF 文件") from e
    try:
        if pdf.needs_pass:
            raise ValueError("PDF 已加密，需要密码才能打开")
        return len(pdf)
    finally:
        pdf.close()


def _decode_pdf_to_pil_pages(
        raw: bytes, dpi: float = 200.0, max_pixels: int | None = None,
) -> list[PILImage.Image]:
    """PDF bytes → list[PIL.Image]，每页一张 RGB 图。in-memory，不落盘。"""
    import pymupdf

    if raw[:4] != b"%PDF":
        raise ValueError("不是合法 PDF 文件")
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    pages: list[PILImage.Image] = []
    pdf = pymupdf.open(stream=raw, filetype="pdf")
    try:
        if pdf.needs_pass:
            raise ValueError("PDF 已加密，需要密码才能打开")
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            if max_pixels is not None and pix.width * pix.height > max_pixels:
                raise ValueError(
                    f"PDF 第 {page_num + 1} 页像素数超过上限"
                )
            img = PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(img)
    finally:
        pdf.close()
    return pages


def _cancelled_response(
        request_id: str,
        voucher_id: str,
        elapsed: float,
        queue_wait: float = 0.0,
) -> JSONResponse:
    """构造统一的 200 cancelled 响应（前端据 status=cancelled 切 UI 状态）。"""
    return JSONResponse(
        status_code=200,
        content={
            "success": False,
            "request_id": request_id,
            "voucher_id": voucher_id,
            "status": "cancelled",
            "elapsed_seconds": round(elapsed, 3),
            "queue_wait_seconds": round(queue_wait, 3),
            "scheduler_pending": scheduler.pending_size() if scheduler else 0,
            "pool_size": CFG.pool_size,
            "pool_free": pool_ref.qsize() if pool_ref else 0,
            "pages": [],
            "regions": [],
        },
    )


# ─── 核心：把 N 个 PIL page 提交到 scheduler，等所有完成 ────────────
async def _process_pages(
        request_id: str,
        pil_pages: list[PILImage.Image],
        voucher_id: str = "",
) -> tuple[list, float]:
    """把一组 PIL 图全部提交给 BatchScheduler，等所有 page 完成后合并结果。

    返回 ``(list[PageResult], queue_wait_seconds)``。

    CancelledError（来自 uvicorn 关 client 连接 / 服务 shutdown）时只 log + 重新抛出，
    pending job 不自动取消 —— 由调用方决定是否要外部清掉（自己实现离线检测逻辑）。
    voucher_id 透传到 Job；前端的 alive_check 倒计时把 voucher 剔除时，
    KickDeadUser 会调 ``scheduler.cancel_voucher(voucher_id)`` 把 pending + in-flight
    都干掉。
    """
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler 尚未初始化")

    submit_t = time.perf_counter()
    jobs: list[Job] = []
    for idx, pil_img in enumerate(pil_pages):
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        job = scheduler.submit(
            request_id=request_id, page_index=idx, image=pil_img,
            voucher_id=voucher_id,
        )
        jobs.append(job)

    try:
        pages = await asyncio.gather(*(j.fut for j in jobs))
        queue_wait = time.perf_counter() - submit_t
        return list(pages), queue_wait
    except asyncio.CancelledError:
        # CancelledError 不再做自动释放；调用方可在 handler 层加自己的策略。
        # 注意：已经 submit 到 scheduler 但还没被 worker 拿走的 job 不会被取消，
        # 会继续被处理 —— 调 offline 检测的代码自行负责清理。
        queue_wait = time.perf_counter() - submit_t
        logger.info(
            "CANCEL: user=%s voucher=%s queue_wait=%.2fs (no auto-release)",
            request_id, voucher_id, queue_wait,
        )
        raise  # 让 handler 决定如何返回 200+cancelled


async def _build_ocr_response_with_images(
        request_id: str,
        pil_pages: list[PILImage.Image],
        page_results: list,
        elapsed: float,
        queue_wait: float,
        voucher_id: str = "",
) -> OCRResponse:
    all_regions: list[OCRRegion] = []
    out_pages: list[OCRPage] = []
    for pil, pr in zip(pil_pages, page_results):
        # PNG 编码是 CPU 密集操作，放线程池避免阻塞事件循环（M-02）
        b64 = await run_in_threadpool(_pil_to_b64_png, pil)
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
        voucher_id=voucher_id,
    )


# ─── 路由：健康 ───────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version=_app_version,
        pool_size=CFG.pool_size,
        pool_free=pool_ref.qsize() if pool_ref else 0,
        scheduler_pending=scheduler.pending_size() if scheduler else 0,
        batch_max=CFG.batch_max,
        alive_voucher=uuid.uuid4().hex[:8],
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
    try:
        rgb = _decode_raw_image(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _check_pixel_budget(rgb.shape[1], rgb.shape[0])
    pil = _rgb_to_pil(rgb)

    request_id = _sanitize_user_id(request.query_params.get("user_id"))
    voucher_id = request.query_params.get("voucher_id") or ""
    logger.info(
        "UPLOAD: user=%s voucher=%s endpoint=/ocr/upload file=%s size=%d bytes mime=%s",
        request_id, voucher_id, _safe_filename(file.filename), len(raw), file.content_type or "?",
    )
    st = time.perf_counter()
    try:
        page_results, queue_wait = await _process_pages(request_id, [pil], voucher_id=voucher_id)
    except asyncio.CancelledError:
        elapsed = round(time.perf_counter() - st, 3)
        logger.info(
            "CANCEL: user=%s voucher=%s endpoint=/ocr/upload elapsed=%.2fs",
            request_id, voucher_id, elapsed,
        )
        return _cancelled_response(request_id, voucher_id, elapsed)
    # 同步响应场景：voucher 已被 _kick_dead_user_loop 剔除 → process_batch 给了 cancelled=True 空 PageResult
    if page_results and page_results[0].cancelled:
        elapsed = round(time.perf_counter() - st, 3)
        logger.info(
            "CANCEL_VOUCHER: user=%s voucher=%s endpoint=/ocr/upload elapsed=%.2fs",
            request_id, voucher_id, elapsed,
        )
        return _cancelled_response(request_id, voucher_id, elapsed, queue_wait)
    elapsed = time.perf_counter() - st
    n_regions = sum(len(p.regions) for p in page_results)
    logger.info(
        "COMPLETE: user=%s voucher=%s endpoint=/ocr/upload pages=1 regions=%d elapsed=%.2fs queue_wait=%.2fs",
        request_id, voucher_id, n_regions, elapsed, queue_wait,
    )
    return await _build_ocr_response_with_images(
        request_id, [pil], page_results, elapsed, queue_wait,
        voucher_id=voucher_id,
    )


@app.post("/ocr/base64", response_model=OCRResponse)
async def ocr_base64(req: Base64Request, request: Request):
    try:
        rgb = _decode_b64_to_rgb(req.payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    pil = _rgb_to_pil(rgb)

    request_id = _sanitize_user_id(request.query_params.get("user_id"))
    voucher_id = request.query_params.get("voucher_id") or ""
    logger.info(
        "UPLOAD: user=%s voucher=%s endpoint=/ocr/base64 size=%d bytes (decoded=%dx%d)",
        request_id, voucher_id, len(req.payload), rgb.shape[1], rgb.shape[0],
    )
    st = time.perf_counter()
    try:
        page_results, queue_wait = await _process_pages(request_id, [pil], voucher_id=voucher_id)
    except asyncio.CancelledError:
        elapsed = round(time.perf_counter() - st, 3)
        logger.info(
            "CANCEL: user=%s voucher=%s endpoint=/ocr/base64 elapsed=%.2fs",
            request_id, voucher_id, elapsed,
        )
        return _cancelled_response(request_id, voucher_id, elapsed)
    if page_results and page_results[0].cancelled:
        elapsed = round(time.perf_counter() - st, 3)
        logger.info(
            "CANCEL_VOUCHER: user=%s voucher=%s endpoint=/ocr/base64 elapsed=%.2fs",
            request_id, voucher_id, elapsed,
        )
        return _cancelled_response(request_id, voucher_id, elapsed, queue_wait)
    elapsed = time.perf_counter() - st
    n_regions = sum(len(p.regions) for p in page_results)
    logger.info(
        "COMPLETE: user=%s voucher=%s endpoint=/ocr/base64 pages=1 regions=%d elapsed=%.2fs queue_wait=%.2fs",
        request_id, voucher_id, n_regions, elapsed, queue_wait,
    )
    return await _build_ocr_response_with_images(
        request_id, [pil], page_results, elapsed, queue_wait,
        voucher_id=voucher_id,
    )


# ─── 路由：PDF 多页（流式 / 异步进度） ──────────────────────────────
async def _run_pdf_batch(
        user_id: str,
        pil_pages: list[PILImage.Image],
        voucher_id: str = "",
) -> None:
    """后台 task: 跑完 N 个 page, 每完成一页更新 ``_pdf_progress_store[user_id]``。

    voucher_id 透传到每个 Job；``_kick_dead_user_loop`` 调
    ``scheduler.cancel_voucher`` 后，pending 的 job fut 会被 cancel，in-flight 的
    会在 process_batch 里被过滤掉，page_result 标 cancelled=True 让 progress
    反映这个状态。
    """
    progress = _pdf_progress_store[user_id]
    jobs: list[Job] = []
    try:
        for idx, pil_img in enumerate(pil_pages):
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            job = scheduler.submit(
                request_id=user_id, page_index=idx, image=pil_img,
                voucher_id=voucher_id,
            )
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
                    if page_result.cancelled:
                        # voucher 取消触发的空 PageResult —— 标记 progress 整批 cancelled
                        progress["cancelled"] = True
                        logger.info(
                            "CANCEL_VOUCHER: user=%s voucher=%s endpoint=/ocr/pdf page=%d cancelled",
                            user_id, voucher_id, page_idx,
                        )
                        # 继续等剩余的（可能也有 cancelled），不 break
                        continue
                    page_dict = await _make_pdf_page_dict(page_result, pil_pages[page_idx])
                    progress["pages"][page_idx] = page_dict
                    progress["done_count"] = len(progress["pages"])
                    # 编码完成后释放该页原图，避免进度 store 内 PIL + base64 双重持有（M-13）
                    progress["pil_pages"][page_idx] = None
                except asyncio.CancelledError:
                    progress["cancelled"] = True
                    for j in jobs:
                        if not j.fut.done():
                            j.fut.cancel("cancelled")
                    logger.info(
                        "CANCEL: user=%s voucher=%s endpoint=/ocr/pdf pages_pending=%d",
                        user_id, voucher_id, len([j for j in jobs if not j.fut.done()]),
                    )
                    return
                except Exception as e:
                    logger.exception("PDF page failed: user=%s err=%s", user_id, e)
                    progress.setdefault("errors", []).append({"err": str(e)})
    except Exception as e:
        logger.exception("PDF batch crashed: user=%s", user_id)
        progress["error"] = str(e)
    except asyncio.CancelledError:
        # 整个 _run_pdf_batch task 自身被 cancel（来自 /api/cancel/{user_id}）
        # —— 标记 progress 为 cancelled，并把还没 done 的 job fut 也 cancel 掉，
        # 让 worker 的 set_result 跳过它们
        progress["cancelled"] = True
        for j in jobs:
            if not j.fut.done():
                j.fut.cancel("request cancelled")
        logger.info(
            "CANCEL: user=%s voucher=%s endpoint=/ocr/pdf pages_pending=%d (task cancelled)",
            user_id, voucher_id, len([j for j in jobs if not j.fut.done()]),
        )
        raise  # 让 framework 看到 cancellation
    finally:
        progress["done"] = True
        progress["finished_at"] = time.time()
        # 释放 pil_pages 内存 —— 完成后前端不再需要原图（结果里有 image_b64）
        # 保留 pil_pages 直到所有 page 都处理完，避免 page_dict 拿不到原图
        # 这里可以安全释放了
        progress["pil_pages"] = []
        # 如果所有 page 都 cancelled（process_batch 给的 cancelled=True），整批算 cancelled
        if progress["done_count"] == 0 and not progress.get("cancelled"):
            # 还没设过 cancelled，但所有 page 都是空的 —— 几乎不会发生（除非 layout 完全没识别到任何 box）
            # 不强制覆盖，留给"成功但无结果"的语义
            pass
        logger.info(
            "COMPLETE: user=%s voucher=%s endpoint=/ocr/pdf done=%d/%d wall=%.2fs cancelled=%s",
            user_id, voucher_id, progress["done_count"], progress["total_pages"],
            progress["finished_at"] - progress["started_at"],
            progress.get("cancelled", False),
        )


async def _make_pdf_page_dict(page_result: Any, pil_page: PILImage.Image) -> dict:
    """Build the JSON-serializable page dict returned by the PDF status endpoint."""
    image_b64 = await run_in_threadpool(_pil_to_b64_png, pil_page)
    return {
        "page_index": page_result.page_index,
        "width": page_result.width,
        "height": page_result.height,
        "image_b64": image_b64,
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

    长 PDF 不阻塞 HTTP 响应: 客户端拿到 user_id 后用
    ``GET /api/ocr/pdf-status/{user_id}`` 轮询进度, 每完成一页就拿到新数据。
    """
    _check_content_length(request, CFG.max_pdf_upload_mb)
    raw = await file.read()
    if len(raw) > CFG.max_pdf_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"PDF 超过 {CFG.max_pdf_upload_mb:.0f}MB 限制"
        )
    if raw[:4] != b"%PDF":
        raise HTTPException(status_code=400, detail="不是合法 PDF 文件")

    # 先读元数据（页数 / 加密状态），再决定是否完整光栅化（H-02 / L-12）
    try:
        page_count = await run_in_threadpool(_count_pdf_pages, raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("PDF open failed")
        raise HTTPException(status_code=500, detail="PDF 解析失败")

    if page_count <= 0:
        raise HTTPException(status_code=400, detail="PDF 没有页面")
    if page_count > CFG.max_pdf_pages:
        raise HTTPException(
            status_code=413,
            detail=f"PDF 页数 {page_count} 超过 {CFG.max_pdf_pages} 页上限",
        )

    try:
        pil_pages = await run_in_threadpool(
            _decode_pdf_to_pil_pages, raw, CFG.pdf_dpi, CFG.max_image_pixels,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("PDF decode failed")
        raise HTTPException(status_code=500, detail="PDF 解析失败")

    if not pil_pages:
        raise HTTPException(status_code=400, detail="PDF 没有页面")

    user_id = _sanitize_user_id(request.query_params.get("user_id"))
    voucher_id = request.query_params.get("voucher_id") or ""
    logger.info(
        "UPLOAD: user=%s voucher=%s endpoint=/ocr/pdf file=%s size=%d bytes pages=%d dpi=%.0f",
        user_id, voucher_id, _safe_filename(file.filename), len(raw), len(pil_pages), CFG.pdf_dpi,
    )

    async with _pdf_progress_lock:
        _pdf_progress_store[user_id] = {
            "voucher_id": voucher_id,
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
        task = asyncio.create_task(_run_pdf_batch(user_id, pil_pages, voucher_id))
        _pdf_progress_store[user_id]["task"] = task

    return JSONResponse(
        status_code=202,
        content={
            "success": True,
            "request_id": user_id,
            "voucher_id": voucher_id,
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
async def pdf_status(user_id: str, since: int = Query(default=0, ge=0)):
    user_id = _sanitize_user_id(user_id)
    progress = _pdf_progress_store.get(user_id)
    if not progress:
        return JSONResponse(
            status_code=200,
            content={
                "user_id": user_id,
                "voucher_id": "",
                "total_pages": 0,
                "done": True,
                "done_count": 0,
                "error": "progress expired or unknown user_id",
                "pages": {},
            },
        )
    pages = dict(progress["pages"])
    if since > 0:
        # 增量返回：只回传 >= since 的已完成页，降低轮询带宽（M-12）
        pages = {k: v for k, v in pages.items() if int(k) >= since}
    return JSONResponse(
        status_code=200,
        content={
            "user_id": user_id,
            "voucher_id": progress.get("voucher_id", ""),
            "total_pages": progress["total_pages"],
            "done": progress["done"],
            "done_count": progress["done_count"],
            "cancelled": progress.get("cancelled", False),
            "error": progress.get("error"),
            "pages": pages,
        },
    )


# ─── 主动取消：前端点 remove 按钮时发起的 cancel ─────────────────
@app.post("/api/cancel/{user_id}")
async def cancel_user_request(user_id: str):
    """主动取消某个 upload（user_id）的所有 pending + in-flight 工作。

    跟 /alive 倒计时触发的 voucher 取消是两套独立机制：

    - voucher 取消 → ``_kick_dead_user_loop`` → ``cancel_voucher(voucher_id)`` → 清整个 session
    - request 取消 → 本端点 → ``cancel_request(user_id)`` → 只清这一条 upload

    行为：

    1. ``scheduler.cancel_request(user_id)`` 取消该 request_id 的 pending job fut，
       并把 request_id 加进 ``_cancelled_requests``，worker 下一批 process_batch
       会在 entry + crop 之后两层过滤掉它
    2. 如果是 PDF（``_pdf_progress_store`` 里有这条），``task.cancel()`` 取消后台
       task，并把 ``progress["cancelled"] = True``，前端下次 poll 立刻看到
    3. 单图同步响应路径：handler 仍在 ``await asyncio.gather``，``fut.cancel()`` 后
       gather 会抛 CancelledError，handler 返回 200 cancelled
    """
    user_id = _sanitize_user_id(user_id)
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler 尚未初始化")

    cancelled = await scheduler.cancel_request(user_id)

    pdf_cancelled = False
    pdf_had_progress = False
    if user_id in _pdf_progress_store:
        pdf_had_progress = True
        prog = _pdf_progress_store[user_id]
        task = prog.get("task")
        if task is not None and not task.done():
            task.cancel()
            pdf_cancelled = True
        # 即使 task 还没起来 / 已经结束，也置 cancelled，让 poll 立刻感知
        prog["cancelled"] = True

    logger.info(
        "CANCEL_REQ: user=%s cancelled_pending=%d pdf_had=%s pdf_cancelled=%s",
        user_id, cancelled, pdf_had_progress, pdf_cancelled,
    )
    return {
        "success": True,
        "user_id": user_id,
        "cancelled_pending": cancelled,
        "pdf_had_progress": pdf_had_progress,
        "pdf_cancelled": pdf_cancelled,
    }


# ──────路由: 存活检查────────────────────────────────
@app.post("/alive")
async def alive_user_pend(request: AliveRequest):
    """前端心跳：刷新 voucher 倒计时到 ``CFG.alive_initial_ttl``。

    首次出现的 voucher 视为新登录；已存在的 voucher 重置 TTL 即可（两条分支
    语义等价，统一处理）。
    """
    voucher = request.aliveVoucher
    if not voucher:
        raise HTTPException(status_code=400, detail="voucher 不能为空")
    async with _alive_lock:
        is_new = voucher not in _alive_users
        _alive_users[voucher] = CFG.alive_initial_ttl
    if is_new:
        logger.info("User '%s' Logged in", voucher)
    return {"success": True, "voucher": voucher, "ttl": CFG.alive_initial_ttl}


async def _kick_dead_user_loop() -> None:
    """每 ``CFG.alive_tick_seconds`` 扫一次 alive_users，TTL 减 1。

    归零时调 ``scheduler.cancel_voucher(voucher_id)`` 清掉整批 pending + in-flight
    工作，并从 ``_alive_users`` 删除条目。
    """
    logger.info(
        "alive_check started: tick=%.1fs initial_ttl=%d",
        CFG.alive_tick_seconds, CFG.alive_initial_ttl,
    )
    while True:
        try:
            await asyncio.sleep(CFG.alive_tick_seconds)
        except asyncio.CancelledError:
            logger.info("alive_check exited")
            return
        try:
            await _kick_dead_user_once()
        except Exception as e:
            logger.exception("alive_check tick error: %s", e)


async def _kick_dead_user_once() -> None:
    """One tick of the dead-user reaper. Safe to call from tests."""
    if not _alive_users:
        return
    # 用 list() 快照 —— 迭代中 del dict key 在 Python 3 上是 RuntimeError
    async with _alive_lock:
        snapshot = list(_alive_users.items())
    for voucher, ttl in snapshot:
        async with _alive_lock:
            # 双检：snapshot 期间可能被 /alive 重置过。
            # 只有当前值仍等于快照值才递减；否则说明已续期，本次 tick 不动它（M-06）
            current = _alive_users.get(voucher)
            if current is None or current != ttl:
                continue
            if ttl - 1 <= 0:
                # 倒计时归零 → voucher 失效
                # 1) 让 scheduler 清掉这个 voucher 的所有 pending + in-flight 工作
                _alive_users.pop(voucher, None)
            else:
                _alive_users[voucher] = ttl - 1
                continue
        # 锁外调 scheduler.cancel_voucher（内部有自己的锁，避免嵌套锁死锁）
        if scheduler is not None:
            try:
                await scheduler.cancel_voucher(voucher)
            except Exception as e:
                logger.exception("cancel_voucher(%s) failed: %s", voucher, e)
        logger.info("User '%s' Lost Connection", voucher)


# ─── 静态资源挂载（catch-all）────────────────────────────────────
# html=True 让 `/` 自动返回 index.html；其它路径尝试匹配 static/ 下的文件。
# 必须放在所有 API 路由注册之后，否则会拦截 API 请求。
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

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
