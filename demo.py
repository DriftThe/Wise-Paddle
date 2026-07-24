"""
Wise-Paddle uvicorn 服务（批处理调度版）

设计：单 pipeline 一次处理一批（最多 5 张图），多用户的 page 任务
由 BatchScheduler 跨用户聚合后送给 pipeline。layout detection 和
VL 推理都能 batch 起来，单 pipeline 实例能打满 GPU 吞吐。

HTTP 接口：
- GET  /                          —— 前端 UI（单页 HTML）
- GET  /health                    —— 健康 + scheduler/pool 状态
- POST /ocr/upload                —— 上传图片（multipart），返回 OCR 结果 + 原图 base64
- POST /ocr/base64                —— base64 JSON body，同上
- POST /ocr/pdf                   —— PDF 占位（暂不真处理，返回 status=pending + placeholder）

响应里 image_b64 是原图 PNG，前端拿到可以直接画到 canvas 上叠检测框。
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
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

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
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
        "Building pipeline pool: %d pipeline(s), batch_max=%d, flush=%dms",
        POOL_SIZE, BATCH_MAX, int(BATCH_FLUSH_MS),
    )
    pool = PipelinePool(size=POOL_SIZE, factory=_make_pipeline)
    await pool.init()
    pool_ref = pool

    sched = BatchScheduler(
        pool=pool,
        max_batch=BATCH_MAX,
        flush_ms=BATCH_FLUSH_MS,
        n_workers=POOL_SIZE,  # 1 worker per pipeline slot
    )
    await sched.start()
    scheduler = sched
    logger.info("Service ready: pool=%d, batch_max=%d", POOL_SIZE, BATCH_MAX)
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


app = FastAPI(title="Wise-Paddle Service", version="0.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Request / Response 模型 ─────────────────────────────────────
class Base64Request(BaseModel):
    payload: str = Field(..., description="图片 base64 字符串，可带 data URI 前缀")


class OCRRegion(BaseModel):
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
    regions: list[OCRRegion]


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


class HealthResponse(BaseModel):
    status: str
    version: str
    pool_size: int
    pool_free: int
    scheduler_pending: int
    batch_max: int


# ─── 工具函数 ────────────────────────────────────────────────────
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


def _image_to_b64_png(rgb: np.ndarray) -> str:
    """rgb HWC uint8 → png base64（无 data URI 前缀）"""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _get_scheduler() -> BatchScheduler:
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler 尚未初始化")
    return scheduler


# ─── 路由 ────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def index():
    """前端 UI"""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="0.2.0",
        pool_size=POOL_SIZE,
        pool_free=pool_ref.qsize() if pool_ref else 0,
        scheduler_pending=scheduler.pending_size() if scheduler else 0,
        batch_max=BATCH_MAX,
    )


async def _process_one_image(rgb: np.ndarray) -> tuple[str, OCRPage, float]:
    """把一张 RGB 图送进 scheduler，等结果，转成 OCRPage（含原图 base64）。

    返回 (request_id, page, queue_wait_seconds)。
    """
    from PIL import Image as PILImage

    sched = _get_scheduler()
    request_id = uuid.uuid4().hex[:8]
    pil_img = PILImage.fromarray(rgb)

    submit_t = time.perf_counter()
    job = sched.submit(request_id=request_id, page_index=0, image=pil_img)
    page = await job.fut  # 阻塞等到 BatchScheduler 分发完成
    queue_wait = time.perf_counter() - submit_t

    return request_id, OCRPage(
        page_index=page.page_index,
        width=page.width,
        height=page.height,
        image_b64=_image_to_b64_png(rgb),
        regions=[
            OCRRegion(
                label=r.label,
                score=round(r.score, 4),
                rect=r.rect,
                md_path=str(r.md_path) if r.md_path else None,
                markdown=r.markdown,
            )
            for r in page.regions
        ],
    ), queue_wait


@app.post("/ocr/upload", response_model=OCRResponse)
async def ocr_upload(file: UploadFile = File(...)):
    raw = await file.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码上传的图片")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sched = _get_scheduler()
    st = time.perf_counter()
    request_id, page, queue_wait = await _process_one_image(rgb)
    elapsed = time.perf_counter() - st

    return OCRResponse(
        success=True,
        request_id=request_id,
        status="done",
        elapsed_seconds=round(elapsed, 3),
        queue_wait_seconds=round(queue_wait, 3),
        scheduler_pending=sched.pending_size(),
        pool_size=POOL_SIZE,
        pool_free=pool_ref.qsize() if pool_ref else 0,
        pages=[page],
    )


@app.post("/ocr/base64", response_model=OCRResponse)
async def ocr_base64(req: Base64Request):
    try:
        rgb = _decode_b64_to_rgb(req.payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    sched = _get_scheduler()
    st = time.perf_counter()
    request_id, page, queue_wait = await _process_one_image(rgb)
    elapsed = time.perf_counter() - st

    return OCRResponse(
        success=True,
        request_id=request_id,
        status="done",
        elapsed_seconds=round(elapsed, 3),
        queue_wait_seconds=round(queue_wait, 3),
        scheduler_pending=sched.pending_size(),
        pool_size=POOL_SIZE,
        pool_free=pool_ref.qsize() if pool_ref else 0,
        pages=[page],
    )


@app.post("/ocr/pdf", response_model=OCRResponse)
async def ocr_pdf(file: UploadFile = File(...)):
    """PDF 占位：暂不真处理，只接住请求并返回状态。前端拿 placeholder 渲染。

    等 BatchScheduler 稳定后会加 pdf2image + 多页 batch 提交。
    """
    raw = await file.read()
    if not raw[:4] == b"%PDF":
        raise HTTPException(status_code=400, detail="不是合法 PDF 文件")
    request_id = uuid.uuid4().hex[:8]
    logger.info("PDF placeholder: request_id=%s size=%d (no real processing yet)", request_id, len(raw))
    return JSONResponse(
        status_code=202,
        content={
            "success": True,
            "request_id": request_id,
            "status": "pending",
            "message": "PDF 暂未实现，请用图片",
            "elapsed_seconds": 0.0,
            "queue_wait_seconds": 0.0,
            "scheduler_pending": scheduler.pending_size() if scheduler else 0,
            "pool_size": POOL_SIZE,
            "pool_free": pool_ref.qsize() if pool_ref else 0,
            "pages": [],
        },
    )


# ─── 启动入口 ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "demo:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
        log_level="info",
    )
