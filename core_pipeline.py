"""
Wise-Paddle OCR pipeline.

Pipeline tree (per page):

    PIL.Image (任意尺寸)
        │
        ▼
    LayoutDetector  ── 内部 processor 自动 resize 到 800x800
        │ boxes (float xyxy)  / labels / scores
        ▼
    ┌─ 转 int rectangle (np.round → int) ─┐
    │  NMS (IoU)                          │  BoxFilter
    │  面积阈值                            │
    │  分数阈值                            │
    └────────────────────────────────────┘
        │ 保留的 LayoutBox
        ▼
    RegionCropper  ── numpy 切片裁剪 (BGR/HWC uint8)
        │
        ▼
    按 label 分桶 → 同桶 batch 提交给 VLPredictor (PaddleOCR-VL-1.6)
        │
        ▼
    每张裁剪图得到一段 markdown → 写入 text_result/page_<i>_box_<j>.md

并发模型：PipelinePool = asyncio.Queue(maxsize=N)，每个元素是完整的
OCRPipeline 实例。N 个并发槽位可同时占用 N 个 pipeline 跑推理。
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable, Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger("wise-paddle.pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# 兼容性补丁：transformers 5.x 不再带 "default" rope init 入口，老模型需要补
# ─────────────────────────────────────────────────────────────────────────────
def _patch_rope_default() -> None:
    import transformers.modeling_rope_utils as r

    if "default" in r.ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters(
        config,
        device=None,
        seq_len=None,
        layer_type=None,
    ):
        base = config.rope_theta
        dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).float().to(device)
                / dim
            )
        )
        return inv_freq, 1.0

    r.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    logger.info("Patched transformers ROPE_INIT_FUNCTIONS['default']")


_patch_rope_default()


# ─────────────────────────────────────────────────────────────────────────────
# 数据类
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LayoutBox:
    """单个版面区域。"""

    xyxy: np.ndarray  # float32, 形状 (4,) —— [x1, y1, x2, y2]（原图坐标系）
    label_id: int
    label_name: str
    score: float

    @property
    def int_rect(self) -> tuple[int, int, int, int]:
        return tuple(int(round(float(v))) for v in self.xyxy)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class RegionResult:
    """一个裁剪区域的最终结果。"""

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
    """一页的处理结果。"""

    page_index: int
    width: int
    height: int
    elapsed_seconds: float
    regions: list[RegionResult] = field(default_factory=list)

    @property
    def markdown(self) -> str:
        return "\n\n".join(r.markdown for r in self.regions)


@dataclass
class Job:
    """一个待处理的 page 任务（由 BatchScheduler 跨用户聚合）。"""

    request_id: str
    page_index: int
    image: Image.Image
    fut: "asyncio.Future[PageResult]" = field(default=None)  # type: ignore[type-arg]


# ─────────────────────────────────────────────────────────────────────────────
# 1) Layout detection —— PP-DocLayoutV3
# ─────────────────────────────────────────────────────────────────────────────
class LayoutDetector:
    """把任意尺寸的图过 PP-DocLayoutV3，返回 (box, label, score) 三元组。"""

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        score_threshold: float = 0.5,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        logger.info("Loading layout model: %s", model_path)
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = (
            AutoModelForObjectDetection.from_pretrained(model_path, dtype=dtype)
            .to(device)
            .eval()
        )
        self.device = device
        self.score_threshold = score_threshold
        self.id2label: dict[int, str] = {
            int(k): v for k, v in self.model.config.id2label.items()
        }
        # processor 内置 800x800 resize；只要原图长边合理，processor 自动适配
        self._max_long_side = 1600  # 防止 4K+ 大图把显存打爆

    def _maybe_downscale(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        m = max(w, h)
        if m <= self._max_long_side:
            return image
        scale = self._max_long_side / m
        return image.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.BILINEAR,
        )

    @torch.no_grad()
    def detect(self, images: Sequence[Image.Image]) -> list[list[LayoutBox]]:
        if not images:
            return []
        scaled = [self._maybe_downscale(im.convert("RGB")) for im in images]

        inputs = self.processor(images=scaled, return_tensors="pt").to(self.device)
        target_sizes = torch.tensor(
            [[im.height, im.width] for im in images], device=self.device
        )
        outputs = self.model(**inputs)
        # post_process_object_detection 在 threshold 处做一次初筛；后面 BoxFilter
        # 再做更细的 IoU / 面积过滤
        raw = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.score_threshold
        )

        out: list[list[LayoutBox]] = []
        for r, src in zip(raw, images):
            boxes = r["boxes"].detach().cpu().numpy()
            labels = r["labels"].detach().cpu().numpy()
            scores = r["scores"].detach().cpu().numpy()
            page_boxes: list[LayoutBox] = []
            for box, lid, sc in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box
                # 截到原图边界内
                x1 = max(0.0, min(float(x1), src.width))
                x2 = max(0.0, min(float(x2), src.width))
                y1 = max(0.0, min(float(y1), src.height))
                y2 = max(0.0, min(float(y2), src.height))
                page_boxes.append(
                    LayoutBox(
                        xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
                        label_id=int(lid),
                        label_name=self.id2label.get(int(lid), str(int(lid))),
                        score=float(sc),
                    )
                )
            out.append(page_boxes)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2) Box filter —— NMS + 面积 / 分数门槛
# ─────────────────────────────────────────────────────────────────────────────
class BoxFilter:
    """纯 numpy NMS + 过滤；不依赖额外库。"""

    def __init__(
        self,
        iou_threshold: float = 0.5,
        min_area: float = 16 * 16,
        min_score: float = 0.5,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.min_area = min_area
        self.min_score = min_score

    def _iou(self, a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    def filter(self, boxes: list[LayoutBox]) -> list[LayoutBox]:
        # 先按分数从高到低
        keep: list[LayoutBox] = []
        candidates = sorted(boxes, key=lambda b: b.score, reverse=True)
        for b in candidates:
            if b.score < self.min_score or b.area < self.min_area:
                continue
            if any(self._iou(b.xyxy, k.xyxy) > self.iou_threshold for k in keep):
                continue
            keep.append(b)
        return keep


# ─────────────────────────────────────────────────────────────────────────────
# 3) Region cropper —— numpy 切片
# ─────────────────────────────────────────────────────────────────────────────
class RegionCropper:
    """用 numpy 把 LayoutBox 对应的区域切出来，返回 (crop_rgb, int_rect)。"""

    def crop(self, image: Image.Image, box: LayoutBox) -> np.ndarray:
        # PIL 转 RGB numpy (H, W, 3) uint8
        arr = np.asarray(image.convert("RGB"))
        x1, y1, x2, y2 = box.int_rect
        # clamp 到合法范围
        h, w = arr.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return arr[y1:y2, x1:x2].copy()


# ─────────────────────────────────────────────────────────────────────────────
# 4) VL predictor —— PaddleOCR-VL-1.6，按 label 分桶后 batch
# ─────────────────────────────────────────────────────────────────────────────
class VLPredictor:
    """包裹 PaddleOCR-VL-1.6，给一批裁剪图做 VL 识别。"""

    # PaddleOCR-VL 官方推荐的 task prompt（来自 PaddleOCR-VL-1.6 README）
    DEFAULT_PROMPTS: dict[str, str] = {
        "text": "OCR:",
        "paragraph_title": "OCR:",
        "doc_title": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "image": "OCR:",
        "chart": "Chart Recognition:",
        "abstract": "OCR:",
        "reference": "OCR:",
        "reference_content": "OCR:",
        "footer": "OCR:",
        "header": "OCR:",
        "footnote": "OCR:",
        "seal": "OCR:",
        "number": "OCR:",
        "_default": "OCR:",
    }

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        prompts: Optional[dict[str, str]] = None,
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 256,
        max_pixels: int = 1280 * 28 * 28,  # 官方默认 1MP
        max_forward_batch: int = 4,        # 单次 VL forward 最多 image 数（显存上限）
        attn_impl: str = "sdpa",
    ) -> None:
        logger.info("Loading VL model: %s", model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        # 关键：decoder-only 必须 left-padding，否则 batch 推理全乱
        try:
            self.processor.tokenizer.padding_side = "left"
        except Exception:
            pass
        # 关键：必须用 AutoModelForImageTextToText，不是 AutoModelForCausalLM
        # 之前的 CausalLM 入口导致 model 看不到 image，全靠编造
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=dtype,
                attn_implementation=attn_impl,
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self.max_forward_batch = max(1, int(max_forward_batch))
        # v1.6 image_processor.size = SizeDict(shortest_edge=112896, longest_edge=1003520)
        # 调用 apply_chat_template 时通过 images_kwargs 覆盖
        self.shortest_edge = 112896
        self.prompts = {**self.DEFAULT_PROMPTS, **(prompts or {})}
        # EOS：模型的 generation_config.json 里写的是 </s> (id=2)
        tok = self.processor.tokenizer
        self.eos_token_id = tok.convert_tokens_to_ids("</s>") or tok.eos_token_id or 2
        self.pad_token_id = (
            self.processor.tokenizer.pad_token_id
            or tok.convert_tokens_to_ids("<unk>")
            or 0
        )

    def _prompt_for(self, label: str) -> str:
        return self.prompts.get(label, self.prompts["_default"])

    def _build_messages(self, images: Sequence[Image.Image], label: str) -> list[list[dict]]:
        """给每张图造一个 messages（apply_chat_template batched 模式需要 list of conversations）。"""
        user_text = self._prompt_for(label)
        return [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
            for img in images
        ]

    def _forward_once(self, images: Sequence[Image.Image], label: str) -> list[str]:
        """单次 VL forward，调用方负责保证 len(images) <= self.max_forward_batch。"""
        if not images:
            return []
        conversations = self._build_messages(images, label)
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,                     # batch 推理要 padding
            images_kwargs={
                "size": {
                    "shortest_edge": self.shortest_edge,
                    "longest_edge": self.max_pixels,
                }
            },
        ).to(self.device)
        ids = inputs["input_ids"]
        # image token 实际通过 mm_token_type_ids 标记 (apply_chat_template 不插入 <|image_pad|>)
        cnt = 0
        if "mm_token_type_ids" in inputs:
            cnt = int((inputs["mm_token_type_ids"] == 1).sum().item())
        logger.info(
            "  [VL fwd=%d label=%s] seq_len=%d image_tokens=%d",
            len(images), label, ids.shape[1], cnt,
        )
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True,
            repetition_penalty=1.15,
        )
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)

    @torch.no_grad()
    def recognize_batch(
        self, images: Sequence[Image.Image], label: str
    ) -> list[str]:
        """同 label 的图用同一 prompt 做 batch 推理。len > max_forward_batch 时拆 sub-batch。

        例：12 张 text crop, max_forward_batch=4 → 3 次 VL forward (4+4+4)
        """
        if not images:
            return []
        cap = self.max_forward_batch
        if len(images) <= cap:
            return self._forward_once(images, label)
        # 拆 sub-batch
        out: list[str] = []
        for start in range(0, len(images), cap):
            chunk = list(images[start:start + cap])
            out.extend(self._forward_once(chunk, label))
        return out

    @torch.no_grad()
    def recognize_grouped(
        self, items: list[tuple[Image.Image, str]]
    ) -> list[str]:
        """按 label 分桶 → 每个桶一次 batch 推理 → 按原顺序还原。"""
        # 桶：(label -> [(idx, img)])
        buckets: dict[str, list[tuple[int, Image.Image]]] = {}
        for idx, (img, lab) in enumerate(items):
            buckets.setdefault(lab, []).append((idx, img))

        results: list[Optional[str]] = [None] * len(items)
        for label, bucket in buckets.items():
            indices = [i for i, _ in bucket]
            imgs = [im for _, im in bucket]
            texts = self.recognize_batch(imgs, label)
            for i, t in zip(indices, texts):
                results[i] = t
        return [r or "" for r in results]  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# 5) Pipeline —— 把上述 4 步串成一条主干
# ─────────────────────────────────────────────────────────────────────────────
class OCRPipeline:
    """对一张图跑 layout detection → 过滤 → 裁剪 → VL 识别 → 落盘。"""

    def __init__(
        self,
        layout_model_path: str,
        vl_model_path: str,
        device: torch.device,
        output_dir: Path,
        box_filter: Optional[BoxFilter] = None,
        score_threshold: float = 0.5,
        max_new_tokens: int = 256,
        max_regions: int = 100,   # 几乎不限制，让每图所有 region 都进 VL
        vl_max_forward_batch: int = 4,  # 单次 VL forward image 上限（显存）
    ) -> None:
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.layout = LayoutDetector(
            layout_model_path, device, score_threshold=score_threshold
        )
        self.filter = box_filter or BoxFilter()
        self.cropper = RegionCropper()
        self.vl = VLPredictor(
            vl_model_path, device,
            max_new_tokens=max_new_tokens,
            max_forward_batch=vl_max_forward_batch,
        )
        self.max_regions = max_regions

    def process_page(
        self,
        image: Image.Image,
        page_index: int = 0,
        request_id: Optional[str] = None,
    ) -> PageResult:
        st = time.perf_counter()
        image = image.convert("RGB")
        w, h = image.size

        # 1) layout detection（内部自动 resize）
        layouts = self.layout.detect([image])[0]
        # 2) 过滤（NMS + 面积 + 分数）
        kept = self.filter.filter(layouts)
        # 2.5) 限制单页最多送进 VL 的 region 数；按分数截断
        kept = kept[: self.max_regions]
        # 3) 裁剪
        crops: list[tuple[Image.Image, LayoutBox]] = []
        for box in kept:
            arr = self.cropper.crop(image, box)
            if arr.size == 0 or arr.shape[0] < 2 or arr.shape[1] < 2:
                continue
            crops.append((Image.fromarray(arr), box))
        # 4) 按 label 分桶 → VL batch
        items = [(img, box.label_name) for img, box in crops]
        markdowns = self.vl.recognize_grouped(items)

        # 5) 落盘 text_result/request_id/page_X_box_Y.md
        # request_id 隔离不同请求的输出，避免并发 path 冲突
        if request_id is None:
            import uuid
            request_id = uuid.uuid4().hex[:8]
        req_dir = self.output_dir / request_id
        req_dir.mkdir(parents=True, exist_ok=True)
        regions: list[RegionResult] = []
        for (img, box), md in zip(crops, markdowns):
            md_path = req_dir / f"page_{page_index}_box_{box.label_id}_{box.label_name}.md"
            # 同 (page, label_name) 只留一份，加 box_index 避免重名
            md_path = self._unique_path(md_path)
            md_path.write_text(md or "", encoding="utf-8")
            regions.append(
                RegionResult(
                    page_index=page_index,
                    box_index=len(regions),
                    label=box.label_name,
                    score=box.score,
                    rect=box.int_rect,
                    crop_shape=(img.height, img.width),
                    markdown=md,
                    md_path=md_path,
                )
            )

        return PageResult(
            page_index=page_index,
            width=w,
            height=h,
            elapsed_seconds=time.perf_counter() - st,
            regions=regions,
        )

    @staticmethod
    def _unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        i = 1
        while True:
            cand = path.with_name(f"{path.stem}_{i}{path.suffix}")
            if not cand.exists():
                return cand
            i += 1

    def process_batch(self, jobs: list[Job]) -> list[PageResult]:
        """一次处理 N 张图（跨用户共享 layout + VL 推理）。

        流水线：
        1) layout detection 一次喂所有图（PP-DocLayoutV3 内部 batch）
        2) 每图独立 filter + crop
        3) 跨图按 label 分桶 → VL batch（跨用户共享 GPU）
        4) markdown 落盘到各自 request_id/ 子目录，返回 PageResult
        """
        if not jobs:
            return []
        st = time.perf_counter()

        # 1) layout detection 整批一次
        images_rgb: list[Image.Image] = [j.image.convert("RGB") for j in jobs]
        layouts_per_page = self.layout.detect(images_rgb)

        # 2) 每图独立 filter + crop
        crops_per_page: list[list[tuple[Image.Image, LayoutBox]]] = []
        for image, layouts in zip(images_rgb, layouts_per_page):
            kept = self.filter.filter(layouts)[: self.max_regions]
            crops: list[tuple[Image.Image, LayoutBox]] = []
            for box in kept:
                arr = self.cropper.crop(image, box)
                if arr.size == 0 or arr.shape[0] < 2 or arr.shape[1] < 2:
                    continue
                crops.append((Image.fromarray(arr), box))
            crops_per_page.append(crops)

        # 3) 跨图按 label 桶聚合（同一个 bucket 一次 VL forward）
        # bucket: label_name -> list of (page_idx, crop_idx, crop_img)
        buckets: dict[str, list[tuple[int, int, Image.Image]]] = {}
        for p_idx, crops in enumerate(crops_per_page):
            for c_idx, (img, box) in enumerate(crops):
                buckets.setdefault(box.label_name, []).append((p_idx, c_idx, img))

        # 4) 每个 bucket 调一次 VL batch；结果回填到 md_lookup
        md_lookup: dict[tuple[int, int], str] = {}
        for label, items in buckets.items():
            imgs = [it[2] for it in items]
            texts = self.vl.recognize_batch(imgs, label)
            for (p_idx, c_idx, _), md in zip(items, texts):
                md_lookup[(p_idx, c_idx)] = md

        # 5) 写文件 + 构造 PageResult
        results: list[PageResult] = []
        for p_idx, (job, crops) in enumerate(zip(jobs, crops_per_page)):
            image = images_rgb[p_idx]
            req_dir = self.output_dir / job.request_id
            req_dir.mkdir(parents=True, exist_ok=True)
            regions: list[RegionResult] = []
            for c_idx, (img, box) in enumerate(crops):
                md = md_lookup.get((p_idx, c_idx), "")
                md_path = req_dir / f"page_{job.page_index}_box_{box.label_id}_{box.label_name}.md"
                md_path = self._unique_path(md_path)
                md_path.write_text(md, encoding="utf-8")
                regions.append(
                    RegionResult(
                        page_index=job.page_index,
                        box_index=c_idx,
                        label=box.label_name,
                        score=box.score,
                        rect=box.int_rect,
                        crop_shape=(img.height, img.width),
                        markdown=md,
                        md_path=md_path,
                    )
                )
            results.append(
                PageResult(
                    page_index=job.page_index,
                    width=image.width,
                    height=image.height,
                    elapsed_seconds=time.perf_counter() - st,  # 整批 wall time
                    regions=regions,
                )
            )
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 6) Pipeline pool —— asyncio.Queue 模式，支持 N 路并发
# ─────────────────────────────────────────────────────────────────────────────
class PipelinePool:
    """asyncio.Queue 实现的 pipeline 池。lease() 上下文管理器借出 / 归还。"""

    def __init__(
        self,
        size: int,
        factory: Callable[[], OCRPipeline],
    ) -> None:
        if size < 1:
            raise ValueError("pool size must be >= 1")
        self.size = size
        self._factory = factory
        self._queue: asyncio.Queue[OCRPipeline] = asyncio.Queue(maxsize=size)

    async def init(self) -> None:
        logger.info("Initializing pipeline pool: %d pipelines", self.size)
        for i in range(self.size):
            logger.info("  building pipeline %d/%d", i + 1, self.size)
            pipe = self._factory()
            await self._queue.put(pipe)
        logger.info("Pipeline pool ready")

    async def close(self) -> None:
        # 当前实现里 pipeline 不持有需要显式释放的资源；保留接口以便扩展
        logger.info("Pipeline pool closed")

    def qsize(self) -> int:
        return self._queue.qsize()

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[OCRPipeline]:
        wait = time.perf_counter()
        pipe = await self._queue.get()
        waited = time.perf_counter() - wait
        logger.info(
            "lease: waited %.3fs, free=%d/%d", waited, self._queue.qsize(), self.size
        )
        try:
            yield pipe
        finally:
            await self._queue.put(pipe)
            logger.info(
                "release: free=%d/%d", self._queue.qsize(), self.size
            )


# ─────────────────────────────────────────────────────────────────────────────
# 7) BatchScheduler —— 攒批 + 借 pipeline + 分发结果
# ─────────────────────────────────────────────────────────────────────────────
class BatchScheduler:
    """在 PipelinePool 之上做的"卡车"调度器：

    - 上游：任意线程/协程 submit(request_id, page_idx, image) → 返回 Future[PageResult]
    - 下游：n_workers 个 worker 协程循环 ——
        * 抢一个 Job 当头
        * 继续等 flush_ms 时间，凑够 max_batch 个
        * 从 pool 借一个 pipeline
        * 调 pipeline.process_batch([...]) 一次处理整批（跨用户共享 layout + VL）
        * 把结果 set_result 回每个 Job 的 Future
    """

    def __init__(
        self,
        pool: PipelinePool,
        max_batch: int = 5,
        flush_ms: float = 250.0,
        n_workers: Optional[int] = None,
    ) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        self.pool = pool
        self.max_batch = max_batch
        self.flush_seconds = flush_ms / 1000.0
        # 默认开跟 pool 一样多的 worker；pool=2 → 2 路 batch 并行
        self.n_workers = n_workers if n_workers is not None else pool.size
        self._pending: list[Job] = []
        self._lock = asyncio.Lock()
        self._wakeup = asyncio.Event()
        self._closed = False
        self._worker_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        logger.info(
            "BatchScheduler starting %d worker(s), max_batch=%d, flush=%.0fms",
            self.n_workers, self.max_batch, self.flush_seconds * 1000,
        )
        for i in range(self.n_workers):
            self._worker_tasks.append(asyncio.create_task(self._worker_loop(i)))
        logger.info("BatchScheduler ready")

    async def close(self) -> None:
        self._closed = True
        self._wakeup.set()  # 唤醒所有 worker 让它们看到 _closed
        for t in self._worker_tasks:
            t.cancel()
        for t in self._worker_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._worker_tasks.clear()
        logger.info("BatchScheduler closed")

    def submit(self, request_id: str, page_index: int, image: Image.Image) -> Job:
        """提交一个 page 任务，返回 Job（Job.fut 是 asyncio.Future[PageResult]）。"""
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[PageResult] = loop.create_future()
        job = Job(request_id=request_id, page_index=page_index, image=image, fut=fut)
        # 直接 append + set wakeup（无 await，不算临界区）
        self._pending.append(job)
        self._wakeup.set()
        return job

    def pending_size(self) -> int:
        return len(self._pending)

    async def _worker_loop(self, wid: int) -> None:
        """worker 主循环：等至少 1 个 job → 凑 batch → 借 pipeline 处理 → 分发结果。"""
        logger.info("scheduler worker[%d] started", wid)
        while not self._closed:
            # 1) 等到至少 1 个 job
            while not self._pending and not self._closed:
                self._wakeup.clear()
                await self._wakeup.wait()
            if self._closed:
                break

            # 2) 抢 1 个 job 当头
            async with self._lock:
                if not self._pending:
                    continue  # close 唤醒
                batch = [self._pending.pop(0)]
                if not self._pending:
                    self._wakeup.clear()

            # 3) 凑 batch（直到 max_batch 或 flush_seconds 超时）
            deadline = time.perf_counter() + self.flush_seconds
            while len(batch) < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                async with self._lock:
                    while self._pending and len(batch) < self.max_batch:
                        batch.append(self._pending.pop(0))
                    if not self._pending:
                        self._wakeup.clear()
                if len(batch) >= self.max_batch:
                    break

            # 4) 借 pipeline + 跑批 + 分发
            t0 = time.perf_counter()
            async with self.pool.lease() as pipe:
                wait = time.perf_counter() - t0
                users = sorted({j.request_id for j in batch})
                logger.info(
                    "scheduler[%d] lease: waited %.3fs, free=%d/%d, batch=%d (%d user %s)",
                    wid, wait, self.pool.qsize(), self.pool.size,
                    len(batch), len(users), users,
                )
                from starlette.concurrency import run_in_threadpool
                results = await run_in_threadpool(pipe.process_batch, batch)
            dt = time.perf_counter() - t0
            for job, page in zip(batch, results):
                if not job.fut.done():
                    job.fut.set_result(page)
            logger.info(
                "scheduler[%d] done: batch=%d in %.2fs, free=%d/%d",
                wid, len(batch), dt, self.pool.qsize(), self.pool.size,
            )
        logger.info("scheduler worker[%d] exited", wid)


# ─────────────────────────────────────────────────────────────────────────────
# 工厂 + 默认路径
# ─────────────────────────────────────────────────────────────────────────────
LAYOUT_MODEL_PATH = "./model/PP-DocLayoutV3"
VL_MODEL_PATH = "./model/PaddleOCR-VL-1.6"
DEFAULT_OUTPUT_DIR = Path("./text_result")


def make_default_pipeline(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: Optional[torch.device] = None,
) -> OCRPipeline:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return OCRPipeline(
        layout_model_path=LAYOUT_MODEL_PATH,
        vl_model_path=VL_MODEL_PATH,
        device=device,
        output_dir=output_dir,
    )
