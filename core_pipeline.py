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
import collections
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    AutoModelForImageTextToText,
    AutoProcessor,
)

logger = logging.getLogger("wise-paddle.pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# 兼容性补丁：transformers 5.x 不再带 "default" rope init 入口，老模型需要补
# ─────────────────────────────────────────────────────────────────────────────
def _patch_rope_default() -> None:
    """Patch transformers 5.x to restore the missing 'default' RoPE init entry.

    Older model configs reference ``rope_type="default"`` which was removed in
    transformers 5.x. We re-register an equivalent implementation so those
    configs keep loading without manual edits.
    """
    import transformers.modeling_rope_utils as rope_utils

    if "default" in rope_utils.ROPE_INIT_FUNCTIONS:
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

    rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    logger.info("Patched transformers ROPE_INIT_FUNCTIONS['default']")


# Module-level patch; idempotent and safe to call multiple times.
_patch_rope_default()

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────
# 取消追踪集合的软上限。超过后触发一次清理（移除已 done 的 user）。
_CANCELLED_TRACK_SOFT_LIMIT = 4096
# _unique_path 重名时最多尝试次数，避免死循环。
_UNIQUE_PATH_MAX_ATTEMPTS = 10_000
# 用户书签闲置多久后清理（_user_order / _user_completed 防泄漏）
_USER_IDLE_PRUNE_S = 600.0
# 合法 request_id 白名单（与 app.py 的 _sanitize_user_id 同口径，纵深防御）
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


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
    """一页的处理结果。

    ``cancelled=True`` 表示这个 page 在 process_batch 入口/中段被 voucher 取消
    过滤掉了，``regions`` 为空，handler 据此可以走 "cancelled" 响应路径。
    """

    page_index: int
    width: int
    height: int
    elapsed_seconds: float
    regions: list[RegionResult] = field(default_factory=list)
    cancelled: bool = False

    @property
    def markdown(self) -> str:
        return "\n\n".join(r.markdown for r in self.regions)


@dataclass
class Job:
    """一个待处理的 page 任务（由 BatchScheduler 跨用户聚合）。

    ``voucher_id`` 是上层 alive_check 用的会话级 ID；
    ``scheduler.cancel_voucher(voucher_id)`` 会一次性清掉所有挂这个 voucher
    的 pending job，并把 voucher 标记为 "已取消"，让正在 process_batch 里的
    同 voucher job 在 crop 之后、VL 之前被丢弃。空字符串表示不绑定 voucher
    （旧代码兼容）。
    """

    request_id: str
    page_index: int
    image: Image.Image
    fut: "asyncio.Future[PageResult]" = field(default=None)  # type: ignore[type-arg]
    voucher_id: str = ""


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
        """If the image's long side exceeds ``_max_long_side``, downscale it.

        Returns the original image unchanged when it is already small enough.
        """
        w, h = image.size
        m = max(w, h)
        if m <= self._max_long_side:
            return image
        scale = self._max_long_side / m
        return image.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.BILINEAR,
        )

    @torch.no_grad()
    def detect(self, images: Sequence[Image.Image]) -> list[list[LayoutBox]]:
        """Run layout detection on a batch of images.

        Args:
            images: One or more PIL images (any size, any mode).

        Returns:
            A list (one entry per input image) of ``LayoutBox`` lists.
        """
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
    """纯 numpy NMS + 过滤；不依赖额外库。

    可调精度参数（影响 layout 召回/裁剪质量）：

    - ``iou_threshold``: NMS 重叠上限（越大越激进去重）
    - ``min_area``:      最小框面积（像素²）
    - ``min_score``:     最低置信度
    - ``unclip_ratio``:  NMS 后把框向外扩的比例（0.05 = 每边扩 5%）。给 VL 更多
                        上下文，提升 OCR 准确率；过大会把别的 region 也包进来。
                        doclayout 边界偏紧时这个最有用。
    - ``expand_pixels``: 每边再多扩 N 个像素（绝对值）。和 ratio 叠加生效。
    """

    def __init__(
            self,
            iou_threshold: float = 0.5,
            min_area: float = 16 * 16,
            min_score: float = 0.5,
            unclip_ratio: float = 0.0,
            expand_pixels: float = 0.0,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.min_area = float(min_area)
        self.min_score = float(min_score)
        self.unclip_ratio = float(unclip_ratio)
        self.expand_pixels = float(expand_pixels)

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        """Compute IoU between two xyxy boxes (float arrays of length 4)."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    def _unclip(self, box: np.ndarray) -> np.ndarray:
        """按 ratio + 绝对像素把框向外扩，返回 (x1,y1,x2,y2)。"""
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        dx = w * self.unclip_ratio + self.expand_pixels
        dy = h * self.unclip_ratio + self.expand_pixels
        return np.array([x1 - dx, y1 - dy, x2 + dx, y2 + dy], dtype=np.float32)

    def filter(
            self,
            boxes: list[LayoutBox],
            page_size: Optional[tuple[int, int]] = None,
    ) -> list[LayoutBox]:
        """过滤 + NMS + 可选 unclip。

        Args:
            boxes: 待过滤的 ``LayoutBox`` 列表。
            page_size: ``(W, H)`` 可选；给 unclip 提供边界 clamp（防止扩出图外）。

        Returns:
            过滤后保留的 ``LayoutBox`` 列表（按分数降序）。
        """
        # 先按分数从高到低
        keep: list[LayoutBox] = []
        candidates = sorted(boxes, key=lambda b: b.score, reverse=True)
        for b in candidates:
            if b.score < self.min_score or b.area < self.min_area:
                continue
            if any(self._iou(b.xyxy, k.xyxy) > self.iou_threshold for k in keep):
                continue
            # unclip：在 NMS 之后，避免影响 NMS 决策
            if self.unclip_ratio > 0 or self.expand_pixels > 0:
                expanded = self._unclip(b.xyxy).copy()
                if page_size is not None:
                    w, h = page_size
                    expanded[0] = max(0.0, expanded[0])
                    expanded[1] = max(0.0, expanded[1])
                    expanded[2] = min(float(w), expanded[2])
                    expanded[3] = min(float(h), expanded[3])
                b = LayoutBox(
                    xyxy=expanded,
                    label_id=b.label_id,
                    label_name=b.label_name,
                    score=b.score,
                )
            keep.append(b)
        return keep


# ─────────────────────────────────────────────────────────────────────────────
# 3) Region cropper —— numpy 切片
# ─────────────────────────────────────────────────────────────────────────────
class RegionCropper:
    """用 numpy 把 LayoutBox 对应的区域切出来，返回 RGB ``np.ndarray``。"""

    def crop(self, rgb: np.ndarray, box: LayoutBox) -> np.ndarray:
        """Crop ``box`` from an RGB ``(H, W, 3)`` uint8 array.

        Returns a 1×1 black placeholder if the clamped box collapses to empty.
        """
        x1, y1, x2, y2 = box.int_rect
        # clamp 到合法范围
        h, w = rgb.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return rgb[y1:y2, x1:x2].copy()


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
            max_pixels: int = 1280 * 28 * 28,  # 官方默认 1MP（longest_edge）
            min_pixels: int = 112896,  # 官方默认 shortest_edge
            max_forward_batch: int = 10,  # 单次 VL forward 最多 image 数（显存上限）
            attn_impl: str = "sdpa",
            repetition_penalty: float = 1.15,  # 防止 batch 推理陷入重复循环
            do_sample: bool = False,  # greedy 解码；想更"活"可以改 True
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
        self.max_new_tokens = int(max_new_tokens)
        self.max_pixels = int(max_pixels)
        self.min_pixels = int(min_pixels)
        self.max_forward_batch = max(1, int(max_forward_batch))
        self.repetition_penalty = float(repetition_penalty)
        self.do_sample = bool(do_sample)
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

    def _build_messages(
            self, images: Sequence[Image.Image], label: str
    ) -> list[list[dict]]:
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
        """单次 VL forward，调用方负责保证 ``len(images) <= self.max_forward_batch``。"""
        if not images:
            return []
        conversations = self._build_messages(images, label)
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,  # batch 推理要 padding
            images_kwargs={
                "size": {
                    "shortest_edge": self.min_pixels,
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
            do_sample=self.do_sample,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True,
            repetition_penalty=self.repetition_penalty,
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
    """对一张图跑 layout detection → 过滤 → 裁剪 → VL 识别 → 落盘。

    所有可调精度/速度/显存参数都从构造参数传进来（app.py 从 .env 读）。
    """

    def __init__(
            self,
            layout_model_path: str,
            vl_model_path: str,
            device: torch.device,
            output_dir: Path,
            # ---- LayoutDetector ----
            score_threshold: float = 0.5,
            # ---- BoxFilter ----
            box_iou_threshold: float = 0.5,
            box_min_area: float = 16 * 16,
            box_min_score: float = 0.5,
            box_unclip_ratio: float = 0.0,  # 0.05 = 每边向外扩 5%，提精度
            box_expand_pixels: float = 0.0,  # 额外每边扩 N 像素
            # ---- VLPredictor ----
            max_new_tokens: int = 256,
            vl_min_pixels: int = 112896,
            vl_max_pixels: int = 1280 * 28 * 28,
            vl_max_forward_batch: int = 4,
            vl_repetition_penalty: float = 1.15,
            vl_do_sample: bool = False,
            # ---- runtime ----
            max_regions: int = 100,
            dtype: torch.dtype = torch.bfloat16,
            attn_impl: str = "sdpa",
    ) -> None:
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.layout = LayoutDetector(
            layout_model_path, device, score_threshold=score_threshold,
        )
        # NOTE: 不用 self.filter —— 会 shadow Python 内置 filter()
        self.box_filter = BoxFilter(
            iou_threshold=box_iou_threshold,
            min_area=box_min_area,
            min_score=box_min_score,
            unclip_ratio=box_unclip_ratio,
            expand_pixels=box_expand_pixels,
        )
        self.cropper = RegionCropper()
        self.vl = VLPredictor(
            vl_model_path, device,
            max_new_tokens=max_new_tokens,
            min_pixels=vl_min_pixels,
            max_pixels=vl_max_pixels,
            max_forward_batch=vl_max_forward_batch,
            repetition_penalty=vl_repetition_penalty,
            do_sample=vl_do_sample,
            attn_impl=attn_impl,
            dtype=dtype,
        )
        self.max_regions = int(max_regions)

    def _ensure_output_dir(self) -> None:
        """每次处理前都确认根目录在（外部可能 mavis-trash 掉）。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _safe_request_dir(self, request_id: str) -> Path:
        """构造 request 输出目录并做防穿越校验（H-01 纵深防御）。"""
        if not _REQUEST_ID_RE.fullmatch(request_id or ""):
            raise ValueError(f"illegal request_id: {request_id!r}")
        base = self.output_dir.resolve()
        req_dir = (base / request_id).resolve()
        if not req_dir.is_relative_to(base):
            raise ValueError(f"request_id escapes output dir: {request_id!r}")
        return req_dir

    @staticmethod
    def _unique_path(path: Path) -> Path:
        """Return ``path`` if it doesn't exist, else append ``_1``, ``_2`` …

        Raises ``FileExistsError`` after ``_UNIQUE_PATH_MAX_ATTEMPTS`` tries to
        avoid an unbounded loop on pathological filesystems.
        """
        if not path.exists():
            return path
        for i in range(1, _UNIQUE_PATH_MAX_ATTEMPTS + 1):
            cand = path.with_name(f"{path.stem}_{i}{path.suffix}")
            if not cand.exists():
                return cand
        raise FileExistsError(
            f"cannot find unique path under {path} after "
            f"{_UNIQUE_PATH_MAX_ATTEMPTS} attempts"
        )

    @staticmethod
    def _empty_page_result(job: Job) -> PageResult:
        """构造一个空 PageResult（voucher 已取消的占位）。"""
        return PageResult(
            page_index=job.page_index,
            width=0,
            height=0,
            elapsed_seconds=0.0,
            regions=[],
            cancelled=True,
        )

    def _crop_page(
            self,
            image: Image.Image,
            layouts: list[LayoutBox],
    ) -> list[tuple[Image.Image, LayoutBox]]:
        """Filter boxes on one page and crop them into ``(PIL.Image, LayoutBox)`` pairs."""
        kept = self.box_filter.filter(
            layouts, page_size=(image.width, image.height)
        )[: self.max_regions]
        crops: list[tuple[Image.Image, LayoutBox]] = []
        # 每页只做一次 RGB 转换（L-10）
        rgb = np.asarray(image.convert("RGB"))
        for box in kept:
            arr = self.cropper.crop(rgb, box)
            if arr.size == 0 or arr.shape[0] < 2 or arr.shape[1] < 2:
                continue
            crops.append((Image.fromarray(arr), box))
        return crops

    @staticmethod
    def _write_regions(
            req_dir: Path,
            page_index: int,
            crops: list[tuple[Image.Image, LayoutBox]],
            md_lookup: dict[tuple[int, int], str],
            page_offset: int,
    ) -> list[RegionResult]:
        """Write markdown for each crop and build ``RegionResult`` list.

        ``page_offset`` is the index of this page within the current active
        batch (used to look up ``md_lookup``); it differs from ``page_index``
        when the batch was filtered mid-flight.
        """
        req_dir.mkdir(parents=True, exist_ok=True)
        regions: list[RegionResult] = []
        for c_idx, (img, box) in enumerate(crops):
            md = md_lookup.get((page_offset, c_idx), "")
            # label 消毒 + uuid 短后缀，避免特殊字符与并发写同名竞态（L-03 / M-07）
            safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", box.label_name).strip("_") or "box"
            md_path = req_dir / (
                f"page_{page_index}_box_{box.label_id}_{safe_label}"
                f"_{uuid.uuid4().hex[:8]}.md"
            )
            md_path = OCRPipeline._unique_path(md_path)
            md_path.write_text(md, encoding="utf-8")
            regions.append(
                RegionResult(
                    page_index=page_index,
                    box_index=c_idx,
                    label=box.label_name,
                    score=box.score,
                    rect=box.int_rect,
                    crop_shape=(img.height, img.width),
                    markdown=md,
                    md_path=md_path,
                )
            )
        return regions

    def process_page(
            self,
            image: Image.Image,
            page_index: int = 0,
            request_id: Optional[str] = None,
    ) -> PageResult:
        """Process a single image end-to-end (used outside the scheduler path)."""
        st = time.perf_counter()
        self._ensure_output_dir()
        image = image.convert("RGB")
        w, h = image.size

        # 1) layout detection（内部自动 resize）
        layouts = self.layout.detect([image])[0]
        # 2) 裁剪（filter + crop 一步完成）
        crops = self._crop_page(image, layouts)
        # 3) 按 label 分桶 → VL batch
        items = [(img, box.label_name) for img, box in crops]
        markdowns = self.vl.recognize_grouped(items)
        md_lookup: dict[tuple[int, int], str] = {
            (0, c_idx): md for c_idx, md in enumerate(markdowns)
        }

        # 4) 落盘 text_result/request_id/page_X_box_Y.md
        # request_id 隔离不同请求的输出，避免并发 path 冲突
        if request_id is None:
            request_id = uuid.uuid4().hex[:8]
        req_dir = self._safe_request_dir(request_id)
        regions = self._write_regions(req_dir, page_index, crops, md_lookup, 0)

        return PageResult(
            page_index=page_index,
            width=w,
            height=h,
            elapsed_seconds=time.perf_counter() - st,
            regions=regions,
        )

    def process_batch(
            self,
            jobs: list[Job],
            cancelled_vouchers: Optional[set[str]] = None,
            cancelled_requests: Optional[set[str]] = None,
    ) -> list[PageResult]:
        """一次处理 N 张图（跨用户共享 layout + VL 推理）。

        流水线：

        1. layout detection 一次喂所有图（PP-DocLayoutV3 内部 batch）
        2. 每图独立 filter + crop
        3. 跨图按 label 分桶 → VL batch（跨用户共享 GPU）
        4. markdown 落盘到各自 request_id/ 子目录，返回 PageResult

        两层取消过滤（合并 ``cancelled_vouchers`` 和 ``cancelled_requests``）：

        - 入口（防御 take_batch 后才 cancel 的情况）
        - crop 之后、VL 之前（处理 "layout 完成了但还没 VL" 的 crops）

        命中过滤的 job 返回 ``cancelled=True`` 的空 PageResult，不调 VL、不写盘。
        """
        if not jobs:
            return []
        if cancelled_vouchers is None:
            cancelled_vouchers = set()
        if cancelled_requests is None:
            cancelled_requests = set()

        def _is_cancelled(j: "Job") -> bool:
            if j.voucher_id and j.voucher_id in cancelled_vouchers:
                return True
            if j.request_id and j.request_id in cancelled_requests:
                return True
            return False

        st = time.perf_counter()
        self._ensure_output_dir()

        # 防御：request_id 缺失 / 非法会写到 output_dir 根目录或越界，污染文件
        for j in jobs:
            if not _REQUEST_ID_RE.fullmatch(j.request_id or ""):
                raise ValueError(
                    f"Job.request_id must be 1-64 chars of [A-Za-z0-9_-]; "
                    f"got {j.request_id!r}"
                )

        n = len(jobs)

        # --- 第 1 次过滤：入口 --- 拒掉 voucher / request 已取消的
        # 保留原 jobs 列表的索引位置，结果按原顺序回填
        active1_idx: list[int] = [
            i for i, j in enumerate(jobs) if not _is_cancelled(j)
        ]
        if not active1_idx:
            return [self._empty_page_result(j) for j in jobs]

        active1_jobs = [jobs[i] for i in active1_idx]

        # 1) layout detection 整批一次
        images_rgb: list[Image.Image] = [j.image.convert("RGB") for j in active1_jobs]
        layouts_per_page = self.layout.detect(images_rgb)

        # 2) 每图独立 filter + crop（page_size 让 unclip clamp 到图内）
        crops_per_page: list[list[tuple[Image.Image, LayoutBox]]] = [
            self._crop_page(image, layouts)
            for image, layouts in zip(images_rgb, layouts_per_page)
        ]

        # --- 第 2 次过滤：crop 之后、VL 之前 ---
        # cancel_voucher / cancel_request 在 process_batch 运行期间被另一个线程
        # （GIL 切换）调时，这次过滤能捕到（在 entry 和 crop 之间发生了 cancel）
        active2_idx_in_a1: list[int] = [
            i for i, j in enumerate(active1_jobs) if not _is_cancelled(j)
        ]
        if not active2_idx_in_a1:
            # 全部被第 2 次过滤掉
            results: list[Optional[PageResult]] = [None] * n
            for ai, j in enumerate(active1_jobs):
                results[active1_idx[ai]] = self._empty_page_result(j)
            for idx in range(n):
                if results[idx] is None:
                    results[idx] = self._empty_page_result(jobs[idx])
            return results  # type: ignore[return-value]

        # 缩到 active2 的子集
        active2_jobs = [active1_jobs[i] for i in active2_idx_in_a1]
        active2_in_orig_idx = [active1_idx[i] for i in active2_idx_in_a1]
        crops2_per_page = [crops_per_page[i] for i in active2_idx_in_a1]
        images2_rgb = [images_rgb[i] for i in active2_idx_in_a1]

        # 3) 跨图按 label 桶聚合（同一个 bucket 一次 VL forward）
        md_lookup = self._run_vl_buckets(crops2_per_page)

        # 4) 写文件 + 构造 active2 的 PageResult
        active2_results: list[PageResult] = []
        for p_idx, (job, crops) in enumerate(zip(active2_jobs, crops2_per_page)):
            image = images2_rgb[p_idx]
            req_dir = self._safe_request_dir(job.request_id)
            regions = self._write_regions(
                req_dir, job.page_index, crops, md_lookup, p_idx
            )
            active2_results.append(
                PageResult(
                    page_index=job.page_index,
                    width=image.width,
                    height=image.height,
                    elapsed_seconds=time.perf_counter() - st,  # 本页完成时刻的墙钟耗时（L-09）
                    regions=regions,
                )
            )

        # 回填到原 jobs 顺序：active2 的塞回原位置，缺口（被第 1 次过滤的）填空
        results2: list[Optional[PageResult]] = [None] * n
        for ai2, pr in enumerate(active2_results):
            results2[active2_in_orig_idx[ai2]] = pr
        for idx in range(n):
            if results2[idx] is None:
                results2[idx] = self._empty_page_result(jobs[idx])
        return results2  # type: ignore[return-value]

    def _run_vl_buckets(
            self,
            crops2_per_page: list[list[tuple[Image.Image, LayoutBox]]],
    ) -> dict[tuple[int, int], str]:
        """Group crops by label, run VL batch per group, return ``md_lookup``.

        ``md_lookup`` maps ``(page_idx_in_batch, crop_idx_in_page)`` → markdown.
        """
        # bucket: label_name -> list of (page_idx, crop_idx, crop_img)
        buckets: dict[str, list[tuple[int, int, Image.Image]]] = {}
        for p_idx, crops in enumerate(crops2_per_page):
            for c_idx, (img, box) in enumerate(crops):
                buckets.setdefault(box.label_name, []).append((p_idx, c_idx, img))

        md_lookup: dict[tuple[int, int], str] = {}
        for label, items in buckets.items():
            imgs = [it[2] for it in items]
            texts = self.vl.recognize_batch(imgs, label)
            for (p_idx, c_idx, _), md in zip(items, texts):
                md_lookup[(p_idx, c_idx)] = md
        return md_lookup


# ─────────────────────────────────────────────────────────────────────────────
# 6) Pipeline pool —— asyncio.Queue 模式，支持 N 路并发
# ─────────────────────────────────────────────────────────────────────────────
class PipelinePool:
    """asyncio.Queue 实现的 pipeline 池。``lease()`` 上下文管理器借出 / 归还。"""

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
    """在 PipelinePool 之上做的调度器，按 max-min fairness 拼批。

    上游：任意线程/协程 ``submit(request_id, page_idx, image)`` → 返回
    ``Future[PageResult]``。

    下游：``n_workers`` 个 worker 协程循环 —

    * 抢 1 个 job 当头（按 fair 选 user）
    * 继续等 ``flush_ms`` 时间，凑够 ``max_batch`` 个（按 fair 选 user）
    * 从 pool 借一个 pipeline
    * 调 ``pipeline.process_batch([...])`` 一次处理整批（跨用户共享 layout + VL）
    * 把结果 ``set_result`` 回每个 Job 的 Future
    * 更新每个 user 的"已完成数"用于下次公平分配

    公平调度策略 (max-min fairness)：

    * 每个 user 独立 ``_user_pending: deque[Job]`` 队列
    * 维护 ``_user_completed: dict[user, int]`` 跟踪每个 user 已完成的 job 数
    * 拼 batch 时按 ``(completed, 首次出现顺序)`` 排序 user，completed 小的优先
    * 第一轮：每个 user 各拿 1 张
    * 第二轮：继续从 completed 最小的 user 拿，填满 max_batch
    * 效果：长任务（PDF 多页）不会独占 batch，新来的小任务能立刻被分到

    用户在线/离线检测 + 主动释放逻辑在调用方实现（v0.6 起由 app.py 自管）。
    scheduler 只在 ``close()`` 时一次性取消所有 pending job。
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
        # per-user 独立队列 + 完成数跟踪
        self._user_pending: dict[str, "collections.deque[Job]"] = {}
        self._user_order: list[str] = []  # user 首次出现顺序（断 ties）
        self._user_completed: dict[str, int] = {}
        self._user_last_active: dict[str, float] = {}  # 单调钟最近活动时间，用于闲置清理
        self._pending_count = 0
        self._lock = asyncio.Lock()
        self._wakeup = asyncio.Event()
        self._closed = False
        self._worker_tasks: list[asyncio.Task] = []
        # voucher 取消追踪：cancel_voucher 把 voucher 加进来；worker 把 set 传给
        # pipeline.process_batch 做 entry + crop 之后的两层过滤。in-flight 的 crops
        # 在 crop 之后、VL 之前被丢掉；VL 已经发起的那次 forward 跑完但结果不写盘。
        # 该 set 只增不删（voucher 短，几千条也才 KB 级，不回收）—— 但当超过
        # _CANCELLED_TRACK_SOFT_LIMIT 时触发一次 _gc_cancelled_tracking 清理。
        self._cancelled_vouchers: set[str] = set()
        # request 取消追踪：同上，但按 request_id（也就是 user_id / 单次 upload）。
        # 跟 voucher 取消是两套独立维度 —— 前者是"用户离线全部清"，后者是"前端点
        # remove 只清这一条"。同时在两层过滤里 union 取并集。
        self._cancelled_requests: set[str] = set()

    async def start(self) -> None:
        logger.info(
            "BatchScheduler starting %d worker(s), max_batch=%d, flush=%.0fms "
            "(max-min fair)",
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
        # 关闭时取消所有 pending 任务
        await self._cancel_all_pending("scheduler closed")
        logger.info("BatchScheduler closed")

    def submit(
            self,
            request_id: str,
            page_index: int,
            image: Image.Image,
            voucher_id: str = "",
    ) -> Job:
        """提交一个 page 任务，返回 Job（``Job.fut`` 是 ``asyncio.Future[PageResult]``）。

        ``voucher_id`` —— 会话级 ID（前端 alive_check 用）。空字符串表示不绑定
        voucher（旧代码兼容）。``cancel_voucher(voucher_id)`` 会把这个 voucher
        的所有 pending job fut 取消，并把 voucher 标记到 ``_cancelled_vouchers``，
        让正在 in-flight 的同 voucher job 在 process_batch 里被两层过滤掉。

        必须在 event loop 线程内调用（因为要创建 Future）。
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[PageResult] = loop.create_future()
        job = Job(
            request_id=request_id,
            page_index=page_index,
            image=image,
            fut=fut,
            voucher_id=voucher_id,
        )
        # 放进 per-user 队列（首次出现的 user 加进 _user_order）
        if request_id not in self._user_pending:
            self._user_pending[request_id] = collections.deque()
            self._user_order.append(request_id)
        self._user_pending[request_id].append(job)
        self._user_last_active[request_id] = time.monotonic()
        self._pending_count += 1
        self._wakeup.set()
        return job

    async def cancel_voucher(self, voucher_id: str) -> int:
        """一次性取消 voucher 的所有 pending + in-flight 工作。

        行为：

        1. 遍历所有 user 的 pending 队列，``fut.cancel()`` 该 voucher 的 job，从队列里移除
        2. 把 voucher 加进 ``_cancelled_vouchers``；worker 在调 process_batch 时把这个
           set 传进去，pipeline 在入口 + crop 之后两层过滤掉同 voucher 的 job
        3. ``_wakeup.set()`` 唤醒 worker，让 pending 减少后立刻看到空位

        返回被取消的 pending job 数。
        """
        if not voucher_id:
            return 0
        cancelled_count = 0
        async with self._lock:
            cancelled_count = self._cancel_in_pending(
                lambda job: job.voucher_id == voucher_id
            )
            # 标记 voucher 已取消（worker 下一批 process_batch 入口 + crop 之后会看到）
            self._cancelled_vouchers.add(voucher_id)
            self._maybe_gc_cancelled_tracking()
        # wakeup 在 lock 外（避免锁内 await 死锁；_wakeup.set() 是非阻塞的，但保持锁外是好习惯）
        self._wakeup.set()
        if cancelled_count:
            logger.info(
                "scheduler cancelled voucher=%s, %d pending job(s); "
                "in-flight will be dropped at next filter",
                voucher_id, cancelled_count,
            )
        else:
            logger.info(
                "scheduler marked voucher=%s as cancelled (no pending; "
                "in-flight will be dropped at next filter)",
                voucher_id,
            )
        return cancelled_count

    async def cancel_request(self, request_id: str) -> int:
        """一次性取消 request_id 的所有 pending + in-flight 工作。

        跟 ``cancel_voucher`` 是平行的两个维度：voucher 是"用户离线全部清"，
        request 是"前端点 remove 只清这一条"。

        返回被取消的 pending job 数。
        """
        if not request_id:
            return 0
        cancelled_count = 0
        async with self._lock:
            cancelled_count = self._cancel_in_pending(
                lambda job: job.request_id == request_id
            )
            # 标记 request 已取消（worker 下一批 process_batch 入口 + crop 之后会看到）
            self._cancelled_requests.add(request_id)
            self._maybe_gc_cancelled_tracking()
        self._wakeup.set()
        if cancelled_count:
            logger.info(
                "scheduler cancelled request=%s, %d pending job(s); "
                "in-flight will be dropped at next filter",
                request_id, cancelled_count,
            )
        else:
            logger.info(
                "scheduler marked request=%s as cancelled (no pending; "
                "in-flight will be dropped at next filter)",
                request_id,
            )
        return cancelled_count

    def pending_size(self) -> int:
        return self._pending_count

    def _cancel_in_pending(self, match_fn: Callable[[Job], bool]) -> int:
        """Cancel & remove pending jobs matching ``match_fn``. Caller must hold ``_lock``.

        Returns the number of cancelled pending jobs.
        """
        cancelled_count = 0
        for u, q in list(self._user_pending.items()):
            kept: list[Job] = []
            user_cancelled = 0
            while q:
                job = q.popleft()
                if match_fn(job) and not job.fut.done():
                    job.fut.cancel("cancelled by scheduler")
                    user_cancelled += 1
                else:
                    kept.append(job)
            # 没匹配上的放回原队列（保持顺序）
            for j in reversed(kept):
                q.appendleft(j)
            # 队列被清空就删 user 键
            if not q:
                self._user_pending.pop(u, None)
            cancelled_count += user_cancelled
        if cancelled_count:
            self._pending_count = max(0, self._pending_count - cancelled_count)
        return cancelled_count

    def _prune_idle_users(self, now: float) -> None:
        """清理闲置用户书签，防止 _user_order / _user_completed 无限增长（H-05）。

        调用者必须持有 ``self._lock``。仅清理"队列已空 + 超过空闲窗口"的用户；
        有 pending 或刚活动过的用户不受影响。
        """
        for u in list(self._user_pending.keys()):
            if self._user_pending[u]:
                continue
            last = self._user_last_active.get(u)
            if last is not None and now - last > _USER_IDLE_PRUNE_S:
                del self._user_pending[u]
                self._user_completed.pop(u, None)
                self._user_last_active.pop(u, None)
                if u in self._user_order:
                    self._user_order.remove(u)
                logger.info("scheduler pruned idle user=%s", u)

    def _maybe_gc_cancelled_tracking(self) -> None:
        """Soft-GC cancelled tracking sets when they grow too large.

        Removes entries for users that have no pending jobs and no completions
        in the current ``_user_completed`` snapshot. Caller must hold ``_lock``.

        The cancelled-tracking sets only grow in the original design because
        vouchers/requests are short strings; in practice a few thousand entries
        is only KB-level memory. But for long-running services with many
        distinct users, we cap the growth here.
        """
        if (
                len(self._cancelled_vouchers) + len(self._cancelled_requests)
                < _CANCELLED_TRACK_SOFT_LIMIT
        ):
            return
        # Conservative GC: only drop cancelled entries whose user_id is no longer
        # active (not in _user_pending and not recently completed).
        active_users = set(self._user_pending.keys()) | set(self._user_completed.keys())
        before_r = len(self._cancelled_requests)
        # We can only safely drop request-level cancellations for users no
        # longer active. Voucher-level cancellations are kept as-is (they may
        # be shared across many requests and we can't tell which are stale).
        self._cancelled_requests = {
            r for r in self._cancelled_requests if r in active_users
        }
        gc_r = before_r - len(self._cancelled_requests)
        if gc_r:
            logger.info(
                "scheduler GC: removed %d request(s) from cancelled tracking "
                "sets (vouchers retained by design)",
                gc_r,
            )

    async def _cancel_all_pending(self, reason: str) -> None:
        """一次性取消所有 user 的 pending job（仅在 close() 时调用）。

        取消已经 submit 但 worker 还没 take 走的 job fut —— 调用方 catch 到
        CancelledError 自然能感知。in-flight（worker 已 take 走）的 job 不动。
        """
        total = 0
        for u, q in list(self._user_pending.items()):
            n = 0
            while q:
                job = q.pop()
                if not job.fut.done():
                    job.fut.cancel(reason)
                n += 1
            self._pending_count = max(0, self._pending_count - n)
            total += n
            if n:
                logger.info(
                    "scheduler cancelled user=%s, %d pending job(s) (%s)",
                    u, n, reason,
                )
        if total:
            self._wakeup.set()
            logger.info(
                "scheduler cancelled all pending: %d total job(s) (%s)",
                total, reason,
            )

    def _take_fair_batch(self, max_n: int) -> list[Job]:
        """max-min fair 选 batch：优先选 '已完成数最少' 的 user。

        调用者必须持有 ``self._lock``。
        """
        # 顺带清理闲置用户书签，防 _user_order/_user_completed 无限增长（H-05）
        self._prune_idle_users(time.monotonic())
        if not self._user_pending or max_n <= 0:
            return []

        # 排序 key：(已完成数, 首次出现顺序) — 已完成少的 + 来得早的优先
        order_idx = {u: i for i, u in enumerate(self._user_order)}

        def _sort_key(u: str) -> tuple[int, int]:
            return (self._user_completed.get(u, 0), order_idx.get(u, 0))

        # 把 active users 排一次序，后续轮次复用（避免 O(n²) 重复排序）
        active_users = [
            u for u in self._user_order if self._user_pending.get(u)
        ]
        if not active_users:
            return []
        active_users.sort(key=_sort_key)

        result: list[Job] = []

        # 第一轮：每个 user 各拿 1 张
        for u in active_users:
            q = self._user_pending[u]
            if not q:
                continue
            result.append(q.popleft())
            if len(result) >= max_n:
                break

        # 第二轮：填满到 max_n（按已排序的 user 顺序继续拿）
        if len(result) < max_n:
            picked_any = True
            while len(result) < max_n and picked_any:
                picked_any = False
                for u in active_users:
                    q = self._user_pending[u]
                    if not q:
                        continue
                    result.append(q.popleft())
                    picked_any = True
                    if len(result) >= max_n:
                        break

        # 清理空 user 队列（保留 _user_order 顺序历史）
        for u in list(self._user_pending.keys()):
            if not self._user_pending[u]:
                del self._user_pending[u]

        self._pending_count = max(0, self._pending_count - len(result))
        return result

    async def _worker_loop(self, wid: int) -> None:
        """worker 主循环：等至少 1 个 job → 按 fair 凑 batch → 借 pipeline → 跑批 → 分发。

        process_batch 内部抛出的异常不会让 worker 死掉 —— 异常会通过
        ``job.fut.set_exception`` 传给调用方，worker 继续下一轮。
        """
        logger.info("scheduler worker[%d] started", wid)
        while not self._closed:
            # 1) 等到至少 1 个 job
            while self._pending_count == 0 and not self._closed:
                self._wakeup.clear()
                await self._wakeup.wait()
            if self._closed:
                break
            if self._pending_count == 0:
                continue

            # 2) 抢 1 个 job 当头（fair 选）
            async with self._lock:
                if self._pending_count == 0:
                    continue
                batch = self._take_fair_batch(1)
                if not batch:
                    continue
                if self._pending_count == 0:
                    self._wakeup.clear()

            # 3) 凑 batch（max_batch 或 flush_seconds 超时；每轮都 fair 选）
            deadline = time.perf_counter() + self.flush_seconds
            while len(batch) < self.max_batch and self._pending_count > 0:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                async with self._lock:
                    more = self._take_fair_batch(self.max_batch - len(batch))
                    batch.extend(more)
                    if self._pending_count == 0:
                        self._wakeup.clear()
                if len(batch) >= self.max_batch:
                    break

            # 4) 借 pipeline + 跑批 + 分发
            await self._run_batch(wid, batch)

        logger.info("scheduler worker[%d] exited", wid)

    async def _run_batch(self, wid: int, batch: list[Job]) -> None:
        """Lease a pipeline, run ``process_batch``, dispatch results.

        Any exception from ``process_batch`` is propagated to each job's future
        (via ``set_exception``) so the worker loop stays alive.
        """
        from starlette.concurrency import run_in_threadpool

        t0 = time.perf_counter()
        users_in_batch = sorted({j.request_id for j in batch})
        user_in_batch_count: dict[str, int] = {}
        for j in batch:
            user_in_batch_count[j.request_id] = user_in_batch_count.get(j.request_id, 0) + 1

        results: Optional[list[PageResult]] = None
        try:
            async with self.pool.lease() as pipe:
                wait = time.perf_counter() - t0
                logger.info(
                    "scheduler[%d] lease: waited %.3fs, free=%d/%d, batch=%d "
                    "(%d user %s), per_user=%s",
                    wid, wait, self.pool.qsize(), self.pool.size,
                    len(batch), len(users_in_batch), users_in_batch,
                    user_in_batch_count,
                )
                # 把 _cancelled_vouchers + _cancelled_requests 都传给
                # pipeline.process_batch，让 entry + crop 之后两层过滤掉
                # voucher / request 已取消的 job
                results = await run_in_threadpool(
                    pipe.process_batch, batch,
                    self._cancelled_vouchers, self._cancelled_requests,
                )
        except Exception as exc:
            # process_batch 抛了：把异常传给每个 job，不让 worker 死掉
            logger.exception(
                "scheduler[%d] process_batch crashed: %s", wid, exc,
            )
            for job in batch:
                if not job.fut.done():
                    job.fut.set_exception(exc)
            return

        dt = time.perf_counter() - t0
        # 5) 更新每个 user 的已完成数（用于下次 fair 排序）
        for u, c in user_in_batch_count.items():
            self._user_completed[u] = self._user_completed.get(u, 0) + c
        now = time.monotonic()
        for u in user_in_batch_count:
            self._user_last_active[u] = now
        # 6) 分发结果
        for job, page in zip(batch, results or []):
            if not job.fut.done():
                job.fut.set_result(page)
        logger.info(
            "scheduler[%d] done: batch=%d in %.2fs, free=%d/%d, user_completed=%s",
            wid, len(batch), dt, self.pool.qsize(), self.pool.size,
            {u: self._user_completed.get(u, 0) for u in users_in_batch},
        )


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
    """Build an ``OCRPipeline`` with default model paths and auto device selection."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return OCRPipeline(
        layout_model_path=LAYOUT_MODEL_PATH,
        vl_model_path=VL_MODEL_PATH,
        device=device,
        output_dir=output_dir,
    )
