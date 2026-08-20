"""最小单元测试：用户 ID 校验、像素预算、日志转义、PDF 预检与闲置清理（L-06）。

运行方式（项目根目录）：
    .venv\\Scripts\\python.exe -m unittest tests.test_validation
"""

import collections
import io
import struct
import time
import unittest
import zlib
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from PIL import Image

import app as app_module
from core_pipeline import BatchScheduler, Job, LayoutBox, OCRPipeline, PageResult


def _png_with_dimensions(width: int, height: int) -> bytes:
    """构造只含 IHDR + 极小 IDAT 的 PNG（未真正编码像素），用于头部尺寸校验测试。"""

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(b"\x00"))
    )


class _StubPool:
    """BatchScheduler 单测用的最小池桩（只提供 size 属性）。"""

    size = 1


class TestSanitizeUserId(unittest.TestCase):
    def test_valid_ids_unchanged(self):
        for v in ("web-abc123", "abc_123", "A-1_b", "12345678"):
            self.assertEqual(app_module._sanitize_user_id(v), v)

    def test_empty_generates_uuid(self):
        uid = app_module._sanitize_user_id("")
        self.assertIsInstance(uid, str)
        self.assertEqual(len(uid), 8)

    def test_path_traversal_rejected(self):
        for v in ("../../evil", "a/b", "a\\b", "..", ".", "a b", "中文"):
            with self.assertRaises(HTTPException):
                app_module._sanitize_user_id(v)

    def test_too_long_rejected(self):
        with self.assertRaises(HTTPException):
            app_module._sanitize_user_id("x" * 65)

    def test_valid_voucher_ids(self):
        for v in ("", "v1", "voucher-a", "a" * 128):
            self.assertEqual(app_module._sanitize_voucher_id(v), v)

    def test_invalid_voucher_ids_rejected(self):
        for v in ("a b", "a/b", "a\\b", "中文", "a" * 129, "\n"):
            with self.assertRaises(HTTPException):
                app_module._sanitize_voucher_id(v)



class TestPixelBudget(unittest.TestCase):
    def test_within_budget(self):
        app_module._check_pixel_budget(2000, 2000)  # 4MP < 50MP 默认上限

    def test_over_budget(self):
        with self.assertRaises(HTTPException):
            app_module._check_pixel_budget(100_000, 100_000)

    def test_invalid_dimensions(self):
        with self.assertRaises(HTTPException):
            app_module._check_pixel_budget(0, 100)


class TestDecodeRawImage(unittest.TestCase):
    def test_accepts_small_image(self):
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(buf, format="PNG")
        rgb = app_module._decode_raw_image(buf.getvalue())
        self.assertEqual(rgb.shape[:2], (8, 8))

    def test_rejects_over_budget_header_before_decode(self):
        # 8000x8000 = 64MP，超过 50MP 默认上限但低于 PIL 炸弹阈值（~89MP），
        # 应在真正解码像素前被头部尺寸检查拒绝
        with self.assertRaises(HTTPException) as ctx:
            app_module._decode_raw_image(_png_with_dimensions(8_000, 8_000))
        self.assertEqual(ctx.exception.status_code, 413)

    def test_rejects_bomb_image(self):
        # 超过 PIL DecompressionBombError 阈值（~178MP）的头部，
        # open() 阶段即被拒，需透传为 413 而不是回退 cv2
        with self.assertRaises(HTTPException) as ctx:
            app_module._decode_raw_image(_png_with_dimensions(100_000, 100_000))
        self.assertEqual(ctx.exception.status_code, 413)


class TestSafeFilename(unittest.TestCase):
    def test_none_or_empty(self):
        self.assertEqual(app_module._safe_filename(None), "<unnamed>")
        self.assertEqual(app_module._safe_filename(""), "<unnamed>")

    def test_control_chars_escaped(self):
        self.assertEqual(
            app_module._safe_filename("a\nb\rc\td\x1b[e.txt"),
            "a\\x0ab\\x0dc\\x09d\\x1b[e.txt",
        )

    def test_normal_name_kept(self):
        self.assertEqual(app_module._safe_filename("报告.pdf"), "报告.pdf")


class TestDecodePdfPreRenderCheck(unittest.TestCase):
    def test_oversized_page_rejected_before_render(self):
        import pymupdf

        doc = pymupdf.open()
        # 10000pt x 10000pt @200DPI ≈ 27778^2 px ≈ 7.7 亿像素，远超 50MP 上限
        doc.new_page(width=10_000, height=10_000)
        raw = doc.tobytes()
        doc.close()
        with self.assertRaises(ValueError) as ctx:
            app_module._decode_pdf_to_pil_pages(raw, dpi=200.0, max_pixels=50_000_000)
        self.assertIn("像素数超过上限", str(ctx.exception))

    def test_normal_page_renders(self):
        import pymupdf

        doc = pymupdf.open()
        doc.new_page(width=595, height=842)  # A4 @72dpi
        raw = doc.tobytes()
        doc.close()
        pages = app_module._decode_pdf_to_pil_pages(
            raw, dpi=72.0, max_pixels=50_000_000
        )
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].size, (595, 842))


class TestPruneIdleUsers(unittest.TestCase):
    def setUp(self):
        self.sched = BatchScheduler(_StubPool(), n_workers=1)

    def test_prune_removes_idle_keeps_active(self):
        now = time.monotonic()
        self.sched._user_order = ["u1", "u2", "u3"]
        self.sched._user_pending = {"u2": collections.deque([object()])}
        self.sched._user_completed = {"u1": 5, "u2": 3, "u3": 2}
        self.sched._user_last_active = {
            "u1": now - 1_000.0,
            "u2": now,
            "u3": now - 1_000.0,
        }
        self.sched._prune_idle_users(now)
        self.assertEqual(self.sched._user_order, ["u2"])
        self.assertEqual(self.sched._user_completed, {"u2": 3})
        self.assertEqual(set(self.sched._user_last_active), {"u2"})

    def test_recent_user_not_pruned(self):
        now = time.monotonic()
        self.sched._user_order = ["u1"]
        self.sched._user_completed = {"u1": 1}
        self.sched._user_last_active = {"u1": now - 60.0}  # 空闲窗口（600s）内
        self.sched._prune_idle_users(now)
        self.assertEqual(self.sched._user_order, ["u1"])
        self.assertEqual(self.sched._user_completed, {"u1": 1})


class TestReviveVoucher(unittest.TestCase):
    """NEW-01 复活语义：cancel_voucher 后 revive_voucher 应移除取消标记。"""

    def setUp(self):
        self.sched = BatchScheduler(_StubPool(), n_workers=1)

    def test_cancel_then_revive_removes_marker(self):
        import asyncio

        async def scenario():
            # 1) 取消 voucher → 标记进入 _cancelled_vouchers
            await self.sched.cancel_voucher("v1")
            self.assertIn("v1", self.sched._cancelled_vouchers)
            # 2) 复活 → 标记移除
            removed = await self.sched.revive_voucher("v1")
            self.assertEqual(removed, 1)
            self.assertNotIn("v1", self.sched._cancelled_vouchers)

        asyncio.run(scenario())

    def test_revive_unknown_voucher_is_idempotent(self):
        import asyncio

        async def scenario():
            self.assertEqual(await self.sched.revive_voucher("ghost"), 0)
            self.assertNotIn("ghost", self.sched._cancelled_vouchers)
            # 空串也不报错
            self.assertEqual(await self.sched.revive_voucher(""), 0)

        asyncio.run(scenario())

    def test_revive_does_not_touch_request_cancellations(self):
        import asyncio

        async def scenario():
            await self.sched.cancel_request("req-1")
            await self.sched.revive_voucher("v2")
            self.assertIn("req-1", self.sched._cancelled_requests)

        asyncio.run(scenario())



class TestCancelJobsAndGenerations(unittest.TestCase):
    """NEW-03/H-1/H-2 回归：精确取消 Job + 新代数不被旧取消标记污染。"""

    def setUp(self):
        self.sched = BatchScheduler(_StubPool(), n_workers=1)

    def test_cancel_jobs_removes_cancelled_future_from_queue(self):
        import asyncio

        async def scenario():
            j = self.sched.submit("u1", 0, None, voucher_id="")
            j.fut.cancel("direct cancel")
            removed = await self.sched.cancel_jobs([j])
            self.assertEqual(removed, 1)
            self.assertEqual(self.sched.pending_size(), 0)
            self.assertNotIn("u1", self.sched._user_pending)

        asyncio.run(scenario())

    def test_new_generation_after_cancel_not_poisoned(self):
        import asyncio

        async def scenario():
            await self.sched.cancel_request("u1")
            gen = self.sched.next_request_generation("u1")
            j = self.sched.submit("u1", 0, None, voucher_id="", generation=gen)
            cancelled_gen = self.sched._cancelled_requests.get("u1", 0)
            self.assertGreater(gen, cancelled_gen)
            self.assertEqual(j.generation, gen)
            self.assertFalse(j.generation <= cancelled_gen)

        asyncio.run(scenario())

    def test_cancel_jobs_marks_old_generation_but_not_new(self):
        import asyncio

        async def scenario():
            old_gen = self.sched.next_request_generation("u1")
            old_job = self.sched.submit("u1", 0, None, voucher_id="", generation=old_gen)
            await self.sched.cancel_jobs([old_job])
            new_gen = self.sched.next_request_generation("u1")
            new_job = self.sched.submit("u1", 1, None, voucher_id="", generation=new_gen)
            self.assertEqual(self.sched._cancelled_requests["u1"], old_gen)
            self.assertFalse(new_job.generation <= self.sched._cancelled_requests["u1"])

        asyncio.run(scenario())


class TestVoucherGuard(unittest.TestCase):
    """M-04 归属校验：_register_request_voucher + _check_voucher_guard。"""

    def setUp(self):
        app_module._request_vouchers.clear()
        app_module._request_vouchers_ts.clear()

    def test_mismatch_rejected(self):
        import asyncio

        async def scenario():
            app_module._register_request_voucher("user-1", "voucher-a")
            with self.assertRaises(HTTPException) as ctx:
                await app_module._check_voucher_guard("user-1", "voucher-b")
            self.assertEqual(ctx.exception.status_code, 403)

        asyncio.run(scenario())

    def test_match_allowed(self):
        import asyncio

        async def scenario():
            app_module._register_request_voucher("user-1", "voucher-a")
            # 匹配的 voucher 放行
            await app_module._check_voucher_guard("user-1", "voucher-a")
            # 未登记用户（任务不存在 / 已过期）放行，兼容旧客户端
            await app_module._check_voucher_guard("unknown-user", "")
            # 已登记用户不带 voucher（攻击者绕过路径）→ 403
            with self.assertRaises(HTTPException) as ctx:
                await app_module._check_voucher_guard("user-1", "")
            self.assertEqual(ctx.exception.status_code, 403)

        asyncio.run(scenario())

    def test_register_no_voucher_maps_empty(self):
        import asyncio

        async def scenario():
            app_module._register_request_voucher("user-2", "")
            self.assertEqual(app_module._request_vouchers.get("user-2"), "")
            # 登记为空串 → 任何 voucher（包括空串）都视为不匹配（安全加固）
            with self.assertRaises(HTTPException) as ctx:
                await app_module._check_voucher_guard("user-2", "whatever")
            self.assertEqual(ctx.exception.status_code, 403)
            # 空串调用也不放行，防止“空 voucher 猜 user_id”绕过归属校验
            with self.assertRaises(HTTPException) as ctx2:
                await app_module._check_voucher_guard("user-2", "")
            self.assertEqual(ctx2.exception.status_code, 403)

        asyncio.run(scenario())



    def test_expired_mapping_allows_old_client(self):
        import asyncio

        async def scenario():
            app_module._register_request_voucher("user-3", "voucher-a")
            app_module._request_vouchers_ts["user-3"] = (
                time.monotonic() - app_module._REQUEST_VOUCHER_TTL_S - 10
            )
            # 过期后视为未登记，旧客户端不带 voucher 也放行
            await app_module._check_voucher_guard("user-3", "")

        asyncio.run(scenario())



class TestRunBatchLeaseRace(unittest.TestCase):
    """回归：worker 取消时不能在线程池跑完前提前归还 pipeline。"""

    def test_cancel_does_not_release_pipeline_until_thread_finishes(self):
        import asyncio

        class FakePipe:
            def process_batch(self, jobs, cancelled_vouchers, cancelled_requests):
                time.sleep(0.2)
                return [
                    PageResult(
                        page_index=j.page_index,
                        width=1,
                        height=1,
                        elapsed_seconds=0.0,
                    )
                    for j in jobs
                ]

        class FakePool:
            size = 1

            def __init__(self, pipe):
                self.pipe = pipe
                self.released = False

            def qsize(self):
                return 0

            @asynccontextmanager
            async def lease(self):
                try:
                    yield self.pipe
                finally:
                    self.released = True

        async def scenario():
            pool = FakePool(FakePipe())
            sched = BatchScheduler(pool, n_workers=1)
            job = sched.submit("u1", 0, None, voucher_id="", generation=1)
            t = asyncio.create_task(sched._run_batch(0, [job]))
            await asyncio.sleep(0.05)
            t.cancel()
            # 线程仍在跑，pipeline 不应提前归还
            self.assertFalse(pool.released)
            with self.assertRaises(asyncio.CancelledError):
                await t
            # 取回 job future 的异常，避免 asyncio 报告未检索异常
            with self.assertRaises(asyncio.CancelledError):
                await job.fut
            self.assertTrue(pool.released)

        asyncio.run(scenario())




class _FakeLayout:
    """Fake LayoutDetector: 每张图产出一个 text 框；可注入 detect 阶段的钩子。"""

    def __init__(self, on_detect=None):
        self.on_detect = on_detect

    def detect(self, images):
        if self.on_detect is not None:
            self.on_detect()
        return [
            [LayoutBox(np.array([0, 0, 20, 20], dtype=np.float32), 0, "text", 0.9)]
            for _ in images
        ]


class _FakeBoxFilter:
    """Fake BoxFilter: 不过滤、不扩框。"""

    def filter(self, layouts, page_size=None):
        return layouts


class _FakeCropper:
    """Fake RegionCropper: 固定返回一块 20x20 黑色 RGB。"""

    def crop(self, rgb, box):
        return np.zeros((20, 20, 3), dtype=np.uint8)


class _FakeVL:
    """Fake VLPredictor: 按 (label, 桶内序号) 返回确定性 markdown。"""

    def recognize_batch(self, images, label):
        return [f"md-{label}-{i}" for i in range(len(images))]


class TestProcessBatch(unittest.TestCase):
    """回归：简化后的 process_batch 保持两层取消过滤 + 结果顺序回填。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _pipeline(self, layout_hook=None):
        pipe = OCRPipeline.__new__(OCRPipeline)  # 跳过 __init__（不加载真实模型）
        pipe.output_dir = Path(self._tmp.name)
        pipe.max_regions = 100
        pipe.layout = _FakeLayout(layout_hook)
        pipe.box_filter = _FakeBoxFilter()
        pipe.cropper = _FakeCropper()
        pipe.vl = _FakeVL()
        pipe.output_dir.mkdir(parents=True, exist_ok=True)
        return pipe

    def _job(self, request_id, page_index, fut, voucher_id="", generation=1):
        img = Image.new("RGB", (64, 64), "white")
        return Job(request_id, page_index, img, fut, voucher_id=voucher_id, generation=generation)

    def test_order_preserved_and_markdown_written(self):
        import asyncio

        async def scenario():
            pipe = self._pipeline()
            loop = asyncio.get_running_loop()
            jobs = [
                self._job("req-a", 0, loop.create_future()),
                self._job("req-b", 1, loop.create_future()),
                self._job("req-c", 2, loop.create_future()),
            ]
            results = pipe.process_batch(jobs, set(), {})
            # 结果按输入顺序回填，且各自落盘到对应 request 目录
            self.assertEqual([r.page_index for r in results], [0, 1, 2])
            self.assertEqual([r.cancelled for r in results], [False, False, False])
            for job, r in zip(jobs, results):
                self.assertTrue(r.regions)
                md_files = list((Path(self._tmp.name) / job.request_id).glob("*.md"))
                self.assertEqual(len(md_files), 1)
                self.assertIn(r.regions[0].markdown, md_files[0].read_text(encoding="utf-8"))

        asyncio.run(scenario())

    def test_layer1_voucher_cancelled(self):
        import asyncio

        async def scenario():
            pipe = self._pipeline()
            loop = asyncio.get_running_loop()
            j1 = self._job("r1", 0, loop.create_future(), voucher_id="v1")
            j2 = self._job("r2", 1, loop.create_future(), voucher_id="v2")
            results = pipe.process_batch([j1, j2], cancelled_vouchers={"v1"}, cancelled_requests={})
            self.assertTrue(results[0].cancelled)
            self.assertEqual(results[0].regions, [])
            self.assertFalse(results[1].cancelled)
            self.assertTrue(results[1].regions)

        asyncio.run(scenario())

    def test_layer2_cancel_after_layout(self):
        import asyncio

        async def scenario():
            cancelled = {"v2"}

            def hook():
                cancelled.add("v1")  # layout 检测期间 v1 被取消
            pipe = self._pipeline(layout_hook=hook)
            loop = asyncio.get_running_loop()
            j1 = self._job("r1", 0, loop.create_future(), voucher_id="v1")
            j2 = self._job("r2", 1, loop.create_future(), voucher_id="v2")
            results = pipe.process_batch([j1, j2], cancelled_vouchers=cancelled, cancelled_requests={})
            # j2 在入口被过滤；j1 在 crop 之后、VL 之前被第二层过滤捕获
            self.assertTrue(results[0].cancelled)
            self.assertTrue(results[1].cancelled)

        asyncio.run(scenario())

    def test_request_generation_cancelled(self):
        import asyncio

        async def scenario():
            pipe = self._pipeline()
            loop = asyncio.get_running_loop()
            old = self._job("r1", 0, loop.create_future(), generation=2)
            new = self._job("r1", 1, loop.create_future(), generation=3)
            results = pipe.process_batch([old, new], set(), {"r1": 2})
            # 旧代数（generation <= 2）被取消，新代数不受影响
            self.assertTrue(results[0].cancelled)
            self.assertFalse(results[1].cancelled)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
