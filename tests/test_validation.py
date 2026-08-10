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

from fastapi import HTTPException

import app as app_module
from core_pipeline import BatchScheduler


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


if __name__ == "__main__":
    unittest.main()
