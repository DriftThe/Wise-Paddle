"""最小单元测试：用户 ID 校验与像素预算（L-06）。

运行方式（项目根目录）：
    .venv\\Scripts\\python.exe -m unittest tests.test_validation
"""

import unittest

from fastapi import HTTPException

import app as app_module


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


if __name__ == "__main__":
    unittest.main()
