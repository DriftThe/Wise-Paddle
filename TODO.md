# Wise-Paddle 修复 TODO

> 依据 [CODE_REVIEW.md](CODE_REVIEW.md) 整理，按优先级逐项完成；每完成一项勾选并同步提交。

## 高优先级

- [x] H-01 `user_id` 严格校验 + 写盘路径防穿越（服务端生成/白名单 + resolve 校验）
- [x] H-02 PDF 先查页数/加密状态再渲染，超限即 413
- [x] H-03 `/ocr/base64` 与上传入口增加 base64 长度 + 像素预算限制
- [x] H-04 心跳 TTL 加长 + 前端隐藏页 beacon 保活，避免任务被误杀
- [x] H-05 scheduler `_user_order` / `_user_completed` 空闲用户清理

## 中优先级

- [x] M-01 CORS 改用 `allow_origin_regex`（端口通配符不再失效）
- [x] M-02 PNG 编码移入线程池，避免阻塞事件循环
- [x] M-03 README / ARCHITECTURE 与代码对齐（移除 `/api/release` 旧描述）
- [x] M-04 user_id 统一校验（取消/进度接口同口径）
- [x] M-05 全局异常响应不再泄露内部错误
- [x] M-06 `_kick_dead_user_once` TTL 双检逻辑修正
- [x] M-07 `_unique_path` 写入竞态（文件名加 uuid 后缀）
- [x] M-08 前端 PDF 预览页数上限
- [x] M-09 requirements.txt 移除重复的 opencv-python
- [x] M-10 `.env` 移出 Git 跟踪
- [x] M-11 前端超时/清空队列时通知服务端取消
- [x] M-12 pdf-status 支持 `since` 增量返回
- [x] M-13 每页完成后释放 `pil_pages` 条目

## 低优先级

- [x] L-01 版本号统一为 0.6.0
- [x] L-02 `Image.BILINEAR` → `Image.Resampling.BILINEAR`
- [x] L-03 MD 文件名 label 消毒
- [x] L-05 日志中文件名转义（防注入）
- [x] L-06 为 `_sanitize_user_id` 增加最小单测
- [ ] L-07 运行产物忽略（已在基线提交中完成）
- [x] L-08 `/alive` 拒绝空 voucher
- [x] L-09 单页 `elapsed_seconds` 独立计时
- [x] L-10 每页仅一次 RGB 转换
- [x] L-11 取消追踪 GC 日志修正
- [x] L-12 加密 PDF 返回 400 友好提示
- [x] L-13 上传图片处理 EXIF 方向

## 说明

- L-04（`md_path` 路径泄露）：保留 API 字段但改返回相对路径；避免破坏前端兼容。
- M-04 完整鉴权（会话/Token）不在本次范围，仅做 ID 校验与一致性加固。
