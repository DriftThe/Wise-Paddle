// ============================================================================
// Wise-Paddle · 前端逻辑  v0.6.0  (含主动 cancel 路径，2026-07-28)
// ----------------------------------------------------------------------------
// 配套 HTML: static/index.html
//
// 入口顺序（按代码出现顺序）:
//   1. 常量 + PDF.js 初始化
//   2. DOM 引用 + 全局状态
//   3. 工具函数 (uid / showToast / copyText / escape)
//   4. /health 轮询 + 心跳保活
//   5. PDF 占位抽帧
//   6. 拖拽 / 文件输入
//   7. 队列渲染 / 清空 (remove 按钮：pending→仅删；processing→主动 cancel 路径)
//   8. 开始处理 / processOne (给 item 存 userId/ctl/abortCtl/pollAbort 给 §7 用)
//   9. PDF 流式轮询 (pollPdfProgress) 顶部检查 item.pollAbort
//  10. 增量填页 (fillOnePageSlot)
//  11. 占位卡构造 (buildPlaceholderCard)
//  12. 拿到结果后升级 (fillCardWithResult)
//  13. canvas 框 + hover-copy (wirePageCanvas)
//  14. 错误 / 取消状态 (markCardError / markCardCancelled)
//  15. 框绘制 / 命中测试 (hitTestBox / drawBoxes)
//  16. 多结果导航 (result-switcher)
//  17. 清空结果
//
// 主动取消链路: §7 按钮 → fetch /api/cancel/{userId} + abortCtl.abort() + pollAbort=true
//   → 服务端 scheduler.cancel_request(userId) 取消 pending futs
//   → _cancelled_requests 进 process_batch 两层过滤
//   → 单图 handler CancelledError → 200 cancelled；PDF task.cancel() → progress.cancelled=true
//   → 前端 markCardCancelled 标 "已取消" + toast "任务已取消"
// ============================================================================
"use strict";

// ===== 常量 =====
/** PDF.js 版本（与 index.html 里的 <script src> 保持一致） */
const PDFJS_VERSION = "3.11.174";
/** PDF.js CDN base URL */
const PDFJS_CDN_BASE = `https://cdn.jsdelivr.net/npm/pdfjs-dist@${PDFJS_VERSION}/legacy/build`;
/** /health 轮询间隔 */
const HEALTH_POLL_INTERVAL_MS = 1500;
/** 心跳保活间隔 */
const KEEPALIVE_INTERVAL_MS = 5000;
/** PDF 进度轮询间隔 */
const PDF_POLL_INTERVAL_MS = 1500;
/** PDF 进度轮询失败后重试间隔 */
const PDF_POLL_RETRY_MS = 2500;
/** 单图请求总超时（避免挂死） */
const REQUEST_TIMEOUT_MS = 15 * 60 * 1000;
/** waitForLegalVoucher 轮询间隔 */
const VOUCHER_POLL_MS = 50;
/** PDF 占位抽帧最大宽度 */
const PDF_PREVIEW_MAX_WIDTH = 1000;
/** PDF 占位抽帧页数上限（防止超大 PDF 卡死浏览器） */
const MAX_PREVIEW_PAGES = 20;
/** Toast 显示时长 */
const TOAST_DURATION_MS = 1500;

// ===== PDF.js setup =====
if (window.pdfjsLib) {
    pdfjsLib.GlobalWorkerOptions.workerSrc = `${PDFJS_CDN_BASE}/pdf.worker.min.js`;
}

// ===== DOM 引用 + 全局状态 =====
const $ = (sel) => document.querySelector(sel);
const resultList = $("#result-list");
const emptyState = $("#empty");
const toastEl = $("#toast");
const queueSection = $("#queue-section");
const queueEl = $("#queue");
const queueCountEl = $("#queue-count");
const startBtn = $("#start-btn");
const clearQueueBtn = $("#clear-queue-btn");
const clearResultsBtn = $("#clear-results-btn");
const queueStatusEl = $("#queue-status");
const fileInput = $("#file-input");
const resultSwitcher = $("#result-switcher");
const rsTabs = $("#rs-tabs");
const rsPrev = $("#rs-prev");
const rsNext = $("#rs-next");
const rsInfo = $("#rs-info");

/** 当前会话的存活 voucher；空字符串表示尚未从 /health 拿到 */
let aliveVoucher = "";
/** 结果卡片自增序号（用于 tab + 滚动定位） */
let resultSeq = 0;
/** 心跳定时器 ID（只在 voucher 首次拿到后启动一次） */
let keepAliveTimerId = null;

// ===== 队列状态 =====
/**
 * @typedef {Object} QueueItem
 * @property {string} id
 * @property {File} file
 * @property {boolean} isPdf
 * @property {string|null} fileUrl        - 单图的 ObjectURL
 * @property {number} pageCount          - 1 = 单图, N = PDF 页数
 * @property {string} status             - pending | processing | done | error | cancelled
 * @property {Array<{b64:string,w:number,h:number}>} previewPages  - PDF 抽帧占位
 * @property {string} [userId]           - processOne 时分配的 client user_id
 * @property {Object} [ctl]              - 结果卡 controller
 * @property {number} [t0]               - performance.now() 起点
 * @property {AbortController} [abortCtl] - 单图 fetch 的 AbortController
 * @property {boolean} [pollAbort]       - PDF 轮询是否应停止
 */

/** @type {QueueItem[]} */
const queue = [];

// ===== 工具函数 =====
/**
 * 生成一个 8 字符的随机 ID（用于 queue item id 和 client user_id）。
 * @returns {string}
 */
function uid() {
    return Math.random().toString(36).slice(2, 10);
}

/**
 * 在底部 toast 显示一条消息。
 * @param {string} msg
 * @param {boolean} [isErr=false] - true 显示红色错误样式
 */
function showToast(msg, isErr = false) {
    toastEl.textContent = msg;
    toastEl.classList.toggle("err", isErr);
    toastEl.classList.add("show");
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(
        () => toastEl.classList.remove("show"),
        TOAST_DURATION_MS,
    );
}

/**
 * 复制文本到剪贴板。优先用 Clipboard API，失败时回退到 deprecated execCommand。
 * @param {string} text
 * @param {boolean} [silent=false] - true 时不弹 toast
 * @returns {Promise<boolean>} 是否复制成功
 */
async function copyText(text, silent = false) {
    if (!text) {
        if (!silent) showToast("该区域无文字", true);
        return false;
    }
    try {
        await navigator.clipboard.writeText(text);
    } catch (e) {
        // 兼容严格安全的情况: 如 http 协议、沙盒环境等未启用 Clipboard API 的情况
        // execCommand 已废弃，但作为 fallback 仍是最可靠的兜底方案
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        try {
            document.execCommand("copy");
        } catch (copyErr) {
            if (!silent) showToast("复制失败", true);
            ta.remove();
            return false;
        }
        ta.remove();
    }
    if (!silent) showToast(`已复制 ${text.length} 字符`);
    return true;
}

/**
 * HTML-escape a string for safe insertion into textContent / innerHTML.
 * Escapes & < > " ' — sufficient for both element text and attribute context.
 * @param {string} s
 * @returns {string}
 */
function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
        ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]),
    );
}

/**
 * Attribute-escape a string for safe insertion into a double-quoted attribute.
 * Escapes & < > " to cover all attribute-injection vectors (not just quotes).
 * @param {string} s
 * @returns {string}
 */
function escapeAttr(s) {
    return String(s ?? "").replace(/[&<>"]/g, (c) =>
        ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]),
    );
}

// ===== Health polling + KeepAlive =====
/**
 * 轮询 /health：拿到 aliveVoucher 后启动心跳保活。
 * 拿到 voucher 之前每 1.5s 重试一次；拿到后停止重试（保留定时器无意义）。
 */
async function pollHealth() {
    try {
        if (aliveVoucher === "") {
            const r = await fetch("/health");
            if (!r.ok) {
                showToast(`HTTP ${r.status} - /health 异常`, true);
                return;
            }
            const d = await r.json();
            $("#stat-pool").textContent = `${d.pool_free}/${d.pool_size}`;
            $("#stat-pending").textContent = d.scheduler_pending;
            $("#stat-batch").textContent = d.batch_max;
            aliveVoucher = d.alive_voucher;
            // 启动心跳（只在首次拿到 voucher 时启动一次，避免重复 setInterval）
            if (keepAliveTimerId === null) {
                keepAliveTimerId = setInterval(keepAlive, KEEPALIVE_INTERVAL_MS);
                await keepAlive();
            }
        }
    } catch (e) {
        console.warn("[pollHealth] failed:", e);
        showToast("404 - 与主机失去连接", true);
    }
}

/**
 * 发送心跳到 /alive，刷新服务端 voucher TTL。
 * 失败时不弹错误 toast（避免心跳失败时频繁打扰用户），仅 console.warn。
 */
async function keepAlive() {
    if (!aliveVoucher) return;
    try {
        const response = await fetch("/alive", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({aliveVoucher}),
            keepalive: true,
        });
        if (!response.ok) {
            console.warn(`[keepAlive] HTTP ${response.status}`);
        }
    } catch (e) {
        console.warn("[keepAlive] failed:", e);
    }
}

// 启动 /health 轮询
const healthPollTimerId = setInterval(pollHealth, HEALTH_POLL_INTERVAL_MS);
pollHealth();

// 隐藏/恢复标签页时补一次心跳：
// 隐藏时用 sendBeacon（不依赖定时器精度），避免后台节流导致任务被服务端误杀（H-04）
document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") {
        if (aliveVoucher && navigator.sendBeacon) {
            navigator.sendBeacon(
                "/alive",
                new Blob([JSON.stringify({aliveVoucher})], {type: "application/json"}),
            );
        }
    } else if (document.visibilityState === "visible") {
        keepAlive();
    }
});

// ===== PDF 占位抽帧（不占 GPU，几百 ms） =====
/**
 * 把 PDF 文件抽成 N 张 PNG data URL，用于队列缩略图和占位卡。
 * @param {File} pdfFile
 * @param {number} [maxW=1000] - 抽帧最大宽度
 * @returns {Promise<{previews:Array<{b64:string,w:number,h:number}>, total:number}>}
 */
async function renderPdfPages(pdfFile, maxW = PDF_PREVIEW_MAX_WIDTH) {
    if (!window.pdfjsLib) throw new Error("PDF.js 未加载");
    const buf = await pdfFile.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({data: buf}).promise;
    const out = [];
    // 只抽前 MAX_PREVIEW_PAGES 页，避免超大 PDF 把浏览器卡死（M-08）
    const lastPage = Math.min(pdf.numPages, MAX_PREVIEW_PAGES);
    for (let i = 1; i <= lastPage; i++) {
        const page = await pdf.getPage(i);
        const baseViewport = page.getViewport({scale: 1});
        const scale = Math.min(2, maxW / baseViewport.width);
        const viewport = page.getViewport({scale});
        const canvas = document.createElement("canvas");
        canvas.width = Math.floor(viewport.width);
        canvas.height = Math.floor(viewport.height);
        await page.render({canvasContext: canvas.getContext("2d"), viewport}).promise;
        out.push({b64: canvas.toDataURL("image/png"), w: canvas.width, h: canvas.height});
    }
    return {previews: out, total: pdf.numPages};
}

// ===== Drop zone =====
const dropZone = $("#drop-zone");
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", (e) => handleFiles(e.target.files));

/**
 * 处理用户拖入 / 选择的文件列表。PDF 异步抽帧；单图直接生成 ObjectURL。
 * @param {FileList|File[]} fileList
 */
async function handleFiles(fileList) {
    for (const f of fileList) {
        const isPdf = f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf");
        const item = {
            id: uid(),
            file: f,
            isPdf,
            fileUrl: isPdf ? null : URL.createObjectURL(f),
            pageCount: 1,
            status: "pending",
            previewPages: [],
        };
        queue.push(item);
        renderQueue();
        if (isPdf) {
            try {
                const res = await renderPdfPages(f);
                item.previewPages = res.previews;
                item.pageCount = res.total;
                renderQueue();
            } catch (e) {
                console.error("PDF preview failed", e);
                showToast(`PDF 预览失败：${e.message}`, true);
                item.status = "error";
                renderQueue();
            }
        }
    }
    fileInput.value = "";
}

/**
 * 渲染队列 UI。每次 queue 状态变化都全量重渲染（队列通常很短，O(n) 可接受）。
 */
function renderQueue() {
    const pending = queue.filter((q) => q.status === "pending").length;
    const processing = queue.filter((q) => q.status === "processing").length;
    queueCountEl.textContent = String(queue.length);
    if (queue.length === 0) {
        queueSection.style.display = "none";
        return;
    }
    queueSection.style.display = "";
    startBtn.disabled = pending === 0;

    queueEl.innerHTML = queue.map((q) => {
        let thumb;
        let extraTag = "";
        if (q.isPdf) {
            const firstPage = q.previewPages[0];
            const src = firstPage ? firstPage.b64 : "";
            const pagesTag = q.pageCount > 1 ? `<span class="pages-tag">${q.pageCount} 页</span>` : "";
            extraTag = pagesTag;
            thumb = src
                ? `<img src="${escapeAttr(src)}" alt="pdf" />`
                : `<div style="display:flex;align-items:center;justify-content:center;height:100%;font-size:32px;color:var(--warn);">PDF</div>`;
        } else {
            thumb = `<img src="${escapeAttr(q.fileUrl || "")}" alt="" />`;
        }
        const statusLabel = QUEUE_STATUS_LABEL[q.status] || q.status;
        return `
      <div class="queue-item ${escapeAttr(q.status)}" data-id="${escapeAttr(q.id)}">
        ${thumb}
        ${extraTag}
        <span class="badge ${escapeAttr(q.status)}">${escapeHtml(statusLabel)}</span>
        <button class="remove" data-id="${escapeAttr(q.id)}" title="移除" aria-label="移除">×</button>
        <div class="file-name">${escapeHtml(q.file.name)}</div>
      </div>`;
    }).join("");

    queueStatusEl.textContent = processing > 0
        ? `处理中 ${processing} 个文件，剩 ${pending} 个待处理`
        : (pending > 0 ? `${pending} 个待处理` : (queue.length > 0 ? `全部 ${queue.length} 个已完成` : ""));

    queueEl.querySelectorAll(".remove").forEach((btn) => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            const id = btn.dataset.id;
            const idx = queue.findIndex((q) => q.id === id);
            if (idx < 0) return;
            const item = queue[idx];
            console.log("[Wise-Paddle] remove clicked:", id, "status=", item.status, "userId=", item.userId);
            if (item.status === "processing") {
                handleCancelProcessingItem(btn, item);
                return;
            }
            // pending / done / error / cancelled：仅从队列里清掉
            if (item.fileUrl) URL.revokeObjectURL(item.fileUrl);
            queue.splice(idx, 1);
            renderQueue();
        });
    });
}

/** 队列状态 → 中文标签映射 */
const QUEUE_STATUS_LABEL = {
    pending: "待处理",
    processing: "处理中",
    done: "完成",
    error: "失败",
    cancelled: "已取消",
};

/**
 * 处理"正在处理中的 item 被 remove"的主动取消路径。
 * 4 步：通知服务端 cancel → abort fetch → 停 PDF 轮询 → UI 立即标 cancelled。
 * @param {HTMLButtonElement} btn - 被点击的 remove 按钮
 * @param {QueueItem} item
 */
function handleCancelProcessingItem(btn, item) {
    // 1) 通知服务端 cancel（fire-and-forget，scheduler 那边会 cancel futs）
    //    voucher_id 用于归属校验（M-04）
    if (item.userId) {
        const cancelQs = new URLSearchParams({voucher_id: aliveVoucher || ""});
        fetch(`/api/cancel/${encodeURIComponent(item.userId)}?${cancelQs.toString()}`, {method: "POST"})
            .catch((err) => console.warn("[Wise-Paddle] cancel fetch error:", err));
    }
    // 2) 中断进行中的 fetch（单图同步路径会立刻抛 AbortError）
    if (item.abortCtl) item.abortCtl.abort("user cancelled");
    // 3) 通知 pollPdfProgress 停止轮询（PDF 异步路径）
    item.pollAbort = true;
    // 4) UI 立刻置 cancelled —— 不等服务端响应，前端先给反馈
    item.status = "cancelled";
    const wallMs = item.t0 ? (performance.now() - item.t0) : 0;
    if (item.ctl) markCardCancelled(item.ctl, item, wallMs);
    // 直接修改 queue-item 元素，让用户立即看到变化（不等 renderQueue）
    const itemEl = btn.closest(".queue-item");
    if (itemEl) {
        itemEl.classList.remove("processing", "pending", "done", "error");
        itemEl.classList.add("cancelled");
        const badge = itemEl.querySelector(".badge");
        if (badge) {
            badge.classList.remove("processing", "pending", "done", "error");
            badge.classList.add("cancelled");
            badge.textContent = "已取消";
        }
    }
    showToast("任务已取消");
    // 最后再 renderQueue 同步 queue 数组（不再 splice，所以 item 还在）
    renderQueue();
}

clearQueueBtn.addEventListener("click", () => {
    queue.forEach((q) => {
        if (q.fileUrl) URL.revokeObjectURL(q.fileUrl);
        // 处理中的任务：先通知服务端取消，再中断 fetch / 轮询（M-11）
        if (q.status === "processing") {
            if (q.userId) {
                const cancelQs = new URLSearchParams({voucher_id: aliveVoucher || ""});
                fetch(`/api/cancel/${encodeURIComponent(q.userId)}?${cancelQs.toString()}`, {method: "POST"})
                    .catch(() => {});
            }
            q.pollAbort = true;
            if (q.abortCtl) q.abortCtl.abort("queue cleared");
        }
    });
    queue.length = 0;
    renderQueue();
});

// ===== 开始处理 =====
startBtn.addEventListener("click", async () => {
    const items = queue.filter((q) => q.status === "pending" && (!q.isPdf || q.pageCount > 0));
    if (items.length === 0) return;
    items.forEach((q) => q.status = "processing");
    renderQueue();
    await Promise.all(items.map((q) => processOne(q)));
});

/**
 * 处理一个 queue item：建占位卡 → 提交 OCR → 根据单图/PDF 走不同路径。
 * @param {QueueItem} item
 */
async function processOne(item) {
    await waitForLegalVoucher();
    const t0 = performance.now();
    // 1) 立即建一个"处理中"placeholder card
    const ctl = buildPlaceholderCard(item);
    // 2) 申请一个 client-side user_id, 通过 query string 传给服务端做 per-user 输出隔离
    const clientUserId = `web-${uid()}`;
    // 同时带上 session 级的 aliveVoucher（pollHealth 第一次拉 /health 时拿到）
    // —— 服务端 _kick_dead_user_loop 倒计时归零时调 scheduler.cancel_voucher(aliveVoucher)，
    // 把这个 voucher 名下所有 pending + in-flight 一起干掉
    const url = item.isPdf ? "/ocr/pdf" : "/ocr/upload";
    const qs = new URLSearchParams({
        user_id: clientUserId,
        voucher_id: aliveVoucher || "",
    });
    // 3) 在 item 上存取消需要的字段：remove 按钮 handler 要用
    item.userId = clientUserId;
    item.ctl = ctl;
    item.t0 = t0;
    item.abortCtl = new AbortController();
    item.pollAbort = false;
    // 总超时（避免挂死）
    const timeoutId = setTimeout(
        () => item.abortCtl.abort("timeout"),
        REQUEST_TIMEOUT_MS,
    );
    try {
        const fd = new FormData();
        fd.append("file", item.file);
        const r = await fetch(
            `${url}?${qs.toString()}`,
            {method: "POST", body: fd, signal: item.abortCtl.signal},
        );
        clearTimeout(timeoutId);
        const wallMs = performance.now() - t0;
        // 主动取消时 fetch 抛 AbortError —— 走到 catch 分支，由 catch 区分"主动取消"vs"真错误"
        if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            item.status = "error";
            renderQueue();
            markCardError(ctl, err.detail || `HTTP ${r.status}`, wallMs);
            return;
        }
        const data = await r.json();
        // 队列状态切到 processing
        item.status = "processing";
        renderQueue();
        // 区分 PDF 异步流式 vs 单图同步
        if (item.isPdf && data.total_pages && data.poll_url) {
            // PDF: 后台跑批, 走轮询路径
            pollPdfProgress(ctl, item, clientUserId, data, t0);
        } else {
            // 单图 / base64: 同步响应, 立即填充
            const wallMs2 = performance.now() - t0;
            item.status = "done";
            renderQueue();
            fillCardWithResult(ctl, item, data, wallMs2);
        }
    } catch (e) {
        clearTimeout(timeoutId);
        // 主动取消时 fetch 抛 AbortError，但 item.status 可能已经被 remove handler
        // 置为 "cancelled" —— 这种情况不要再覆盖成 error
        if (item.status === "cancelled" || item.pollAbort) return;
        // 超时 / 网络错误：通知服务端取消，避免 GPU 任务继续空转（M-11）
        if (item.userId) {
            const cancelQs = new URLSearchParams({voucher_id: aliveVoucher || ""});
            fetch(`/api/cancel/${encodeURIComponent(item.userId)}?${cancelQs.toString()}`, {method: "POST"})
                .catch(() => {});
        }
        item.status = "error";
        renderQueue();
        markCardError(ctl, String(e), performance.now() - t0);
    }
}

/**
 * 等待 aliveVoucher 从服务端拿到（pollHealth 第一次成功后才有值）。
 * 用 VOUCHER_POLL_MS 间隔的轮询，避免 CPU 空转。
 * @returns {Promise<void>}
 */
async function waitForLegalVoucher() {
    while (!aliveVoucher) {
        await new Promise((resolve) => setTimeout(resolve, VOUCHER_POLL_MS));
    }
}

/**
 * PDF 流式轮询: 每 PDF_POLL_INTERVAL_MS 拉一次 /api/ocr/pdf-status/{user_id},
 * 每收到新 page 立即更新对应 page-slot (而不是等所有 page 跑完).
 * 全部 done 后停止轮询.
 *
 * @param {Object} ctl - buildPlaceholderCard 返回的 controller
 * @param {QueueItem} item
 * @param {string} userId
 * @param {Object} initData - /ocr/pdf 返回的初始数据（含 poll_url, total_pages）
 * @param {number} t0 - performance.now() 起点
 */
async function pollPdfProgress(ctl, item, userId, initData, t0) {
    const totalPages = initData.total_pages;
    const pollUrl = initData.poll_url;
    const filledPages = new Set();

    // head 更新
    const rid = ctl.head.querySelector('[data-role="rid"]');
    if (rid) rid.textContent = (userId || "").replace(/^web-/, "");
    const pageinfo = ctl.head.querySelector('[data-role="pageinfo"]');
    if (pageinfo) pageinfo.textContent = `0 / ${totalPages} 页 · 处理中…`;
    const statusEl = ctl.head.querySelector('[data-role="status"]');
    if (statusEl) statusEl.textContent = "处理中";

    const tick = async () => {
        // 主动取消：用户点了 remove 按钮，停止轮询
        if (item.pollAbort) return;
        try {
            // 增量拉取：从第一个未填充页开始取，降低轮询带宽（M-12）
            let since = 0;
            for (let i = 0; i < totalPages; i++) {
                if (!filledPages.has(i)) {
                    since = i;
                    break;
                }
            }
            const sep = pollUrl.includes("?") ? "&" : "?";
            // voucher_id 用于归属校验（M-04）
            const r = await fetch(`${pollUrl}${sep}since=${since}&voucher_id=${encodeURIComponent(aliveVoucher || "")}`, {cache: "no-store"});
            if (!r.ok) {
                setTimeout(tick, PDF_POLL_INTERVAL_MS);
                return;
            }
            const data = await r.json();

            // 处理每个新 page (后端返回所有已完成页，前端用 filledPages 去重)
            const pageIds = Object.keys(data.pages || {}).map(Number).sort((a, b) => a - b);
            for (const pidx of pageIds) {
                if (filledPages.has(pidx)) continue;
                const pageData = data.pages[pidx];
                fillOnePageSlot(ctl, item, pidx, pageData);
                filledPages.add(pidx);
                // 切对应 page-slot 状态为 done, chip 状态为 done
                if (ctl.pageSlots[pidx]) ctl.pageSlots[pidx].dataset.status = "done";
                if (ctl.chips) {
                    const chip = ctl.chips.querySelector(`.page-chip[data-page-index="${pidx}"]`);
                    if (chip) {
                        chip.classList.remove("processing");
                        chip.classList.add("done");
                    }
                }
            }

            // 更新 head 计数
            const doneCount = data.done_count || filledPages.size;
            const phaseLabel = data.cancelled ? "已取消" : (data.done ? "完成" : "处理中");
            if (pageinfo) pageinfo.textContent = `${doneCount} / ${totalPages} 页 · ${data.done ? "完成" : "处理中…"}`;
            if (statusEl) statusEl.textContent = `${phaseLabel} ${doneCount}/${totalPages}`;

            if (data.done || data.cancelled || data.error) {
                const wallMs = performance.now() - t0;
                if (data.cancelled) {
                    item.status = "cancelled";
                } else if (data.error) {
                    item.status = "error";
                } else {
                    item.status = "done";
                }
                renderQueue();
                if (data.error) {
                    markCardError(ctl, data.error, wallMs);
                } else if (data.cancelled) {
                    // 服务端取消（voucher 倒计时归零等）：跟前端主动 cancel 同一套 UI
                    markCardCancelled(ctl, item, wallMs);
                }
                // 切所有未完成 page-slot 状态
                for (let i = 0; i < ctl.totalPages; i++) {
                    if (!filledPages.has(i)) {
                        if (ctl.pageSlots[i]) {
                            ctl.pageSlots[i].dataset.status = data.cancelled ? "cancelled" :
                                (data.error ? "error" : "waiting");
                        }
                        if (ctl.chips) {
                            const chip = ctl.chips.querySelector(`.page-chip[data-page-index="${i}"]`);
                            if (chip) {
                                chip.classList.remove("processing");
                                if (data.cancelled) chip.classList.add("cancelled");
                                else if (data.error) chip.classList.add("error");
                            }
                        }
                    }
                }
                // timing 更新
                const big = ctl.head.querySelector('[data-role="server"]');
                if (big) {
                    big.classList.remove("processing", "error", "cancelled");
                    if (data.cancelled) {
                        big.classList.add("cancelled");
                        big.textContent = "取消";
                    } else if (data.error) {
                        big.classList.add("error");
                        big.textContent = "失败";
                    } else {
                        big.innerHTML = `${(wallMs / 1000).toFixed(2)}<span style="font-size:14px;color:var(--fg-dim);">s</span>`;
                    }
                }
                const wallEl = ctl.head.querySelector('[data-role="wall"]');
                if (wallEl) wallEl.textContent = `${(wallMs / 1000).toFixed(2)}s`;
                return;  // 停止轮询
            }
            setTimeout(tick, PDF_POLL_INTERVAL_MS);
        } catch (e) {
            setTimeout(tick, PDF_POLL_RETRY_MS);  // 网络错误慢点重试
        }
    };
    tick();
}

/**
 * 单 page 增量更新: 把一个已完成 page 的 OCR 结果填到对应 page-slot
 * @param {Object} ctl
 * @param {QueueItem} item
 * @param {number} pageIndex
 * @param {Object} pageData
 */
function fillOnePageSlot(ctl, item, pageIndex, pageData) {
    const isPdf = item.isPdf;
    const slot = ctl.pageSlots[pageIndex];
    if (!slot) return;
    // 替换 image src 为 server 返回的图 (替换原 PDF preview)
    const finalSrc = pageData.image_b64
        ? `data:image/png;base64,${pageData.image_b64}`
        : (isPdf && item.previewPages[pageIndex] ? item.previewPages[pageIndex].b64 : (item.fileUrl || ""));
    const finalW = pageData.width || (item.previewPages[pageIndex] && item.previewPages[pageIndex].w) || 0;
    const finalH = pageData.height || (item.previewPages[pageIndex] && item.previewPages[pageIndex].h) || 0;
    const wrap = slot.querySelector('[data-role="wrap"]');
    if (wrap) {
        wrap.dataset.imgB64 = finalSrc;
        wrap.dataset.w = String(finalW);
        wrap.dataset.h = String(finalH);
        wrap.dataset.status = "done";
        const oldImg = wrap.querySelector("img");
        if (finalSrc) {
            if (oldImg) oldImg.src = finalSrc;
            else {
                const img = document.createElement("img");
                img.src = finalSrc;
                const cv = wrap.querySelector("canvas");
                if (cv) wrap.insertBefore(img, cv);
                else wrap.appendChild(img);
            }
        }
    }
    // 填 region 卡片
    fillRegionsInSlot(slot, pageIndex, pageData.regions || []);
    // wire canvas 检测框 + hover-copy
    if (typeof wirePageCanvas === "function") wirePageCanvas(ctl, pageIndex);
    ctl.card.dataset.status = "done";
    // 更新 tab 状态
    if (typeof updateResultTab === "function") updateResultTab(ctl.seq, "done");
}

/**
 * 把 regions 列表渲染到 slot 内的 regions 容器。
 * @param {HTMLElement} slot
 * @param {number} pageIndex
 * @param {Array} pageRegions
 */
function fillRegionsInSlot(slot, pageIndex, pageRegions) {
    const regionsEl = slot.querySelector(`[data-role="regions-${pageIndex}"]`);
    if (!regionsEl) return;
    if (pageRegions.length === 0) {
        regionsEl.innerHTML = `<div class="empty-state" style="grid-column:1/-1;padding:12px;">未检测到任何区域</div>`;
        return;
    }
    regionsEl.innerHTML = pageRegions.map((r, j) => `
        <div class="region" data-idx="${j}" data-page="${pageIndex}" data-label="${escapeAttr(r.label)}"
             data-rect="${escapeAttr(r.rect.join(","))}" data-score="${escapeAttr(String(r.score))}">
          <div class="r-head">
            <span class="r-label">${escapeHtml(r.label)}</span>
            <span class="r-score">${(r.score * 100).toFixed(1)}% · #${j}</span>
          </div>
          <div class="r-text">${escapeHtml(r.markdown || "")}</div>
        </div>`).join("");
}

// ===== Result card 构造 =====

/**
 * 立即建一个"处理中"占位卡：head 显示文件名 + "提交中"，所有 page-slot 状态=processing。
 * @param {QueueItem} item
 * @returns {Object} controller — 供 fillCardWithResult / markCardError 后续操作
 */
function buildPlaceholderCard(item) {
    emptyState.style.display = "none";
    const seq = ++resultSeq;
    const card = document.createElement("div");
    card.className = "result-card";
    card.dataset.seq = seq;
    card.dataset.kind = item.isPdf ? "pdf" : "image";
    card.dataset.startedAt = String(performance.now());

    // head
    const head = document.createElement("div");
    head.className = "head";
    head.innerHTML = `
    <div class="title">
      ${item.isPdf ? "📄" : "🖼"} <strong>${escapeHtml(item.file.name)}</strong>
      <span class="rid">request_id <b data-role="rid">—</b> · <span data-role="pageinfo">${item.isPdf ? `${item.pageCount} 页` : "1 页"}</span> · <span data-role="status">提交中…</span></span>
    </div>
    <div class="timing">
      <div class="big processing" data-role="server">—</div>
      <div class="small">wall <b data-role="wall">0.00s</b><br>queue <b data-role="queue">0.00s</b></div>
    </div>
  `;
    card.appendChild(head);

    const totalPages = item.isPdf ? Math.max(item.pageCount, 1) : 1;
    const isPdf = item.isPdf;

    // 页面导航（多页才有）
    let nav = null;
    let chips = null;
    let info = null;
    if (totalPages > 1) {
        nav = document.createElement("div");
        nav.className = "page-nav";
        nav.innerHTML = `
      <button class="nav-btn" data-role="prev" disabled>‹</button>
      <div class="page-chips" data-role="chips"></div>
      <button class="nav-btn" data-role="next" disabled>›</button>
      <span class="page-info" data-role="info">1 / ${totalPages}</span>
    `;
        chips = nav.querySelector('[data-role="chips"]');
        info = nav.querySelector('[data-role="info"]');
        for (let i = 0; i < totalPages; i++) {
            const chip = document.createElement("button");
            chip.className = "page-chip processing";
            chip.textContent = `${i + 1}`;
            chip.dataset.pageIndex = i;
            chip.addEventListener("click", () => showPage(i));
            chips.appendChild(chip);
        }
        card.appendChild(nav);
    }

    // page slots
    const pageSlots = [];
    for (let i = 0; i < totalPages; i++) {
        const slot = document.createElement("div");
        slot.className = "page-slot";
        slot.dataset.pageIndex = i;
        slot.dataset.status = "processing";  // 一开始就是 processing
        slot.style.display = (i === 0) ? "" : "none";

        const preview = isPdf ? item.previewPages[i] : null;
        const initialB64 = preview ? preview.b64 : (item.fileUrl || null);
        const initialW = preview ? preview.w : 0;
        const initialH = preview ? preview.h : 0;
        const initialSrc = initialB64 || "";

        if (initialSrc) {
            slot.innerHTML = `
        <div class="canvas-wrap" data-role="wrap" data-img-b64="${escapeAttr(initialSrc)}" data-w="${initialW}" data-h="${initialH}" data-status="processing">
          <img src="${escapeAttr(initialSrc)}" alt="page ${i + 1}" />
          <canvas></canvas>
        </div>
        <div class="regions" data-role="regions-${i}"></div>
      `;
        } else {
            slot.innerHTML = `
        <div class="canvas-wrap" data-role="wrap" data-status="processing">
          <div style="aspect-ratio:1;display:flex;align-items:center;justify-content:center;color:var(--fg-dim);background:#000;">处理中…</div>
          <canvas></canvas>
        </div>
        <div class="regions" data-role="regions-${i}"></div>
      `;
        }
        card.appendChild(slot);
        pageSlots.push(slot);
    }

    // 存储每页的 canvas 重绘函数（issue#2: 页面切换时触发重绘）
    const _pageRedraws = {};
    resultList.prepend(card);
    updateClearResultsBtn();

    // 在 result-switcher 注册该 card
    addResultTab(seq, item);

    const showPage = (idx) => {
        idx = Math.max(0, Math.min(totalPages - 1, idx));
        pageSlots.forEach((s, i) => {
            s.style.display = (i === idx) ? "" : "none";
        });
        if (chips) {
            chips.querySelectorAll(".page-chip").forEach((c, i) => c.classList.toggle("active", i === idx));
            if (info) info.textContent = `${idx + 1} / ${totalPages}`;
        }
        // 页面切换后触发 canvas 重绘（修复隐藏页 canvas 尺寸为 0 的问题）
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                if (_pageRedraws[idx]) _pageRedraws[idx]();
            });
        });
    };
    // 默认显示 page 0
    if (chips) {
        const c0 = chips.querySelector('.page-chip[data-page-index="0"]');
        if (c0) c0.classList.add("active");
    }

    return {card, head, nav, chips, info, pageSlots, totalPages, showPage, seq, _pageRedraws};
}

/**
 * 拿到 server 数据后，把占位卡升级为 done 状态：填 region、wire canvas、更新 timing
 * @param {Object} ctl
 * @param {QueueItem} item
 * @param {Object} data
 * @param {number} wallMs
 */
function fillCardWithResult(ctl, item, data, wallMs) {
    const isPdf = item.isPdf;
    const totalPages = (data.pages || []).length;
    const regions = data.regions || [];
    ctl.card.dataset.status = "done";
    updateResultTab(ctl.seq, "done");

    // 1) head 更新
    const rid = ctl.head.querySelector('[data-role="rid"]');
    if (rid) rid.textContent = data.request_id || "—";
    const pageinfo = ctl.head.querySelector('[data-role="pageinfo"]');
    if (pageinfo) pageinfo.textContent = `${totalPages} 页 · ${regions.length} regions`;
    const statusEl = ctl.head.querySelector('[data-role="status"]');
    if (statusEl) statusEl.textContent = "完成";
    const big = ctl.head.querySelector('[data-role="server"]');
    if (big) {
        big.classList.remove("processing");
        big.classList.remove("error");
        const s = Number(data.elapsed_seconds) || 0;
        big.innerHTML = `${s.toFixed(2)}<span style="font-size:14px;color:var(--fg-dim);">s</span>`;
    }
    const wallEl = ctl.head.querySelector('[data-role="wall"]');
    if (wallEl) wallEl.textContent = `${(wallMs / 1000).toFixed(2)}s`;
    const queueWaitEl = ctl.head.querySelector('[data-role="queue"]');
    if (queueWaitEl) queueWaitEl.textContent = `${(Number(data.queue_wait_seconds) || 0).toFixed(2)}s`;

    // 2) 启用 nav 按钮
    if (ctl.nav) {
        ctl.nav.querySelector('[data-role="prev"]').addEventListener("click", () => {
            const active = Array.from(ctl.chips.querySelectorAll(".page-chip")).findIndex((c) => c.classList.contains("active"));
            if (active > 0) ctl.showPage(active - 1);
        });
        ctl.nav.querySelector('[data-role="next"]').addEventListener("click", () => {
            const active = Array.from(ctl.chips.querySelectorAll(".page-chip")).findIndex((c) => c.classList.contains("active"));
            if (active < ctl.totalPages - 1) ctl.showPage(active + 1);
        });
    }

    // 3) 每个 page-slot 切到 done 状态
    const safeN = Math.min(ctl.totalPages, ctl.pageSlots.length);
    for (let i = 0; i < safeN; i++) {
        const slot = ctl.pageSlots[i];
        if (!slot) continue;
        slot.dataset.status = "done";
        const p = (data.pages && i < data.pages.length) ? data.pages[i] : null;
        const preview = isPdf ? (item.previewPages || [])[i] : null;
        const finalSrc = (p && p.image_b64) ? `data:image/png;base64,${p.image_b64}` : (preview ? preview.b64 : (item.fileUrl || ""));
        const finalW = (p && p.width) || (preview && preview.w) || 0;
        const finalH = (p && p.height) || (preview && preview.h) || 0;
        const wrap = slot.querySelector('[data-role="wrap"]');
        if (wrap) {
            wrap.dataset.imgB64 = finalSrc;
            wrap.dataset.w = String(finalW);
            wrap.dataset.h = String(finalH);
            wrap.dataset.status = "done";
            const oldImg = wrap.querySelector("img");
            if (finalSrc) {
                if (oldImg) {
                    oldImg.src = finalSrc;
                } else {
                    const img = document.createElement("img");
                    img.src = finalSrc;
                    const canvasEl = wrap.querySelector("canvas");
                    if (canvasEl) wrap.insertBefore(img, canvasEl);
                    else wrap.appendChild(img);
                }
            } else {
                const ph = wrap.querySelector("div");
                if (ph) ph.remove();
            }
        }
        // chip 状态
        if (ctl.chips) {
            const chip = ctl.chips.querySelector(`.page-chip[data-page-index="${i}"]`);
            if (chip) {
                chip.classList.remove("processing");
                chip.classList.add("done");
            }
        }
        // regions
        const pageRegions = regions.filter((r) => r.page_index === i);
        fillRegionsInSlot(slot, i, pageRegions);
        // wire canvas 检测框 + hover-copy
        wirePageCanvas(ctl, i);
    }
    // 如果 slot 比实际资料多，剩下的也标 done（带预览图）
    for (let i = safeN; i < ctl.totalPages; i++) {
        const slot = ctl.pageSlots[i];
        if (slot) slot.dataset.status = "done";
    }
}

/**
 * 给一个 page-slot 的 canvas 绑定检测框绘制 + hover 高亮 + click 复制。
 * @param {Object} ctl
 * @param {number} pageIdx
 */
function wirePageCanvas(ctl, pageIdx) {
    const slot = ctl.pageSlots[pageIdx];
    const wrap = slot.querySelector('[data-role="wrap"]');
    if (!wrap) return;
    const img = wrap.querySelector("img");
    const canvas = wrap.querySelector("canvas");
    const regionEls = Array.from(slot.querySelectorAll(".region"));
    if (regionEls.length === 0) return;

    const regionData = regionEls.map((el) => ({
        label: el.dataset.label,
        score: parseFloat(el.dataset.score),
        rect: el.dataset.rect.split(",").map(Number),
        el,
    }));

    const setup = () => {
        const dpr = window.devicePixelRatio || 1;
        const r = wrap.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) return; // 隐藏页跳过（等 showPage 触发重绘）
        canvas.width = Math.round(r.width * dpr);
        canvas.height = Math.round(r.height * dpr);
        canvas.style.width = r.width + "px";
        canvas.style.height = r.height + "px";
        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        drawBoxes(ctx, regionData, parseInt(wrap.dataset.w), parseInt(wrap.dataset.h), r.width, r.height, -1);
    };

    // 存储重绘函数供 showPage 切换页面时调用（修复隐藏页 canvas 尺寸为 0 的问题）
    if (ctl._pageRedraws) ctl._pageRedraws[pageIdx] = setup;

    if (img && img.complete && img.naturalWidth > 0) {
        setup();
    } else if (img) {
        img.addEventListener("load", setup);
    }
    requestAnimationFrame(() => requestAnimationFrame(setup));
    if (!ctl.card._resizeBound) {
        window.addEventListener("resize", () => {
            // 窗口缩放时重绘所有已注册的页（仅当前可见页会实际绘制）
            Object.values(ctl._pageRedraws).forEach((fn) => fn());
        });
        ctl.card._resizeBound = true;
    }

    let lastHoverIdx = -1;
    canvas.addEventListener("mousemove", (e) => {
        if (wrap.dataset.status !== "done") return;
        const idx = hitTestBox(e, canvas, regionData, parseInt(wrap.dataset.w), parseInt(wrap.dataset.h));
        if (idx !== lastHoverIdx) {
            lastHoverIdx = idx;
            const r = wrap.getBoundingClientRect();
            const ctx = canvas.getContext("2d");
            drawBoxes(ctx, regionData, parseInt(wrap.dataset.w), parseInt(wrap.dataset.h), r.width, r.height, idx);
            canvas.style.cursor = idx >= 0 ? "pointer" : "crosshair";
        }
    });
    canvas.addEventListener("mouseleave", () => {
        if (lastHoverIdx >= 0) {
            lastHoverIdx = -1;
            const r = wrap.getBoundingClientRect();
            const ctx = canvas.getContext("2d");
            drawBoxes(ctx, regionData, parseInt(wrap.dataset.w), parseInt(wrap.dataset.h), r.width, r.height, -1);
            canvas.style.cursor = "crosshair";
        }
    });
    canvas.addEventListener("click", (e) => {
        if (wrap.dataset.status !== "done") return;
        const idx = hitTestBox(e, canvas, regionData, parseInt(wrap.dataset.w), parseInt(wrap.dataset.h));
        if (idx >= 0) {
            const text = regionData[idx].el.querySelector(".r-text").textContent;
            copyText(text);
            regionData[idx].el.classList.add("copied");
            setTimeout(() => regionData[idx].el.classList.remove("copied"), 700);
        }
    });

    regionEls.forEach((el) => {
        el.addEventListener("click", () => {
            const text = el.querySelector(".r-text").textContent;
            copyText(text);
            el.classList.add("copied");
            setTimeout(() => el.classList.remove("copied"), 700);
        });
    });
}

/**
 * 把 card 标为错误状态：head 标红，所有 page-slot 标 error，加错误条。
 * @param {Object} ctl
 * @param {string} msg
 * @param {number} wallMs
 */
function markCardError(ctl, msg, wallMs) {
    ctl.card.dataset.status = "error";
    updateResultTab(ctl.seq, "error");
    // head: 标红
    const statusEl = ctl.head.querySelector('[data-role="status"]');
    if (statusEl) statusEl.textContent = "失败";
    const big = ctl.head.querySelector('[data-role="server"]');
    if (big) {
        big.classList.remove("processing");
        big.classList.add("error");
        big.textContent = "失败";
    }
    const wallEl = ctl.head.querySelector('[data-role="wall"]');
    if (wallEl) wallEl.textContent = `${(wallMs / 1000).toFixed(2)}s`;
    // 所有 page-slot 标 error
    ctl.pageSlots.forEach((slot, i) => {
        slot.dataset.status = "error";
        if (ctl.chips) {
            const chip = ctl.chips.querySelector(`.page-chip[data-page-index="${i}"]`);
            if (chip) {
                chip.classList.remove("processing");
                chip.classList.add("error");
            }
        }
    });
    // 在 card 内加错误条
    const errBar = document.createElement("div");
    errBar.className = "pdf-placeholder";
    errBar.style.borderLeftColor = "var(--danger)";
    errBar.style.margin = "16px 20px";
    errBar.innerHTML = `
    <div class="pulse" style="background:var(--danger);"></div>
    <div>${escapeHtml(msg)}</div>
  `;
    ctl.card.appendChild(errBar);
}

/**
 * 把 card 标为取消状态：跟 markCardError 同构，但不带错误条，big timing 文字是"取消"。
 * @param {Object} ctl
 * @param {QueueItem} item
 * @param {number} wallMs
 */
function markCardCancelled(ctl, item, wallMs) {
    ctl.card.dataset.status = "cancelled";
    updateResultTab(ctl.seq, "cancelled");
    const statusEl = ctl.head.querySelector('[data-role="status"]');
    if (statusEl) statusEl.textContent = "已取消";
    const big = ctl.head.querySelector('[data-role="server"]');
    if (big) {
        big.classList.remove("processing", "error");
        big.classList.add("cancelled");
        big.textContent = "取消";
    }
    const wallEl = ctl.head.querySelector('[data-role="wall"]');
    if (wallEl) wallEl.textContent = `${(wallMs / 1000).toFixed(2)}s`;
    ctl.pageSlots.forEach((slot, i) => {
        slot.dataset.status = "cancelled";
        if (ctl.chips) {
            const chip = ctl.chips.querySelector(`.page-chip[data-page-index="${i}"]`);
            if (chip) {
                chip.classList.remove("processing", "done", "error");
                chip.classList.add("cancelled");
            }
        }
    });
}

/**
 * 命中测试：判断鼠标点击在哪个 region 的框内。从后往前遍历（后绘制的在上层）。
 * @param {MouseEvent} e
 * @param {HTMLCanvasElement} canvas
 * @param {Array} regionData
 * @param {number} srcW - 原图宽
 * @param {number} srcH - 原图高
 * @returns {number} 命中的 region 索引，-1 表示未命中
 */
function hitTestBox(e, canvas, regionData, srcW, srcH) {
    const r = canvas.getBoundingClientRect();
    const sx = r.width / srcW, sy = r.height / srcH;
    const cx = (e.clientX - r.left);
    const cy = (e.clientY - r.top);
    for (let i = regionData.length - 1; i >= 0; i--) {
        const [x1, y1, x2, y2] = regionData[i].rect;
        if (cx >= x1 * sx && cx <= x2 * sx && cy >= y1 * sy && cy <= y2 * sy) {
            return i;
        }
    }
    return -1;
}

/**
 * 在 canvas 上绘制所有 region 框。highlightIdx 对应的框高亮（加粗 + 半透明填充 + 标签）。
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array} regions
 * @param {number} srcW
 * @param {number} srcH
 * @param {number} dstW
 * @param {number} dstH
 * @param {number} highlightIdx
 */
function drawBoxes(ctx, regions, srcW, srcH, dstW, dstH, highlightIdx) {
    ctx.clearRect(0, 0, dstW, dstH);
    const sx = dstW / srcW, sy = dstH / srcH;
    const root = getComputedStyle(document.documentElement);
    const colorMap = {
        text: root.getPropertyValue("--box-text").trim(),
        image: root.getPropertyValue("--box-image").trim(),
        table: root.getPropertyValue("--box-table").trim(),
        formula: root.getPropertyValue("--box-table").trim(),
        chart: root.getPropertyValue("--box-image").trim(),
    };
    regions.forEach((r, i) => {
        const [x1, y1, x2, y2] = r.rect;
        const isHi = (i === highlightIdx);
        ctx.lineWidth = isHi ? 3 : 1.5;
        ctx.strokeStyle = colorMap[r.label] || colorMap.text;
        if (isHi) {
            ctx.fillStyle = colorMap[r.label] || colorMap.text;
            ctx.globalAlpha = 0.18;
        } else {
            ctx.fillStyle = "transparent";
            ctx.globalAlpha = 1;
        }
        const w = (x2 - x1) * sx, h = (y2 - y1) * sy;
        ctx.fillRect(x1 * sx, y1 * sy, w, h);
        ctx.globalAlpha = 1;
        ctx.strokeRect(x1 * sx, y1 * sy, w, h);
        if (isHi) {
            ctx.font = '600 12px ui-monospace, Consolas, monospace';
            const txt = `${r.label} · #${i} · ${Math.round(r.score * 100)}%`;
            const tw = ctx.measureText(txt).width;
            const tagH = 18, padX = 6;
            let ty = y1 * sy - tagH - 2;
            if (ty < 0) ty = y1 * sy + 2;
            ctx.fillStyle = colorMap[r.label] || colorMap.text;
            ctx.fillRect(x1 * sx, ty, tw + padX * 2, tagH);
            ctx.fillStyle = "#0e1116";
            ctx.fillText(txt, x1 * sx + padX, ty + 13);
        }
    });
}

// ===== 多结果导航 (result-switcher) =====
/**
 * 在 result-switcher 顶部加一个 tab，最新添加的默认 active。
 * @param {number} seq
 * @param {QueueItem} item
 */
function addResultTab(seq, item) {
    const tab = document.createElement("div");
    tab.className = "rs-tab active";  // 最新添加的默认 active
    tab.dataset.seq = seq;
    // 缩略图：单图用 fileUrl, PDF 用第一页 preview b64
    let thumbSrc = "";
    if (item.isPdf && item.previewPages[0]) {
        thumbSrc = item.previewPages[0].b64;
    } else if (item.fileUrl) {
        thumbSrc = item.fileUrl;
    }
    const statusText = QUEUE_STATUS_LABEL[item.status] || "处理中";
    tab.innerHTML = `
    ${thumbSrc ? `<img class="rs-tab-thumb" src="${escapeAttr(thumbSrc)}" alt="" />` : `<div class="rs-tab-thumb" style="display:flex;align-items:center;justify-content:center;font-size:11px;color:var(--fg-dim);">${item.isPdf ? "PDF" : "IMG"}</div>`}
    <div class="rs-tab-text">
      <div class="rs-tab-num">#${seq}</div>
      <div class="rs-tab-name" title="${escapeAttr(item.file.name)}">${escapeHtml(item.file.name)}</div>
    </div>
    <span class="rs-tab-status ${escapeAttr(item.status)}">${escapeHtml(statusText)}</span>
  `;
    tab.addEventListener("click", () => scrollToResult(seq));
    // 取消之前的 active（最新加的在最前）
    rsTabs.querySelectorAll(".rs-tab.active").forEach((t) => t.classList.remove("active"));
    rsTabs.appendChild(tab);
    updateResultSwitcherNav();
    // 滚动 tabs 到可见
    tab.scrollIntoView({behavior: "smooth", block: "nearest", inline: "nearest"});
}

/**
 * 更新某个 tab 的状态徽标。
 * @param {number} seq
 * @param {string} status - processing | done | error | cancelled
 */
function updateResultTab(seq, status) {
    const tab = rsTabs.querySelector(`.rs-tab[data-seq="${seq}"]`);
    if (!tab) return;
    const statusEl = tab.querySelector(".rs-tab-status");
    if (statusEl) {
        statusEl.classList.remove("processing", "done", "error", "cancelled");
        statusEl.classList.add(status);
        statusEl.textContent = QUEUE_STATUS_LABEL[status] || status;
    }
}

function updateResultSwitcherNav() {
    const all = Array.from(resultList.children);
    const n = all.length;
    resultSwitcher.classList.toggle("show", n > 0);
    rsInfo.textContent = `${n} / ${n}`;
    rsPrev.disabled = true;  // scrollToResult 用 scrollIntoView, prev/next 暂简化
    rsNext.disabled = true;
}

/**
 * 滚动到指定 seq 的结果卡，并高亮闪烁。
 * @param {number} seq
 */
function scrollToResult(seq) {
    const card = resultList.querySelector(`.result-card[data-seq="${seq}"]`);
    if (!card) return;
    // 滚动到该卡片（顶部留给 sticky header + switcher）
    const rect = card.getBoundingClientRect();
    const top = rect.top + window.scrollY - 110;
    window.scrollTo({top, behavior: "smooth"});
    // 高亮闪烁
    card.classList.remove("scroll-target");
    void card.offsetWidth;  // 强制重排重启动画
    card.classList.add("scroll-target");
    setTimeout(() => card.classList.remove("scroll-target"), 1500);
    // 更新 active 状态
    rsTabs.querySelectorAll(".rs-tab").forEach((t) => t.classList.toggle("active", Number(t.dataset.seq) === seq));
}

rsPrev.addEventListener("click", () => {
    const all = Array.from(resultList.children);
    const cur = rsTabs.querySelector(".rs-tab.active");
    if (!cur || all.length === 0) return;
    const curIdx = all.findIndex((c) => Number(c.dataset.seq) === Number(cur.dataset.seq));
    if (curIdx > 0) scrollToResult(Number(all[curIdx - 1].dataset.seq));
});
rsNext.addEventListener("click", () => {
    const all = Array.from(resultList.children);
    const cur = rsTabs.querySelector(".rs-tab.active");
    if (!cur || all.length === 0) return;
    const curIdx = all.findIndex((c) => Number(c.dataset.seq) === Number(cur.dataset.seq));
    if (curIdx < all.length - 1) scrollToResult(Number(all[curIdx + 1].dataset.seq));
});

function clearAllResultTabs() {
    rsTabs.innerHTML = "";
    resultSwitcher.classList.remove("show");
    rsInfo.textContent = "0 / 0";
}

// ===== 清空结果 =====
function updateClearResultsBtn() {
    const has = resultList.children.length > 0;
    clearResultsBtn.style.display = has ? "" : "none";
    if (has) {
        clearResultsBtn.textContent = `清空结果 (${resultList.children.length})`;
    }
}

clearResultsBtn.addEventListener("click", () => {
    resultList.querySelectorAll("img[src^='blob:']").forEach((img) => URL.revokeObjectURL(img.src));
    resultList.innerHTML = "";
    emptyState.style.display = "";
    clearAllResultTabs();
    updateClearResultsBtn();
});
const resultObserver = new MutationObserver(updateClearResultsBtn);
resultObserver.observe(resultList, {childList: true});
updateClearResultsBtn();
