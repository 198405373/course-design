/* 工业产品表面缺陷智能检测系统 —— 前端逻辑 */
"use strict";

const VERDICT_META = {
  NG:     { label: "缺陷 · 不合格", cls: "ng",  hint: "CNN 与传统特征双路同判缺陷，判定为不合格（高风险）。" },
  REVIEW: { label: "疑似 · 待复检", cls: "review", hint: "双模型结论分歧，转入人工复检确认后再放行或判废。" },
  OK:     { label: "正常 · 放行",   cls: "ok",  hint: "双路均判定正常，予以放行。" },
};
const COLOR = { OK: "#16a34a", NG: "#dc2626", REVIEW: "#d97706" };

const $ = (id) => document.getElementById(id);
const api = async (url, opts = {}) => {
  const r = await fetch(url, opts);
  let data = {};
  try { data = await r.json(); } catch (_) {}
  if (!r.ok) throw new Error(data.error || `请求失败(${r.status})`);
  return data;
};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let currentId = null;
let chartDist = null, chartTrend = null;

/* ---------------- 提示 ---------------- */
let toastTimer = null;
function toast(msg, isErr = false) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.toggle("err", isErr);
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 2600);
}

/* ---------------- 上传与检测 ---------------- */
const fileInput = $("fileInput");
const dropZone = $("dropZone");

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => fileInput.files[0] && handleFile(fileInput.files[0]));
["dragover", "dragenter"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove("dragover"); }));
dropZone.addEventListener("drop", (e) => e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0]));
dropZone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });

function handleFile(file) {
  if (!/image\/(jpeg|png|bmp)/.test(file.type)) { toast("仅支持 jpg/png/bmp 图片", true); return; }
  const url = URL.createObjectURL(file);
  const prev = $("preview");
  prev.src = url; prev.hidden = false;
  document.querySelector(".dz-inner").style.display = "none";
  predict(file);
}

async function predict(file) {
  $("loading").classList.remove("hidden");
  $("resultBody").classList.add("hidden");
  $("emptyResult").classList.add("hidden");
  // 新检测开始：清空对旧记录的操作权，避免误点复检按钮作用于上一条记录
  currentId = null;
  $("btnReviewPass").disabled = true;
  $("btnReviewFail").disabled = true;
  try {
    const fd = new FormData();
    fd.append("image", file);
    fd.append("threshold", $("threshold").value);
    const res = await api("/api/predict", { method: "POST", body: fd });
    renderResult(res);
    $("resultCard").scrollIntoView({ behavior: "smooth", block: "nearest" });
    toast(`检测完成 → ${VERDICT_META[res.verdict].label}`);
    refreshHistory(); refreshStats();
  } catch (err) {
    toast(err.message, true);
    $("emptyResult").classList.remove("hidden");
    $("emptyResult").textContent = "检测失败：" + err.message;
  } finally {
    $("loading").classList.add("hidden");
  }
}

/* ---------------- 结果渲染 ---------------- */
function renderResult(res) {
  currentId = res.id;
  $("resultBody").classList.remove("hidden");
  $("emptyResult").classList.add("hidden");

  const meta = VERDICT_META[res.verdict] || VERDICT_META.REVIEW;
  const badge = $("verdictBadge");
  badge.textContent = res.verdict + "  " + meta.label;
  badge.className = "verdict-badge " + meta.cls;
  $("verdictMsg").textContent = `图像 ${esc(res.filename)} · 判定阈值 ${res.threshold}`;

  $("pCnn").textContent = res.p_cnn.toFixed(3);
  $("barCnn").style.width = (res.p_cnn * 100) + "%";
  $("pMl").textContent = res.p_ml.toFixed(3);
  $("barMl").style.width = (res.p_ml * 100) + "%";

  const lat = res.latency_ms != null ? ` · 推理耗时 ${res.latency_ms} ms` : "";
  $("resultMeta").innerHTML =
    `记录 ID #${res.id} · 缺陷带（RF 定位）第 <b>${res.ml_band + 1}</b> 条 / 共 8 条${lat}`;
  $("fuseHint").className = "meta-line fuse-line";
  $("fuseHint").textContent = meta.hint;

  const reviewable = res.verdict !== "OK";
  $("btnReviewPass").disabled = !reviewable;
  $("btnReviewFail").disabled = !reviewable;
  const viewBtn = $("btnViewImage");
  viewBtn.hidden = !res.image_url;
  if (res.image_url) viewBtn.onclick = () => openModal(res.image_url);
}

$("btnReviewPass").addEventListener("click", () => reviewCurrent("pass"));
$("btnReviewFail").addEventListener("click", () => reviewCurrent("fail"));

async function reviewCurrent(result) {
  if (!currentId) return;
  try {
    await api(`/api/records/${currentId}/review`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ result }),
    });
    toast(result === "pass" ? "已标记：复检放行（合格）" : "已标记：复检确认缺陷");
    refreshHistory(); refreshStats();
  } catch (err) { toast(err.message, true); }
}

/* ---------------- 历史记录 ---------------- */
async function refreshHistory() {
  try {
    const { records } = await api("/api/records?limit=100");
    const body = $("histBody");
    body.innerHTML = "";
    $("histEmpty").classList.toggle("hidden", records.length > 0);
    for (const r of records) {
      const tr = document.createElement("tr");
      const revTxt = r.review_result ? (r.review_result === "pass" ? "放行" : "确认缺陷") : "—";
      tr.innerHTML = `
        <td>#${r.id}</td>
        <td class="mini">${esc((r.created_at || "").replace("T", " ").slice(0, 19))}</td>
        <td title="${esc(r.filename)}">${esc(r.filename)}</td>
        <td><span class="chip ${VERDICT_META[r.verdict].cls}">${r.verdict}</span></td>
        <td>${(+r.p_cnn).toFixed(3)}</td>
        <td>${(+r.p_ml).toFixed(3)}</td>
        <td class="mini">${r.ml_band != null ? "第" + (r.ml_band + 1) + "条" : "—"}</td>
        <td class="mini">${revTxt}</td>
        <td>
          <button class="btn link" data-act="view" data-url="/api/image/${r.id}">图</button>
          ${r.verdict !== "OK" && !r.review_result
            ? `<button class="btn link" data-act="pass" data-id="${r.id}">放行</button>
               <button class="btn link" data-act="fail" data-id="${r.id}">判废</button>`
            : ""}
        </td>`;
      body.appendChild(tr);
    }
  } catch (err) { toast(err.message, true); }
}

$("histBody").addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-act]");
  if (!btn) return;
  const act = btn.dataset.act;
  if (act === "view") { openModal(btn.dataset.url); return; }
  try {
    await api(`/api/records/${btn.dataset.id}/review`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ result: act }),
    });
    toast(act === "pass" ? "已放行" : "已确认缺陷");
    refreshHistory(); refreshStats();
  } catch (err) { toast(err.message, true); }
});

/* ---------------- 统计与图表 ---------------- */
async function refreshStats() {
  try {
    const s = await api("/api/stats");
    $("kTotal").textContent = s.total;
    $("kOk").textContent = s.ok;
    $("kNg").textContent = s.ng;
    $("kReview").textContent = s.review;
    $("kRate").textContent = (s.defect_rate * 100).toFixed(1) + "%";
    $("kPass").textContent = s.review_pass;
    $("kFail").textContent = s.review_fail;

    // 饼图：判定分布
    chartDist = chartDist || echarts.init($("chartDist"));
    chartDist.setOption({
      tooltip: { trigger: "item" },
      legend: { bottom: 0, icon: "circle" },
      series: [{
        type: "pie", radius: ["45%", "72%"], center: ["50%", "44%"],
        label: { show: false }, itemStyle: { borderRadius: 6 },
        data: [
          { value: s.ok, name: "正常", itemStyle: { color: COLOR.OK } },
          { value: s.ng, name: "缺陷", itemStyle: { color: COLOR.NG } },
          { value: s.review, name: "待复检", itemStyle: { color: COLOR.REVIEW } },
        ],
      }],
    }, true);

    // 趋势：最近记录判定条带（按时间正序，x 轴用真实检测 ID 与历史表对应）
    const rec = [...(s.recent || [])].reverse(); // 最早在前、最新在后
    chartTrend = chartTrend || echarts.init($("chartTrend"));
    chartTrend.setOption({
      tooltip: {
        trigger: "axis",
        formatter: (ps) => {
          const d = ps[0] && ps[0].data ? ps[0].data.meta : null;
          if (!d) return "";
          const t = (d.created_at || "").replace("T", " ").slice(0, 19);
          return `<b>${VERDICT_META[d.verdict] ? VERDICT_META[d.verdict].label : d.verdict}</b>`
                 + `<br/>检测 ID #${d.id} · ${t}`;
        },
      },
      grid: { left: 34, right: 12, top: 22, bottom: 28 },
      xAxis: {
        type: "category",
        data: rec.map((x) => x.id),
        name: "检测 ID（近50次）", nameLocation: "middle", nameGap: 24,
        axisLabel: { interval: "auto", rotate: rec.length > 24 ? 40 : 0 },
      },
      yAxis: { type: "value", min: 0, max: 1, splitNumber: 1,
               axisLabel: { show: false } },
      series: [{
        type: "bar", barWidth: "70%",
        data: rec.map((x) => ({
          value: 1,
          meta: x,
          itemStyle: { color: COLOR[x.verdict] || "#94a3b8" },
        })),
      }],
    }, true);
  } catch (err) { toast("统计加载失败：" + err.message, true); }
}

/* ---------------- 弹窗 ---------------- */
function openModal(url) { $("modalImg").src = url; $("modal").classList.remove("hidden"); }
$("modalClose").addEventListener("click", () => $("modal").classList.add("hidden"));
$("modal").addEventListener("click", (e) => { if (e.target === $("modal")) $("modal").classList.add("hidden"); });

/* ---------------- 工具按钮 ---------------- */
$("btnRefresh").addEventListener("click", refreshStats);
$("btnRefreshHist").addEventListener("click", refreshHistory);

function resetResultView() {
  currentId = null;
  $("btnReviewPass").disabled = true;
  $("btnReviewFail").disabled = true;
  $("btnViewImage").hidden = true;
  $("resultBody").classList.add("hidden");
  $("emptyResult").classList.remove("hidden");
  $("emptyResult").textContent = "尚无检测记录，请在左侧上传图像";
}

$("btnResetHist").addEventListener("click", async () => {
  if (!window.confirm("确定清空全部检测历史与质检统计吗？此操作不可恢复。")) return;
  try {
    await api("/api/records/reset", { method: "POST" });
    resetResultView();
    refreshHistory();
    refreshStats();
    toast("检测历史已清空，ID 从 #1 重新开始");
  } catch (err) { toast(err.message, true); }
});
window.addEventListener("resize", () => { chartDist && chartDist.resize(); chartTrend && chartTrend.resize(); });

/* ---------------- 启动 ---------------- */
(async () => {
  try {
    await api("/api/health");
    $("statusBadge").textContent = "● 服务正常";
    $("statusBadge").className = "badge ok";
  } catch (_) {
    $("statusBadge").textContent = "● 后端离线";
    $("statusBadge").className = "badge err";
  }
  refreshHistory();
  refreshStats();
})();
