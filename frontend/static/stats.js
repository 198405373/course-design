/* 统计报表页：KPI + ECharts */
"use strict";

let chartDist = null, chartTrend = null, chartReview = null;

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

    // 判定分布（环形）
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

    // 近 50 次趋势
    const rec = s.recent || [];
    chartTrend = chartTrend || echarts.init($("chartTrend"));
    chartTrend.setOption({
      tooltip: { trigger: "axis" },
      grid: { left: 34, right: 10, top: 16, bottom: 26 },
      xAxis: { type: "category", data: rec.map((_, i) => i + 1), name: "近50次" },
      yAxis: { type: "value", min: 0, max: 1, splitNumber: 1 },
      series: [{
        type: "bar", barWidth: "70%",
        data: rec.map((x) => ({ value: 1, itemStyle: { color: COLOR[x.verdict] || "#94a3b8" } })),
      }],
    }, true);

    // 复检分布
    chartReview = chartReview || echarts.init($("chartReview"));
    const hasReview = (s.review_pass + s.review_fail) > 0;
    chartReview.setOption({
      tooltip: { trigger: "item" },
      legend: { bottom: 0, icon: "circle" },
      series: [{
        type: "pie", radius: ["40%", "70%"], center: ["50%", "44%"],
        label: { show: false }, itemStyle: { borderRadius: 6 },
        data: hasReview
          ? [
              { value: s.review_pass, name: "放行(合格)", itemStyle: { color: COLOR.OK } },
              { value: s.review_fail, name: "确认缺陷", itemStyle: { color: COLOR.NG } },
            ]
          : [{ value: 1, name: "暂无复检", itemStyle: { color: "#e2e8f0" } }],
      }],
    }, true);
    $("reviewHint").style.display = hasReview ? "none" : "";
  } catch (err) {
    toast("统计加载失败：" + err.message, true);
  }
}

$("btnRefresh").addEventListener("click", refreshStats);
window.addEventListener("resize", () => {
  chartDist && chartDist.resize();
  chartTrend && chartTrend.resize();
  chartReview && chartReview.resize();
});
initStatusBadge();
refreshStats();
