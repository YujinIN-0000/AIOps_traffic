// post.js — CSV 업로드 & 예측 결과 처리
// 데모 흐름: ① 업로드 → ② 예측 → ③ RMSE → ④ LLM보고서 → ⑤ 재학습결정
const API_BASE = "http://localhost:8001";

// ── Chart.js lifecycle charts (actual data) ─────────────────────
var lifecycleChart = null;
var rmseChart = null;
var lastChartPayload = null;

if (window.Chart && window.ChartAnnotation) {
    Chart.register(window.ChartAnnotation);
}

function _formatLabel(iso) {
    if (!iso) return "";
    var d = new Date(iso);
    if (isNaN(d.getTime())) return "";
    var mm = String(d.getMonth() + 1).padStart(2, "0");
    var dd = String(d.getDate()).padStart(2, "0");
    var hh = String(d.getHours()).padStart(2, "0");
    var mi = String(d.getMinutes()).padStart(2, "0");
    return mm + "-" + dd + " " + hh + ":" + mi;
}

function _rollingRmse(actual, predicted, windowSize) {
    var n = Math.min(actual.length, predicted.length);
    if (n === 0) return [];
    var win = Math.max(10, Math.min(windowSize || 144, n));
    var rmse = new Array(n);
    var cum = new Array(n);
    var count = new Array(n);
    for (var i = 0; i < n; i++) {
        var a = actual[i];
        var p = predicted[i];
        var has = (a != null && p != null && isFinite(a) && isFinite(p));
        var se = has ? (a - p) * (a - p) : 0;
        cum[i] = se + (i > 0 ? cum[i - 1] : 0);
        count[i] = (has ? 1 : 0) + (i > 0 ? count[i - 1] : 0);
        var start = Math.max(0, i - win + 1);
        var sum = cum[i] - (start > 0 ? cum[start - 1] : 0);
        var denom = count[i] - (start > 0 ? count[start - 1] : 0);
        rmse[i] = denom > 0 ? Math.sqrt(sum / denom) : null;
    }
    return rmse;
}

function _findRetrainAndRecover(rmseSeries, threshold) {
    var retrainIdx = -1;
    var recoverIdx = -1;
    for (var i = 0; i < rmseSeries.length; i++) {
        if (rmseSeries[i] > threshold) { retrainIdx = i; break; }
    }
    if (retrainIdx >= 0) {
        for (var j = retrainIdx + 1; j < rmseSeries.length; j++) {
            if (rmseSeries[j] <= threshold) { recoverIdx = j; break; }
        }
    }
    return { retrainIdx: retrainIdx, recoverIdx: recoverIdx };
}

function renderLifecycleCharts(payload) {
    if (!payload || !payload.actual_series || !payload.pred_series) return;
    lastChartPayload = payload;

    var actual = payload.actual_series;
    var predicted = payload.pred_series;
    var labels = (payload.time_index || []).map(_formatLabel);
    if (!labels.length) {
        labels = actual.map(function (_, i) { return "t" + (i + 1); });
    }
    var threshold = payload.rmse_threshold || 200;
    var rmseSeries = _rollingRmse(actual, predicted, 144);
    var phase = _findRetrainAndRecover(rmseSeries, threshold);

    var zoneAnnotations = {};
    if (phase.retrainIdx < 0) {
        zoneAnnotations.zone1 = {
            type: "box",
            xMin: 0, xMax: labels.length - 1,
            backgroundColor: "rgba(16,185,129,0.07)",
            borderColor: "transparent",
        };
    } else {
        zoneAnnotations.zone1 = {
            type: "box",
            xMin: 0, xMax: Math.max(0, phase.retrainIdx - 1),
            backgroundColor: "rgba(16,185,129,0.07)",
            borderColor: "transparent",
        };
        zoneAnnotations.zone2 = {
            type: "box",
            xMin: phase.retrainIdx, xMax: (phase.recoverIdx > 0 ? phase.recoverIdx - 1 : labels.length - 1),
            backgroundColor: "rgba(239,68,68,0.07)",
            borderColor: "transparent",
        };
        if (phase.recoverIdx > 0) {
            zoneAnnotations.zone3 = {
                type: "box",
                xMin: phase.recoverIdx, xMax: labels.length - 1,
                backgroundColor: "rgba(99,102,241,0.07)",
                borderColor: "transparent",
            };
        }
        zoneAnnotations.retrainLine = {
            type: "line",
            xMin: phase.retrainIdx, xMax: phase.retrainIdx,
            borderColor: "rgba(245,158,11,0.85)",
            borderWidth: 2,
            borderDash: [5, 4],
            label: {
                display: true,
                content: "재학습",
                color: "#fcd34d",
                backgroundColor: "rgba(245,158,11,0.15)",
                borderRadius: 4,
                padding: { x: 6, y: 3 },
                font: { size: 10, weight: "600" },
                position: "start",
            }
        };
    }

    if (lifecycleChart) lifecycleChart.destroy();
    if (rmseChart) rmseChart.destroy();

    Chart.defaults.color = "#6B7280";
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 11;

    var lcCtx = document.getElementById("lifecycleChart");
    if (lcCtx) {
        lifecycleChart = new Chart(lcCtx.getContext("2d"), {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "실측 트래픽",
                        data: actual,
                        borderColor: "#ef4444",
                        backgroundColor: "rgba(239,68,68,0.08)",
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.3,
                        fill: false,
                    },
                    {
                        label: "예측 트래픽",
                        data: predicted,
                        borderColor: "#6366f1",
                        backgroundColor: "rgba(99,102,241,0.08)",
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.3,
                        borderDash: [4, 3],
                        fill: false,
                    },
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 900, easing: "easeInOutQuart" },
                interaction: { mode: "index", intersect: false },
                plugins: {
                    legend: { position: "top", align: "end", labels: { boxWidth: 12, usePointStyle: true, pointStyleWidth: 10 } },
                    tooltip: {
                        backgroundColor: "#fff",
                        titleColor: "#111827",
                        bodyColor: "#374151",
                        borderColor: "#E2E8F0",
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {
                            label: function(ctx) {
                                return " " + ctx.dataset.label + ": " + ctx.parsed.y.toFixed(0);
                            }
                        }
                    },
                    annotation: { annotations: zoneAnnotations }
                },
                scales: {
                    x: {
                        ticks: {
                            maxTicksLimit: 10,
                            callback: function(value, index) {
                                var step = Math.ceil(labels.length / 8);
                                if (index % step !== 0) return "";
                                return labels[index] || "";
                            }
                        },
                        grid: { color: "#F1F5F9" }
                    },
                    y: { ticks: { maxTicksLimit: 6 }, grid: { color: "#F1F5F9" } }
                }
            }
        });
    }

    var rmCtx = document.getElementById("rmseTimeChart");
    if (rmCtx) {
        rmseChart = new Chart(rmCtx.getContext("2d"), {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "RMSE",
                        data: rmseSeries,
                        borderColor: "#f59e0b",
                        backgroundColor: function(ctx) {
                            var grad = ctx.chart.ctx.createLinearGradient(0, 0, 0, 200);
                            grad.addColorStop(0, "rgba(245,158,11,0.25)");
                            grad.addColorStop(1, "rgba(245,158,11,0.02)");
                            return grad;
                        },
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.35,
                        fill: true,
                    },
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 900, easing: "easeInOutQuart" },
                interaction: { mode: "index", intersect: false },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: "#fff",
                        titleColor: "#111827",
                        bodyColor: "#374151",
                        borderColor: "#E2E8F0",
                        borderWidth: 1,
                        padding: 10,
                    },
                    annotation: {
                        annotations: (function () {
                            var ann = {
                                threshLine: {
                                    type: "line",
                                    yMin: threshold, yMax: threshold,
                                    borderColor: "rgba(239,68,68,0.7)",
                                    borderWidth: 1.5,
                                    borderDash: [6, 4],
                                    label: {
                                        display: true,
                                        content: "임계값 " + threshold,
                                        color: "#fca5a5",
                                        backgroundColor: "rgba(239,68,68,0.12)",
                                        borderRadius: 4,
                                        padding: { x: 6, y: 3 },
                                        font: { size: 10, weight: "600" },
                                        position: "end",
                                    }
                                }
                            };
                            if (phase.retrainIdx >= 0) {
                                ann.rLine = {
                                    type: "line",
                                    xMin: phase.retrainIdx, xMax: phase.retrainIdx,
                                    borderColor: "rgba(245,158,11,0.7)",
                                    borderWidth: 1.5,
                                    borderDash: [5, 4],
                                };
                            }
                            return ann;
                        })()
                    }
                },
                scales: {
                    x: { ticks: { maxTicksLimit: 8, callback: function() { return ""; } }, grid: { color: "#F1F5F9" } },
                    y: {
                        ticks: { maxTicksLimit: 5 },
                        grid: { color: "#F1F5F9" },
                        title: { display: true, text: "RMSE", color: "#9CA3AF", font: { size: 10 } }
                    }
                }
            }
        });
    }
}

// ── 파이프라인 단계 업데이트 ──────────────────────────────────────────────────
function setPipelineStep(nodeId, state) {
    // state: 'active' | 'done' | '' (초기화)
    var node = document.getElementById("pnode-" + nodeId);
    var line = document.getElementById("parr-" + nodeId);  // p-line (connector)
    if (!node) return;
    node.classList.remove("active", "done");
    if (state) node.classList.add(state);
    // connector line: mark done when the step to its left is done
    if (line) {
        line.classList.remove("done");
        if (state === "done") line.classList.add("done");
    }
}

function resetPipeline() {
    for (var i = 1; i <= 5; i++) setPipelineStep(i, "");
}

// ── 로딩 오버레이 ─────────────────────────────────────────────────────────────
window.showLoading = function (msg) {
    document.getElementById("loadingMsg").textContent = msg || "처리 중입니다...";
    document.getElementById("loadingOverlay").style.display = "flex";
};
window.hideLoading = function () {
    document.getElementById("loadingOverlay").style.display = "none";
};

// ── KPI 카드 업데이트 ─────────────────────────────────────────────────────────
function updateKPI(rmse, threshold, needs) {
    // RMSE 값
    document.getElementById("kpiRmseVal").textContent = rmse.toFixed(2);
    var rmseCard = document.getElementById("kpi-rmse");
    rmseCard.classList.toggle("warn", needs);
    rmseCard.classList.toggle("ok",   !needs);

    // 상태
    document.getElementById("kpiStatusVal").textContent = needs ? "초과" : "정상";
    document.getElementById("kpiStatusSub").textContent =
        needs ? "재학습 권고" : "임계값 이내";
    var statusCard = document.getElementById("kpi-status");
    statusCard.classList.toggle("warn", needs);
    statusCard.classList.toggle("ok",   !needs);

    // 임계값
    document.getElementById("kpiThreshVal").textContent = threshold;

    // 재학습 상태 요약 (액션 패널에 표시)
    var kpiModeVal = document.getElementById("retrainStatusValue");
    if (kpiModeVal) {
        kpiModeVal.textContent = needs ? "재학습 필요" : "정상";
    }
}

// ── 대시보드 표시 ─────────────────────────────────────────────────────────────
function showDashboard() {
    document.getElementById("noPrediction").style.display  = "none";
    document.getElementById("dashboardCards").style.display = "block";
}

// ── 업로드 폼 제출 ────────────────────────────────────────────────────────────
document.getElementById("uploadForm").addEventListener("submit", function (e) {
    e.preventDefault();

    var file = document.getElementById("fileUpload").files[0];
    if (!file) { alert("업로드할 CSV 파일을 선택해주세요."); return; }

    // 이전 상태 초기화
    resetPipeline();
    var retrainBanner = document.getElementById("retrainResultBanner");
    if (retrainBanner) retrainBanner.style.display = "none";

    setPipelineStep(1, "active");
    setPipelineStep(2, "active");
    showLoading(
        "LSTM 예측 실행 중...\n" +
        "모델이 없으면 자동으로 학습을 시작합니다 (3~10분 소요)."
    );

    var formData = new FormData();
    formData.append("file", file);

    fetch(API_BASE + "/upload", { method: "POST", body: formData })
        .then(function (r) {
            if (!r.ok) return r.json().then(function (e) { throw new Error(e.detail || "서버 오류"); });
            return r.json();
        })
        .then(function (data) {
            hideLoading();
            setPipelineStep(1, "done");
            setPipelineStep(2, "done");
            setPipelineStep(3, "done");

            handleUploadResult(data);
        })
        .catch(function (err) {
            hideLoading();
            resetPipeline();
            console.error("upload error:", err);
            alert("업로드/예측 오류:\n" + err.message);
        });
});

// ── 업로드 결과 처리 ──────────────────────────────────────────────────────────
function handleUploadResult(data) {
    var rmse      = data.result_evaluating_LSTM;
    var threshold = data.rmse_threshold;
    var needs     = data.needs_retrain;

    // 예측 그래프 표시
    renderLifecycleCharts(data);

    // 대시보드 표시
    showDashboard();

    // KPI 카드
    updateKPI(rmse, threshold, needs);

    // 모드 토글 동기화 (hidden checkbox, manual 고정)
    var modeToggle = document.getElementById("modeToggle");
    if (modeToggle) modeToggle.checked = false;

    // 재학습 토스트 표시
    var toast = document.getElementById("retrainToast");
    if (needs && toast) {
        toast.classList.add("show");
    }

    // 파이프라인 단계 처리
    if (needs) {
        setPipelineStep(4, "active");
    } else {
        setPipelineStep(4, "done");
        setPipelineStep(5, "done");
    }

    // 자동 모드 재학습 완료 처리
    if (data.auto_retrained) {
        setPipelineStep(4, "done");
        setPipelineStep(5, "done");
        showRetrainBanner(rmse, data.new_rmse, data.new_rmse < rmse);
        updateCompareBox(rmse, data.new_rmse);
    }

    // 섹션4 대시보드로 스크롤
    setTimeout(function () {
        document.getElementById("section-4").scrollIntoView({ behavior: "smooth" });
    }, 400);
}

// ── 리렌더 버튼 ─────────────────────────────────────────────────────────────
var replayBtn = document.getElementById("replayBtn");
if (replayBtn) {
    replayBtn.addEventListener("click", function () {
        if (lastChartPayload) renderLifecycleCharts(lastChartPayload);
    });
}

// ── Toast close ─────────────────────────────────────────────────────────────
var toastClose = document.getElementById("btnToastClose");
if (toastClose) {
    toastClose.addEventListener("click", function () {
        var toast = document.getElementById("retrainToast");
        if (toast) toast.classList.remove("show");
    });
}

// ── 재학습 완료 배너 ──────────────────────────────────────────────────────────
window.showRetrainBanner = function (oldRmse, newRmse, improved) {
    var color  = improved ? "#86efac" : "#fca5a5";
    var label  = improved ? "▼ 개선됨" : "▲ 악화됨";
    var banner = document.getElementById("retrainResultBanner");
    if (!banner) return;
    banner.innerHTML =
        "<strong style='color:" + color + ";'><i class='fas fa-sync-alt'></i>&nbsp; 재학습 완료</strong><br>"
        + "이전 RMSE: <strong>" + (oldRmse ? oldRmse.toFixed(2) : "—") + "</strong>"
        + " &nbsp;→&nbsp; 새 RMSE: <strong>" + newRmse.toFixed(2) + "</strong>"
        + " <span style='color:" + color + ";'>(" + label + ")</span>";
    banner.style.display = "block";
};

// ── 전후 비교 카드 업데이트 ───────────────────────────────────────────────────
window.updateCompareBox = function (oldRmse, newRmse) {
    var improved = newRmse < oldRmse;
    var cls = improved ? "c-improve" : "c-worsen";
    var arrow = improved ? "▼" : "▲";
    document.getElementById("compareBox").innerHTML =
        '<div class="compare-box">'
        + '<div class="compare-item">'
        + '<div class="c-label">재학습 전 RMSE</div>'
        + '<div class="c-val c-worsen">' + (oldRmse ? oldRmse.toFixed(2) : "—") + '</div>'
        + '</div>'
        + '<div class="compare-arrow">' + arrow + '</div>'
        + '<div class="compare-item">'
        + '<div class="c-label">재학습 후 RMSE</div>'
        + '<div class="c-val ' + cls + '">' + newRmse.toFixed(2) + '</div>'
        + '</div>'
        + '<div class="compare-item">'
        + '<div class="c-label">개선율</div>'
        + '<div class="c-val ' + cls + '">'
        + (oldRmse ? Math.abs(((newRmse - oldRmse) / oldRmse) * 100).toFixed(1) + "%" : "—")
        + '</div>'
        + '</div>'
        + '</div>';
};
