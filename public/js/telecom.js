// telecom.js — 재학습 승인/거부, LLM 보고서, 모드 전환
const API = "http://localhost:8001";

// ── 보고서 보기 (토스트 → report.html 이동) ──────────────────────────────────
var btnViewReport = document.getElementById("btnViewReport");
if (btnViewReport) {
    btnViewReport.addEventListener("click", function () {
        var toast = document.getElementById("retrainToast");
        if (toast) toast.classList.remove("show");
        window.location.href = "/report.html";
    });
}

// ── LLM 보고서 (대시보드 내) ───────────────────────────────────────────────
var btnReport = document.getElementById("btnReport");
if (btnReport) {
    btnReport.addEventListener("click", function () {
        var btn = this;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>&nbsp; 보고서 생성 중...';
        btn.disabled  = true;

        setPipelineStep(4, "active");

        fetch(API + "/report")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                btn.innerHTML = '<i class="fas fa-file-alt"></i>&nbsp; LLM 보고서 보기';
                btn.disabled  = false;
                setPipelineStep(4, "done");

                var panel = document.getElementById("reportPanel");
                var reportEl = document.getElementById("reportContent");
                if (window.marked && data.report) {
                    reportEl.innerHTML = window.marked.parse(data.report);
                } else {
                    reportEl.textContent = data.report;
                }
                panel.style.display = "block";
                panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
            })
            .catch(function (err) {
                btn.innerHTML = '<i class="fas fa-file-alt"></i>&nbsp; LLM 보고서 보기';
                btn.disabled  = false;
                alert("보고서 생성 중 오류: " + err.message);
            });
    });
}

// ── 모드 전환 토글 ────────────────────────────────────────────────────────────
var modeToggle = document.getElementById("modeToggle");
if (modeToggle) modeToggle.addEventListener("change", function () {
    var mode = this.checked ? "auto" : "manual";
    document.getElementById("modeLabel").textContent = this.checked ? "자동 모드" : "수동 모드";

    // KPI 카드 모드 업데이트
    document.getElementById("kpiModeVal").textContent = this.checked ? "자동" : "수동";
    document.getElementById("kpiModeSub").textContent =
        this.checked ? "임계 초과 즉시 재학습" : "운영자 승인 필요";

    fetch(API + "/set-mode", {
        method  : "POST",
        headers : { "Content-Type": "application/json" },
        body    : JSON.stringify({ mode: mode }),
    }).catch(function (err) {
        console.warn("mode set failed:", err);
    });
});

// ── 재학습 이력 추가 ──────────────────────────────────────────────────────────
function appendRetrainHistory(record) {
    var histDiv  = document.getElementById("retrainHistory");
    if (!histDiv) return;
    var empty = histDiv.querySelector(".history-empty");
    if (empty) histDiv.innerHTML = "";

    var improved = record.improved !== undefined
        ? record.improved
        : record.new_rmse < record.old_rmse;
    var arrow = improved ? "▼ 개선" : "▲ 악화";
    var color = improved ? "#86efac" : "#fca5a5";

    var row = document.createElement("div");
    row.className = "history-row";
    row.innerHTML =
        '<span class="history-ts">' + record.timestamp + '</span>'
        + '<span style="color:#aaa;">[' + record.trigger + ']</span>'
        + '<span>'
        + '<strong>' + (record.old_rmse || 0).toFixed(2) + '</strong>'
        + ' → <strong>' + (record.new_rmse || 0).toFixed(2) + '</strong>'
        + '</span>'
        + '<span style="color:' + color + '; font-weight:700;">' + arrow + '</span>';

    histDiv.prepend(row);
}
window.appendRetrainHistory = appendRetrainHistory;

// ── 대시보드 토스트: 재학습 승인/거부 ──────────────────────────────────────────
var btnRetrain = document.getElementById("btnRetrain");
var btnReject  = document.getElementById("btnReject");

function _setToastButtonsLoading(loading) {
    if (btnRetrain) {
        btnRetrain.disabled = loading;
        btnRetrain.innerHTML = loading
            ? '<i class="fa fa-spinner fa-spin"></i> 재학습 중...'
            : '<i class="fa fa-check"></i> 확인';
    }
    if (btnReject) {
        btnReject.disabled = loading;
    }
}

if (btnRetrain) {
    btnRetrain.addEventListener("click", function () {
        _setToastButtonsLoading(true);
        fetch(API + "/retrain", { method: "POST" })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                _setToastButtonsLoading(false);
                var toast = document.getElementById("retrainToast");
                if (toast) toast.classList.remove("show");

                // 그래프/카드 업데이트
                if (typeof renderLifecycleCharts === "function") {
                    renderLifecycleCharts(data);
                }
                if (typeof updateKPI === "function") {
                    updateKPI(data.new_rmse, window.lastRmseThreshold || 200, data.needs_retrain);
                }
                if (window.showRetrainBanner) {
                    window.showRetrainBanner(data.old_rmse, data.new_rmse, data.improved);
                }
                if (window.updateCompareBox) {
                    window.updateCompareBox(data.old_rmse, data.new_rmse);
                }
                if (Array.isArray(data.retrain_history) && window.renderRetrainHistoryList) {
                    window.renderRetrainHistoryList(data.retrain_history);
                } else {
                    appendRetrainHistory({
                        timestamp: new Date().toISOString(),
                        trigger: "manual_approve",
                        old_rmse: data.old_rmse,
                        new_rmse: data.new_rmse,
                        improved: data.improved
                    });
                }

                if (typeof setPipelineStep === "function") {
                    setPipelineStep(4, "done");
                    setPipelineStep(5, "done");
                }
                window.lastRmse = data.new_rmse;
            })
            .catch(function (err) {
                _setToastButtonsLoading(false);
                alert("재학습 중 오류: " + err.message);
            });
    });
}

if (btnReject) {
    btnReject.addEventListener("click", function () {
        btnReject.disabled = true;
        if (btnRetrain) btnRetrain.disabled = true;
        fetch(API + "/reject", { method: "POST" })
            .then(function (r) { return r.json(); })
            .then(function () {
                var toast = document.getElementById("retrainToast");
                if (toast) toast.classList.remove("show");
                if (typeof setPipelineStep === "function") {
                    setPipelineStep(4, "done");
                    setPipelineStep(5, "done");
                }
            })
            .catch(function (err) {
                btnReject.disabled = false;
                if (btnRetrain) btnRetrain.disabled = false;
                alert("거부 처리 중 오류: " + err.message);
            });
    });
}
