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
    var noRecord = histDiv.querySelector("p");
    if (noRecord) histDiv.innerHTML = "";

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
