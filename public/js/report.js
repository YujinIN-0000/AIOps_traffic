const API_BASE = "http://localhost:8001";

function renderReport(data) {
    const meta    = document.getElementById("reportMeta");
    const content = document.getElementById("reportContent");
    const actionBox = document.getElementById("retrainActionBox");

    if (!data) {
        meta.textContent = "데이터 없음";
        content.textContent = "보고서를 불러오지 못했습니다.";
        return;
    }
    const rmse      = data.rmse;
    const threshold = data.rmse_threshold;
    const needs     = data.needs_retrain;

    if (rmse == null) {
        meta.textContent = "예측 결과 없음 — 먼저 대시보드에서 CSV를 업로드하세요.";
    } else {
        meta.textContent = `RMSE ${rmse} | 임계값 ${threshold} | 상태 ${needs ? "임계값 초과" : "정상"}`;
    }
    if (window.marked && data.report) {
        content.innerHTML = window.marked.parse(data.report);
    } else {
        content.textContent = data.report || "보고서가 없습니다.";
    }

    // 임계값 초과 시 재학습 결정 버튼 표시
    if (needs && actionBox) {
        actionBox.style.display = "block";
    }
}

window.addEventListener("DOMContentLoaded", function () {
    fetch(API_BASE + "/report")
        .then(function (r) { return r.json(); })
        .then(renderReport)
        .catch(function (err) {
            renderReport({ report: "보고서 로드 실패: " + err.message });
        });

    // ── 재학습 확인 ──────────────────────────────────────────────────────────
    var btnRetrain = document.getElementById("btnReportRetrain");
    var btnReject  = document.getElementById("btnReportReject");
    var resultMsg  = document.getElementById("retrainResultMsg");

    if (btnRetrain) {
        btnRetrain.addEventListener("click", function () {
            btnRetrain.disabled = true;
            btnReject.disabled  = true;
            btnRetrain.innerHTML = '<i class="fa fa-spinner fa-spin"></i> 재학습 중...';

            fetch(API_BASE + "/retrain", { method: "POST" })
                .then(function (r) {
                    if (!r.ok) return r.json().then(function (e) { throw new Error(e.detail); });
                    return r.json();
                })
                .then(function (data) {
                    var improved = data.improved;
                    var color    = improved ? "#86efac" : "#fca5a5";
                    var label    = improved ? "▼ 개선됨" : "▲ 악화됨";
                    resultMsg.style.display = "block";
                    resultMsg.style.color   = color;
                    resultMsg.innerHTML =
                        "<strong>재학습 완료</strong> — "
                        + "이전 RMSE: <strong>" + (data.old_rmse || 0).toFixed(2) + "</strong>"
                        + " → 새 RMSE: <strong>" + (data.new_rmse || 0).toFixed(2) + "</strong>"
                        + " <span>(" + label + ")</span>";
                    btnRetrain.innerHTML = '<i class="fa fa-check"></i> 완료';
                    document.getElementById("retrainActionBox").querySelector(".retrain-btn-row").style.display = "none";
                })
                .catch(function (err) {
                    btnRetrain.disabled = false;
                    btnReject.disabled  = false;
                    btnRetrain.innerHTML = '<i class="fa fa-check"></i> 확인 (재학습)';
                    resultMsg.style.display = "block";
                    resultMsg.style.color   = "#fca5a5";
                    resultMsg.textContent   = "오류: " + err.message;
                });
        });
    }

    // ── 거부 ─────────────────────────────────────────────────────────────────
    if (btnReject) {
        btnReject.addEventListener("click", function () {
            btnRetrain.disabled = true;
            btnReject.disabled  = true;

            fetch(API_BASE + "/reject", { method: "POST" })
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    resultMsg.style.display = "block";
                    resultMsg.style.color   = "var(--text-2)";
                    resultMsg.innerHTML =
                        "<i class='fa fa-times'></i> 현 모델을 유지합니다. "
                        + "거부 이력이 기록되었습니다. (RMSE: "
                        + (data.rmse ? data.rmse.toFixed(2) : "—") + ")";
                    document.getElementById("retrainActionBox").querySelector(".retrain-btn-row").style.display = "none";
                })
                .catch(function (err) {
                    btnRetrain.disabled = false;
                    btnReject.disabled  = false;
                    resultMsg.style.display = "block";
                    resultMsg.style.color   = "#fca5a5";
                    resultMsg.textContent   = "오류: " + err.message;
                });
        });
    }
});
