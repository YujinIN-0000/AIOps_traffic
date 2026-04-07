// ── 페이지 로드 시 초기 대시보드 ─────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", function () {
    fetch("http://localhost:8001/init")
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (!data.model_ready) return;  // 모델 없으면 빈 화면 유지
            handleUploadResult(data);
        })
        .catch(function (err) {
            console.warn("[init] 초기 대시보드 로드 실패:", err);
        });
});

// 다운로드 버튼 제거됨 (대시보드 간소화)
