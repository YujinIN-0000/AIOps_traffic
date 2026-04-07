#### 다음 실습 코드는 학습 목적으로만 사용 바랍니다. 문의 : audit@korea.ac.kr 임성열 Ph.D.

# pip install fastapi "uvicorn[standard]" pandas pytz python-multipart matplotlib
# pip install numpy scikit-learn keras tensorflow anthropic

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import asyncio
import base64
import importlib
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from config import (
    UPLOAD_DIR, IMAGE_DIR, MODEL_IMG_DIR,
    MODEL_SAVE_PATH, RMSE_THRESHOLD, RETRAIN_LOG_PATH,
    ORIGINAL_DATA_PATH
)

# .env 로드 (OPENAI_API_KEY 등)
try:
    from dotenv import load_dotenv
    load_dotenv()  # 현재 작업 디렉토리 기준
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # telecom/.env
except Exception:
    pass

# ------------------------------------------------------------------
# 경로 / 설정
# ------------------------------------------------------------------
STD_DIR    = Path(__file__).resolve().parent.parent   # .../model_serving_rpt copy
PUBLIC_DIR = STD_DIR / "public"

APP_ROOT_PATH = os.getenv("APP_ROOT_PATH", "").rstrip("/")
timezone      = pytz.timezone("Asia/Seoul")
CUMULATIVE_DATA_PATH = Path(UPLOAD_DIR) / "_cumulative_training.csv"

# ------------------------------------------------------------------
# 전역 상태 (단일 프로세스 데모용)
# ------------------------------------------------------------------
app_state: dict = {
    "last_rmse"          : None,   # float
    "last_dataset_path"  : None,   # Path
    "needs_retrain"      : False,
    "mode"               : "manual",   # "manual" | "auto"
    "retrain_history"    : [],         # [{timestamp, old_rmse, new_rmse}]
    "cumulative_dataset_path": str(CUMULATIVE_DATA_PATH),
}

router = APIRouter()


# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    for d in (PUBLIC_DIR, UPLOAD_DIR, IMAGE_DIR, MODEL_IMG_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    lifespan=lifespan,
    root_path=APP_ROOT_PATH,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

from fastapi.middleware.cors import CORSMiddleware


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static",      StaticFiles(directory=str(PUBLIC_DIR)),                name="static")
app.mount("/css",         StaticFiles(directory=str(PUBLIC_DIR / "css")),        name="css")
app.mount("/js",          StaticFiles(directory=str(PUBLIC_DIR / "js")),         name="js")
app.mount("/img",         StaticFiles(directory=str(PUBLIC_DIR / "img")),        name="img")
app.mount("/fontawesome", StaticFiles(directory=str(PUBLIC_DIR / "fontawesome")),name="fontawesome")


# ------------------------------------------------------------------
# 유틸
# ------------------------------------------------------------------
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    for f in (PUBLIC_DIR / "favicon.ico", PUBLIC_DIR / "favicon.png"):
        if f.exists():
            return FileResponse(str(f))
    return Response(status_code=204)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    return await call_next(request)


def _b64_png(path: Path) -> str:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


async def _read_csv_async(file_path: Path) -> pd.DataFrame:
    def _read():
        return pd.read_csv(file_path)
    return await asyncio.to_thread(_read)


def _append_retrain_log(record: dict):
    log_path = Path(RETRAIN_LOG_PATH)
    logs = []
    if log_path.exists():
        try:
            logs = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            logs = []
    logs.append(record)
    log_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_to_cumulative(new_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    누적 학습셋 갱신:
    - 최초: 원본 데이터(있으면) + 신규 업로드
    - 이후: 기존 누적 + 신규 업로드
    """
    frames = []
    prev_rows = 0
    original_rows = 0
    if CUMULATIVE_DATA_PATH.exists():
        try:
            prev_df = pd.read_csv(CUMULATIVE_DATA_PATH)
            prev_rows = len(prev_df)
            frames.append(prev_df)
        except Exception:
            pass
    else:
        original_path = Path(ORIGINAL_DATA_PATH)
        if original_path.exists():
            try:
                original_df = pd.read_csv(original_path)
                original_rows = len(original_df)
                frames.append(original_df)
            except Exception:
                pass

    frames.append(new_df)
    merged = pd.concat(frames, ignore_index=True) if len(frames) > 1 else new_df.copy()
    raw_rows = len(merged)
    # 동일 파일 재업로드 시 중복 누적 방지
    merged = merged.drop_duplicates().reset_index(drop=True)
    dedup_removed = raw_rows - len(merged)
    CUMULATIVE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(CUMULATIVE_DATA_PATH, index=False)
    stats = {
        "new_rows": len(new_df),
        "prev_rows": prev_rows,
        "original_rows": original_rows,
        "merged_rows": len(merged),
        "dedup_removed": dedup_removed,
    }
    return merged, stats


def _generate_template_report(rmse: float, needs: bool) -> str:
    status = "임계값 초과" if needs else "정상 범위"
    rec    = "재학습 및 재배포를 권고합니다." if needs else "현 모델 유지를 권고합니다."
    cause  = (
        "최근 트래픽 패턴이 학습 데이터와 상이하게 변화했거나, "
        "계절적 요인(명절·이벤트)으로 인한 급격한 트래픽 증가가 감지되었습니다."
        if needs else
        "모델 예측 성능이 안정적으로 유지되고 있습니다."
    )
    return (
        f"## 텔레콤 트래픽 예측 모델 성능 보고서\n\n"
        f"**현재 RMSE**: {rmse:.2f}  |  **임계값**: {RMSE_THRESHOLD}  |  **상태**: {status}\n\n"
        f"### 1. 성능 저하 원인 분석\n{cause}\n\n"
        f"### 2. 재배포 필요성 권고\n{rec}\n\n"
        f"### 3. 권고 사항\n"
        f"- 최근 7일 트래픽 데이터를 포함하여 재학습 수행\n"
        f"- 재학습 후 검증셋 RMSE가 임계값({RMSE_THRESHOLD}) 미만인지 확인\n"
        f"- 향후 예측 트래픽이 임계값을 초과할 경우, 망 스케일링 시스템과 연동하여 "
        f"자동 알람 및 용량 확장 트리거를 제공할 수 있습니다.\n"
    )


# ------------------------------------------------------------------
# 헬스체크 / 루트
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "root_path": APP_ROOT_PATH or "/"}


@app.get("/")
def root():
    index_html = PUBLIC_DIR / "index.html"
    if not index_html.exists():
        return {"message": "public/index.html not found."}
    html = index_html.read_text(encoding="utf-8", errors="ignore")
    rp = APP_ROOT_PATH or "/"
    if "<base" not in html.lower():
        base_href = rp if rp.endswith("/") else rp + "/"
        html = html.replace("<head>", f'<head><base href="{base_href}">', 1)
    return HTMLResponse(content=html)


@app.get("/report.html")
def report_page():
    report_html = PUBLIC_DIR / "report.html"
    if not report_html.exists():
        return {"message": "public/report.html not found."}
    html = report_html.read_text(encoding="utf-8", errors="ignore")
    rp = APP_ROOT_PATH or "/"
    if "<base" not in html.lower():
        base_href = rp if rp.endswith("/") else rp + "/"
        html = html.replace("<head>", f'<head><base href="{base_href}">', 1)
    return HTMLResponse(content=html)


# ------------------------------------------------------------------
# 초기 대시보드 (모델 + 원본 데이터로 초기 예측 결과 반환)
# ------------------------------------------------------------------
@router.get("/init")
async def init_dashboard():
    """페이지 로드 시 호출 — 기존 모델로 원본 데이터 추론 결과 반환"""
    if not Path(MODEL_SAVE_PATH).exists():
        return {"model_ready": False, "message": "저장된 모델 없음"}

    # 가장 최근 차트 결과가 있으면 그대로 반환 (재학습/업로드 반영)
    if app_state.get("last_chart_payload"):
        payload = app_state["last_chart_payload"]
        return {
            "model_ready"             : True,
            "skip_retrain_check"      : True,
            "result_evaluating_LSTM"  : payload.get("result_evaluating_LSTM"),
            "needs_retrain"           : False,
            "rmse_threshold"          : RMSE_THRESHOLD,
            "mode"                    : app_state["mode"],
            "actual_series"           : payload.get("actual_series", []),
            "pred_series"             : payload.get("pred_series", []),
            "time_index"              : payload.get("time_index", []),
            "retrain_history"         : app_state.get("retrain_history", []),
        }

    # 최근 업로드 데이터가 있으면 우선 사용
    last_path = app_state.get("last_dataset_path")
    candidate_path = Path(last_path) if last_path else Path(ORIGINAL_DATA_PATH)
    if not candidate_path.exists():
        return {"model_ready": False, "message": "원본 데이터 없음"}

    try:
        weight_mod = importlib.import_module(".weight_used_model", package=__package__)
        dataset    = await _read_csv_async(candidate_path)
        plot_path, rmse, actual_list, pred_list, date_list = await asyncio.to_thread(weight_mod.process, dataset)

        retrain_needed = weight_mod.needs_retrain(rmse)
        app_state["last_rmse"]     = rmse
        app_state["needs_retrain"] = retrain_needed
        # last_dataset_path는 기존 값을 유지

        img = Path(plot_path)
        return {
            "model_ready"             : True,
            "skip_retrain_check"      : True,   # 초기 로드: 학습 데이터로 추론 → 재학습 판단 불필요
            "result_visualizing_LSTM" : _b64_png(img),
            "result_evaluating_LSTM"  : round(rmse, 4),
            "needs_retrain"           : False,
            "rmse_threshold"          : RMSE_THRESHOLD,
            "mode"                    : app_state["mode"],
            "actual_series"           : actual_list,
            "pred_series"             : pred_list,
            "time_index"              : date_list,
            "retrain_history"         : app_state.get("retrain_history", []),
        }
    except Exception as e:
        return {"model_ready": False, "message": str(e)}


# ------------------------------------------------------------------
# 업로드 / 예측
# ------------------------------------------------------------------
@router.post("/upload")
async def post_data_set(file: UploadFile = File(...)):
    """
    CSV 업로드 → 텔레콤 LSTM 예측 수행
    - 저장된 모델(telecom_lstm.keras)이 없으면 먼저 학습
    - RMSE > RMSE_THRESHOLD 이면 needs_retrain=True
    - auto 모드이면 즉시 /retrain 로직 실행
    """
    try:
        current_time  = datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
        new_filename  = f"{current_time}_{file.filename}"
        file_location = Path(UPLOAD_DIR) / new_filename

        contents = await file.read()
        await asyncio.to_thread(file_location.write_bytes, contents)

        dataset = await _read_csv_async(file_location)
        cumulative_dataset, merge_stats = await asyncio.to_thread(_merge_to_cumulative, dataset)
        print(
            "[upload] rows "
            f"new_csv={merge_stats['new_rows']}, "
            f"prev_cumulative={merge_stats['prev_rows']}, "
            f"original_seed={merge_stats['original_rows']}, "
            f"total_cumulative={merge_stats['merged_rows']}, "
            f"dedup_removed={merge_stats['dedup_removed']}"
        )

        weight_mod = importlib.import_module(".weight_used_model", package=__package__)
        model_mod  = importlib.import_module(".model",             package=__package__)

        # 저장된 모델이 없으면 최초 학습
        if not Path(MODEL_SAVE_PATH).exists():
            print("[upload] No saved model found. Training from scratch...")
            await asyncio.to_thread(model_mod.train_and_save, cumulative_dataset)

        # 누적 데이터(기존+신규) 전체를 그래프용으로 사용
        combined_df = cumulative_dataset

        # RMSE는 업로드 데이터 기준
        plot_path, rmse, actual_list, pred_list, date_list = await asyncio.to_thread(weight_mod.process, dataset)
        img1 = Path(plot_path)
        if not img1.exists():
            raise HTTPException(status_code=500, detail=f"Plot not found: {img1}")

        retrain_needed = weight_mod.needs_retrain(rmse)

        # 그래프는 합산 데이터 기준으로 교체
        if combined_df is not None and len(combined_df) > len(dataset):
            comb_plot, _, comb_actual, comb_pred, comb_dates = await asyncio.to_thread(
                weight_mod.process, combined_df
            )
            comb_img = Path(comb_plot)
            if comb_img.exists():
                img1 = comb_img
            actual_list = comb_actual
            pred_list   = comb_pred
            date_list   = comb_dates

        # 전역 상태 갱신
        app_state["last_rmse"]         = rmse
        app_state["last_dataset_path"] = str(file_location)
        app_state["needs_retrain"]     = retrain_needed
        app_state["last_chart_payload"] = {
            "actual_series": actual_list,
            "pred_series"  : pred_list,
            "time_index"   : date_list,
            "rmse_threshold": RMSE_THRESHOLD,
            "result_evaluating_LSTM": round(rmse, 4),
        }

        result = {
            "result_visualizing_LSTM" : _b64_png(img1),
            "result_evaluating_LSTM"  : round(rmse, 4),
            "needs_retrain"           : retrain_needed,
            "rmse_threshold"          : RMSE_THRESHOLD,
            "saved_filename"          : new_filename,
            "mode"                    : app_state["mode"],
            "actual_series"           : actual_list,
            "pred_series"             : pred_list,
            "time_index"              : date_list,
            "retrain_history"         : app_state.get("retrain_history", []),
        }

        # 자동 모드: 임계 초과 시 즉시 재학습
        if retrain_needed and app_state["mode"] == "auto":
            print("[upload] Auto mode: triggering retrain...")
            await asyncio.to_thread(model_mod.train_and_save, cumulative_dataset)
            new_plot, new_rmse, new_actual, new_pred, new_dates = await asyncio.to_thread(weight_mod.process, dataset)
            app_state["last_rmse"]     = new_rmse
            app_state["needs_retrain"] = new_rmse > RMSE_THRESHOLD
            app_state["retrain_history"].append({
                "timestamp": datetime.now(timezone).isoformat(),
                "trigger"  : "auto",
                "old_rmse" : round(rmse, 4),
                "new_rmse" : round(new_rmse, 4),
            })
            result["auto_retrained"] = True
            result["new_rmse"]       = round(new_rmse, 4)
            # 재학습 후 그래프도 합산 데이터 기준으로 교체
            if combined_df is not None and len(combined_df) > len(dataset):
                comb_plot, _, comb_actual, comb_pred, comb_dates = await asyncio.to_thread(
                    weight_mod.process, combined_df
                )
                comb_img = Path(comb_plot)
                result["result_visualizing_LSTM"] = _b64_png(comb_img) if comb_img.exists() else _b64_png(Path(new_plot))
                result["actual_series"]  = comb_actual
                result["pred_series"]    = comb_pred
                result["time_index"]     = comb_dates
            else:
                result["result_visualizing_LSTM"] = _b64_png(Path(new_plot))
                result["actual_series"]  = new_actual
                result["pred_series"]    = new_pred
                result["time_index"]     = new_dates
            app_state["last_chart_payload"] = {
                "actual_series": result["actual_series"],
                "pred_series"  : result["pred_series"],
                "time_index"   : result["time_index"],
                "rmse_threshold": RMSE_THRESHOLD,
                "result_evaluating_LSTM": round(new_rmse, 4),
            }

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# 재학습 (운영자 승인 → Human-in-the-Loop)
# ------------------------------------------------------------------
@router.post("/retrain")
async def retrain():
    """운영자 승인 후 재학습 실행"""
    dataset_path = app_state.get("last_dataset_path")
    if not dataset_path or not Path(dataset_path).exists():
        raise HTTPException(status_code=400, detail="업로드된 데이터셋이 없습니다. 먼저 /upload를 실행하세요.")

    try:
        model_mod   = importlib.import_module(".model",            package=__package__)
        weight_mod  = importlib.import_module(".weight_used_model", package=__package__)
        uploaded_df = await _read_csv_async(Path(dataset_path))
        old_rmse    = app_state.get("last_rmse", 0.0)

        # 누적 학습 데이터(기존+신규 전체) 기준으로 재학습
        if CUMULATIVE_DATA_PATH.exists():
            dataset = await _read_csv_async(CUMULATIVE_DATA_PATH)
            print(
                "[retrain] rows "
                f"uploaded_csv={len(uploaded_df)}, "
                f"cumulative_total={len(dataset)}"
            )
        else:
            dataset, merge_stats = await asyncio.to_thread(_merge_to_cumulative, uploaded_df)
            print(
                "[retrain] rows "
                f"uploaded_csv={merge_stats['new_rows']}, "
                f"prev_cumulative={merge_stats['prev_rows']}, "
                f"original_seed={merge_stats['original_rows']}, "
                f"total_cumulative={merge_stats['merged_rows']}, "
                f"dedup_removed={merge_stats['dedup_removed']}"
            )

        await asyncio.to_thread(model_mod.train_and_save, dataset)
        # 평가/시각화는 업로드 데이터 기준으로 수행
        new_plot, new_rmse, actual_list, pred_list, date_list = await asyncio.to_thread(
            weight_mod.process, uploaded_df
        )
        # 그래프는 누적 데이터 기준으로 교체
        if len(dataset) > len(uploaded_df):
            comb_plot, _, comb_actual, comb_pred, comb_dates = await asyncio.to_thread(
                weight_mod.process, dataset
            )
            new_plot   = comb_plot
            actual_list = comb_actual
            pred_list   = comb_pred
            date_list   = comb_dates

        app_state["last_rmse"]     = new_rmse
        app_state["needs_retrain"] = new_rmse > RMSE_THRESHOLD
        app_state["retrain_history"].append({
            "timestamp": datetime.now(timezone).isoformat(),
            "trigger"  : "manual_approve",
            "old_rmse" : round(old_rmse, 4),
            "new_rmse" : round(new_rmse, 4),
        })
        app_state["last_chart_payload"] = {
            "actual_series": actual_list,
            "pred_series"  : pred_list,
            "time_index"   : date_list,
            "rmse_threshold": RMSE_THRESHOLD,
            "result_evaluating_LSTM": round(new_rmse, 4),
        }

        img = Path(new_plot)
        return {
            "status"  : "completed",
            "old_rmse": round(old_rmse, 4),
            "new_rmse": round(new_rmse, 4),
            "improved": new_rmse < old_rmse,
            "needs_retrain": app_state["needs_retrain"],
            "result_visualizing_LSTM": _b64_png(img) if img.exists() else None,
            "actual_series": actual_list,
            "pred_series"  : pred_list,
            "time_index"   : date_list,
            "retrain_history": app_state.get("retrain_history", []),
            "message" : f"재학습 완료. RMSE {old_rmse:.2f} → {new_rmse:.2f}",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# 재학습 거부 (현 모델 유지 + 로그 기록)
# ------------------------------------------------------------------
@router.post("/reject")
async def reject_retrain():
    """운영자 거부 → 현 모델 유지, 거부 이력 로깅"""
    record = {
        "timestamp": datetime.now(timezone).isoformat(),
        "action"   : "reject",
        "rmse"     : round(app_state.get("last_rmse") or 0.0, 4),
        "rmse_threshold": RMSE_THRESHOLD,
    }
    await asyncio.to_thread(_append_retrain_log, record)
    app_state["needs_retrain"] = False
    return {"status": "rejected", "message": "현 모델을 유지합니다. 거부 이력이 기록되었습니다.", **record}


# ------------------------------------------------------------------
# LLM 성능 보고서
# ------------------------------------------------------------------
@router.get("/report")
async def get_report():
    """LLM 기반 모델 성능 보고서 생성 및 반환"""
    rmse  = app_state.get("last_rmse")
    needs = app_state.get("needs_retrain", False)

    if rmse is None:
        return {"report": "아직 예측 결과가 없습니다. 먼저 CSV를 업로드해주세요.", "rmse": None, "needs_retrain": False}

    prompt = (
        f"당신은 통신사 네트워크 운영팀을 위한 AI 분석가입니다.\n"
        f"밀라노 텔레콤 인터넷 트래픽 예측 모델의 성능 보고서를 한국어로 작성해주세요.\n\n"
        f"현재 모델 성능:\n"
        f"- RMSE: {rmse:.2f}\n"
        f"- RMSE 임계값: {RMSE_THRESHOLD}\n"
        f"- 상태: {'임계값 초과 (재학습 필요)' if needs else '정상 범위'}\n\n"
        f"다음 항목을 포함해주세요:\n"
        f"1. 성능 저하 원인 분석\n"
        f"2. 재배포 필요성 권고\n"
        f"3. 구체적인 권고 사항\n"
        f"4. 향후 망 스케일링 연동 가능성 언급\n"
    )

    if not needs:
        report_text = "임계값이 초과하지 않아 보고서가 작성되지 않았습니다."
    else:
        report_text = None
        try:
            import requests
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024,
                },
                timeout=30,
            )
            resp.raise_for_status()
            report_text = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[report] OpenAI LLM unavailable ({e})")
            report_text = f"보고서 생성 실패: {e}"

    return {
        "report"       : report_text,
        "rmse"         : round(rmse, 4),
        "rmse_threshold": RMSE_THRESHOLD,
        "needs_retrain": needs,
    }


# ------------------------------------------------------------------
# 모드 전환 (자동 / 수동)
# ------------------------------------------------------------------
@router.post("/set-mode")
async def set_mode(body: dict):
    mode = body.get("mode", "manual")
    if mode not in ("auto", "manual"):
        raise HTTPException(status_code=400, detail="mode must be 'auto' or 'manual'")
    app_state["mode"] = mode
    return {"status": "ok", "mode": mode}


# ------------------------------------------------------------------
# 다운로드 / 뷰
# ------------------------------------------------------------------
@router.get("/download")
async def download():
    weight_mod = importlib.import_module(".weight_used_model", package=__package__)
    img_path   = Path(weight_mod.get_stock_png())
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {img_path}")
    return FileResponse(path=str(img_path), media_type="application/octet-stream", filename="telecom_prediction.png")


@router.get("/download_shapes")
async def download_shapes():
    weight_mod = importlib.import_module(".weight_used_model", package=__package__)
    img_path   = Path(weight_mod.get_model_shapes_png())
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {img_path}")
    return FileResponse(path=str(img_path), media_type="application/octet-stream", filename="model_shapes.png")


@router.get("/view-download")
async def view_download():
    weight_mod = importlib.import_module(".weight_used_model", package=__package__)
    img_path   = Path(weight_mod.get_stock_png())
    img_b64    = _b64_png(img_path)
    return HTMLResponse(content=f"""
        <html><body>
            <h1>Telecom Traffic Prediction</h1>
            <img src="{img_b64}" alt="Telecom Prediction" style="max-width:100%;" />
        </body></html>
    """)


app.include_router(router)

# python -m uvicorn server_model.main:app --port 8001 --reload
