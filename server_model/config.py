import os
from pathlib import Path

# 기본 경로 설정 (환경 변수에서 가져오거나 기본값 사용)
# server_model 폴더의 상위 폴더(AIOps_traffic)/server 를 BASE_DIR로 사용
_THIS_DIR = Path(__file__).resolve().parent.parent  # → AIOps_traffic/
BASE_DIR = os.getenv("BASE_DIR", str(_THIS_DIR / "server"))

UPLOAD_DIR    = os.path.join(BASE_DIR, "uploaded_files")
MODEL_DIR     = os.path.join(BASE_DIR, "model")
IMAGE_DIR     = os.path.join(BASE_DIR, "view-model-architecture")
MODEL_IMG_DIR = os.path.join(BASE_DIR, "model-images")

# 파일 경로
DATA_PATH              = os.path.join(UPLOAD_DIR, "milan_telecom_final_1.csv")
MODEL_SAVE_PATH        = os.path.join(MODEL_DIR,  "result", "stock_lstm_model.keras")
MODEL_PLOT_PATH        = os.path.join(IMAGE_DIR,  "model.png")
MODEL_SHAPES_PLOT_PATH = os.path.join(IMAGE_DIR,  "shapes", "model_shapes.png")
PREDICTION_PLOT_PATH   = os.path.join(IMAGE_DIR,  "telecom_prediction.png")

# 하이퍼파라미터
LOOKBACK    = 1008   # 7일 × 144 스텝/일 (10분 단위)
TRAIN_RATIO = 0.72
# TEST_RATIO = 0.28  # 나머지 자동 계산

# 성능 임계값
RMSE_THRESHOLD = 200.0

# 거부 이력 로그 경로
RETRAIN_LOG_PATH = os.path.join(BASE_DIR, "retrain_log.json")

# 원본 데이터 경로 (초기 대시보드용)
ORIGINAL_DATA_PATH = os.path.join(BASE_DIR, "uploaded_files", "milan_telecom_timeseries_1.csv")