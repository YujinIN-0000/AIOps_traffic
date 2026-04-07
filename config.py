import os
from pathlib import Path

# 기본 경로 설정 (이 파일 기준으로 AIOps_traffic/server 디렉토리 사용)
_THIS_DIR = Path(__file__).resolve().parent  # → AIOps_traffic/
BASE_DIR = os.getenv("BASE_DIR", str(_THIS_DIR / "server"))

UPLOAD_DIR    = os.path.join(BASE_DIR, "uploaded_files")
MODEL_DIR     = os.path.join(BASE_DIR, "model")
IMAGE_DIR     = os.path.join(BASE_DIR, "view-model-architecture")
MODEL_IMG_DIR = os.path.join(BASE_DIR, "model-images")

# 파일 경로
ORIGINAL_DATA_PATH     = os.path.join(UPLOAD_DIR, "milan_telecom_timeseries_1.csv")  # 초기 학습 원본 데이터
DATA_PATH              = os.path.join(UPLOAD_DIR, "milan_telecom_final_1.csv")
MODEL_SAVE_PATH        = os.path.join(MODEL_DIR,  "telecom_lstm.keras")
MODEL_PLOT_PATH        = os.path.join(IMAGE_DIR,  "model.png")
MODEL_SHAPES_PLOT_PATH = os.path.join(IMAGE_DIR,  "shapes", "model_shapes.png")
PREDICTION_PLOT_PATH   = os.path.join(IMAGE_DIR,  "telecom_prediction.png")

# 하이퍼파라미터
LOOKBACK = 1008   # 7일 × 144 스텝/일 (10분 단위)

# 데이터 분할 비율 (train/test 두 구간만 사용)
TRAIN_RATIO = 0.72  # 72%: 학습
# TEST_RATIO = 0.28  # 나머지 28%: 평가 (자동 계산)

# 성능 임계값
RMSE_THRESHOLD = 200.0  # RMSE 임계값 (internet 트래픽 단위)

# 거부 이력 로그 경로
RETRAIN_LOG_PATH = os.path.join(BASE_DIR, "retrain_log.json")