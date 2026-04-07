#### 다음 실습 코드는 학습 목적으로만 사용 바랍니다. 문의 : audit@korea.ac.kr 임성열 Ph.D.

import os
os.environ["MPLBACKEND"] = "Agg"

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout

from config import (
    MODEL_SAVE_PATH, MODEL_PLOT_PATH, MODEL_SHAPES_PLOT_PATH,
    PREDICTION_PLOT_PATH, LOOKBACK, TRAIN_RATIO
)

def _get_splits(n: int):
    """데이터 크기 n에 맞게 train/test 스텝 수 동적 계산"""
    train = int(n * TRAIN_RATIO)
    test  = n - train
    return train, test

FEATURE_COLS = ["internet", "smsin", "callin", "hour", "dayofweek", "is_weekend"]
TARGET_IDX   = 0  # internet 컬럼 인덱스


# ------------------------------------------------------------------
# 내부 유틸
# ------------------------------------------------------------------

def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """밀라노 텔레콤 CSV → datetime 기준 집계 + 시간 피처 추가"""
    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # GridID 5161, 국가코드별 여러 행 → datetime 기준 합산
    numeric = ["internet", "smsin", "callin"]
    agg = df[numeric].groupby(df.index).sum(min_count=1).fillna(0)

    agg["hour"]      = agg.index.hour
    agg["dayofweek"] = agg.index.dayofweek
    agg["is_weekend"] = (agg.index.dayofweek >= 5).astype(int)

    return agg[FEATURE_COLS]


def _make_sequences(scaled: np.ndarray, start: int, count: int):
    """슬라이딩 윈도우 시퀀스 생성"""
    X, y = [], []
    for i in range(start, start + count):
        if i < LOOKBACK:
            continue
        X.append(scaled[i - LOOKBACK:i])
        y.append(scaled[i, TARGET_IDX])
    return np.array(X), np.array(y)


def _plot_predictions(dates, actual: np.ndarray, predicted: np.ndarray) -> str:
    """실제 vs 예측 그래프 저장 (주말 구간 음영 포함)"""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(actual,    color="red",  label="Actual Traffic",    linewidth=0.8, alpha=0.9)
    ax.plot(predicted, color="blue", label="Predicted Traffic", linewidth=0.8, alpha=0.8)

    # 주말 구간 음영
    if dates is not None and len(dates) > 0:
        weekend_mask = pd.Series(dates).dt.dayofweek >= 5
        in_block, blk_start = False, 0
        for idx, is_wknd in enumerate(weekend_mask):
            if is_wknd and not in_block:
                blk_start, in_block = idx, True
            elif not is_wknd and in_block:
                ax.axvspan(blk_start, idx, alpha=0.13, color="gray")
                in_block = False
        if in_block:
            ax.axvspan(blk_start, len(weekend_mask), alpha=0.13, color="gray")

    weekend_patch = mpatches.Patch(color="gray", alpha=0.3, label="Weekend")
    ax.legend(handles=[
        plt.Line2D([0], [0], color="red",  label="Actual Traffic"),
        plt.Line2D([0], [0], color="blue", label="Predicted Traffic"),
        weekend_patch,
    ])
    ax.set_title("Milan Telecom Internet Traffic Prediction (Actual vs Predicted)", fontsize=13)
    ax.set_xlabel("Time Step (10-min intervals)")
    ax.set_ylabel("Internet Traffic")

    os.makedirs(os.path.dirname(PREDICTION_PLOT_PATH), exist_ok=True)
    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT_PATH, dpi=120)
    plt.close(fig)
    return PREDICTION_PLOT_PATH


# ------------------------------------------------------------------
# 학습 (POST /retrain 에서 호출)
# ------------------------------------------------------------------

def train_and_save(df: pd.DataFrame) -> tuple:
    """텔레콤 LSTM 모델 학습 후 저장. (plot_path, rmse) 반환"""
    agg = _preprocess(df)
    data = agg.values  # shape: (N, 6)

    n = len(data)
    train_steps, test_steps = _get_splits(n)
    print(f"[train] 전체={n}, train={train_steps}, test={test_steps}")

    # Scaler는 train 구간만으로 fit (데이터 누수 방지)
    train_data = data[:train_steps]
    sc = MinMaxScaler(feature_range=(0, 1))
    sc.fit(train_data)
    full_scaled = sc.transform(data)

    # train 시퀀스 생성
    X_train, y_train = _make_sequences(full_scaled, LOOKBACK, train_steps - LOOKBACK)

    regressor = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(LOOKBACK, len(FEATURE_COLS))),
        Dropout(0.2),
        LSTM(units=64, return_sequences=True),
        Dropout(0.2),
        LSTM(units=32),
        Dropout(0.2),
        Dense(units=1),
    ])
    regressor.compile(optimizer="adam", loss="mean_squared_error")
    regressor.fit(X_train, y_train, epochs=3, batch_size=64, verbose=1)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    regressor.save(MODEL_SAVE_PATH)
    print(f"[model] saved → {MODEL_SAVE_PATH}")

    # 모델 구조 이미지 (pydot/graphviz 없으면 건너뜀)
    try:
        from keras.utils import plot_model
        os.makedirs(os.path.dirname(MODEL_SHAPES_PLOT_PATH), exist_ok=True)
        plot_model(regressor, to_file=MODEL_PLOT_PATH)
        plot_model(regressor, to_file=MODEL_SHAPES_PLOT_PATH, show_shapes=True)
    except Exception:
        pass

    # 최종 평가는 test 구간만 사용
    test_start = train_steps
    X_test, _ = _make_sequences(full_scaled, test_start, test_steps)

    pred_scaled = regressor.predict(X_test, verbose=0)

    sc_target = MinMaxScaler(feature_range=(0, 1))
    sc_target.fit(train_data[:, TARGET_IDX:TARGET_IDX + 1])
    pred   = sc_target.inverse_transform(pred_scaled)
    actual = data[test_start:test_start + len(pred), TARGET_IDX:TARGET_IDX + 1]

    rmse = math.sqrt(mean_squared_error(actual, pred))

    dates = agg.index[test_start:test_start + len(pred)]
    plot_path = _plot_predictions(dates, actual, pred)

    return plot_path, rmse


# ------------------------------------------------------------------
# 추론 (weight_used_model 과 동일 인터페이스 — 두 번째 모델로 사용)
# ------------------------------------------------------------------

def process(df: pd.DataFrame) -> tuple:
    """저장된 모델로 추론. (plot_path, rmse) 반환"""
    agg = _preprocess(df)
    data = agg.values

    n = len(data)
    train_steps, test_steps = _get_splits(n)

    train_data = data[:train_steps]
    sc = MinMaxScaler(feature_range=(0, 1))
    sc.fit(train_data)
    full_scaled = sc.transform(data)

    test_start = train_steps
    X_test, _ = _make_sequences(full_scaled, test_start, test_steps)

    mdl = load_model(MODEL_SAVE_PATH)
    pred_scaled = mdl.predict(X_test, verbose=0)

    sc_target = MinMaxScaler(feature_range=(0, 1))
    sc_target.fit(train_data[:, TARGET_IDX:TARGET_IDX + 1])
    pred   = sc_target.inverse_transform(pred_scaled)
    actual = data[test_start:test_start + len(pred), TARGET_IDX:TARGET_IDX + 1]

    rmse = math.sqrt(mean_squared_error(actual, pred))
    dates = agg.index[test_start:test_start + len(pred)]
    plot_path = _plot_predictions(dates, actual, pred)

    return plot_path, rmse


# ------------------------------------------------------------------
# 경로 반환 헬퍼
# ------------------------------------------------------------------

def get_stock_png() -> str:
    return PREDICTION_PLOT_PATH

def get_model_shapes_png() -> str:
    return MODEL_SHAPES_PLOT_PATH