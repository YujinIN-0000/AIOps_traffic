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
from keras.models import load_model

from config import (
    MODEL_SAVE_PATH, PREDICTION_PLOT_PATH, MODEL_SHAPES_PLOT_PATH,
    LOOKBACK, TRAIN_RATIO, RMSE_THRESHOLD
)

def _get_splits(n: int):
    train = int(n * TRAIN_RATIO)
    test  = n - train
    return train, test

FEATURE_COLS = ["internet", "smsin", "callin", "hour", "dayofweek", "is_weekend"]
TARGET_IDX   = 0


# ------------------------------------------------------------------
# 내부 유틸 (model.py 와 동일 로직, 독립적으로 유지)
# ------------------------------------------------------------------

def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    numeric = ["internet", "smsin", "callin"]
    agg = df[numeric].groupby(df.index).sum(min_count=1).fillna(0)

    agg["hour"]       = agg.index.hour
    agg["dayofweek"]  = agg.index.dayofweek
    agg["is_weekend"] = (agg.index.dayofweek >= 5).astype(int)

    return agg[FEATURE_COLS]


def _make_sequences(scaled: np.ndarray, start: int, count: int):
    X = []
    for i in range(start, start + count):
        if i < LOOKBACK:
            continue
        X.append(scaled[i - LOOKBACK:i])
    return np.array(X)


def _plot_predictions(dates, actual: np.ndarray, predicted: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(actual,    color="red",  label="Actual Traffic",    linewidth=0.8, alpha=0.9)
    ax.plot(predicted, color="blue", label="Predicted Traffic", linewidth=0.8, alpha=0.8)

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
# 공개 API
# ------------------------------------------------------------------

def process(df: pd.DataFrame) -> tuple:
    """저장된 모델로 추론. (plot_path, rmse, actual_list, pred_list, date_list) 반환"""
    agg  = _preprocess(df)
    data = agg.values

    n = len(data)
    train_steps, test_steps = _get_splits(n)

    train_data = data[:train_steps]
    sc = MinMaxScaler(feature_range=(0, 1))
    sc.fit(train_data)
    full_scaled = sc.transform(data)

    # 전체 구간 예측 (LOOKBACK 이후)
    full_start = LOOKBACK
    full_count = n - LOOKBACK
    X_full = _make_sequences(full_scaled, full_start, full_count)

    mdl = load_model(MODEL_SAVE_PATH)
    pred_scaled = mdl.predict(X_full, verbose=0)

    sc_target = MinMaxScaler(feature_range=(0, 1))
    sc_target.fit(train_data[:, TARGET_IDX:TARGET_IDX + 1])
    pred_full = sc_target.inverse_transform(pred_scaled).flatten()
    actual_full = data[:, TARGET_IDX].flatten()

    # RMSE는 test 구간 기준
    test_start = train_steps
    test_pred = pred_full[(test_start - LOOKBACK):] if test_start >= LOOKBACK else pred_full
    test_actual = actual_full[test_start:]
    rmse = math.sqrt(mean_squared_error(test_actual[:len(test_pred)], test_pred))

    dates = agg.index
    plot_path = _plot_predictions(dates[test_start:test_start + len(test_pred)],
                                  test_actual[:len(test_pred)],
                                  test_pred)

    # 전체 기간용 시리즈 (LOOKBACK 이전은 None)
    pred_list = [None] * LOOKBACK + [round(float(v), 2) for v in pred_full]
    actual_list = [round(float(v), 2) for v in actual_full]
    date_list = [d.isoformat() for d in dates]

    return plot_path, rmse, actual_list, pred_list, date_list


def needs_retrain(rmse: float) -> bool:
    """RMSE가 임계값을 초과하면 재학습 필요"""
    return rmse > RMSE_THRESHOLD


def return_rmse(test, predicted) -> float:
    """float RMSE 반환"""
    return math.sqrt(mean_squared_error(test, predicted))


def get_stock_png() -> str:
    return PREDICTION_PLOT_PATH

def get_model_shapes_png() -> str:
    return MODEL_SHAPES_PLOT_PATH