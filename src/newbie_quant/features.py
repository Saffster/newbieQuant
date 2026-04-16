from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "sma_ratio_10",
    "sma_ratio_20",
    "ema_ratio_12",
    "rsi_14",
    "macd",
    "macd_signal",
    "volatility_10",
    "volume_change_5d",
    "momentum_10",
]


def add_features(frame: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    data = frame.copy()
    close = data["close"]

    data["return_1d"] = close.pct_change()
    data["return_5d"] = close.pct_change(5)
    data["return_10d"] = close.pct_change(10)
    data["sma_ratio_10"] = close / close.rolling(10).mean() - 1
    data["sma_ratio_20"] = close / close.rolling(20).mean() - 1
    data["ema_ratio_12"] = close / close.ewm(span=12, adjust=False).mean() - 1
    data["rsi_14"] = _rsi(close, 14)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema_12 - ema_26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["volatility_10"] = data["return_1d"].rolling(10).std()
    data["volume_change_5d"] = data["volume"].pct_change(5)
    data["momentum_10"] = close / close.shift(10) - 1

    data["future_return"] = close.shift(-horizon) / close - 1
    data["target"] = (data["future_return"] > 0).astype(int)
    return data.dropna().reset_index(drop=True)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))
