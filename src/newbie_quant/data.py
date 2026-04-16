from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class MarketDataRequest:
    ticker: str = "AAPL"
    period: str = "3y"
    seed: int = 42


def load_market_data(csv_path: str | Path | None = None, request: MarketDataRequest | None = None) -> pd.DataFrame:
    """Load OHLCV data from CSV, yfinance if installed, or a deterministic sample series."""
    if csv_path:
        return _normalize_ohlcv(pd.read_csv(csv_path))

    request = request or MarketDataRequest()
    try:
        import yfinance as yf  # type: ignore

        raw = yf.download(request.ticker, period=request.period, auto_adjust=True, progress=False)
        if not raw.empty:
            return _normalize_ohlcv(raw.reset_index())
    except Exception:
        pass

    return generate_sample_ohlcv(seed=request.seed)


def generate_sample_ohlcv(rows: int = 760, seed: int = 42) -> pd.DataFrame:
    """Create deterministic OHLCV data that behaves enough like prices for local demos/tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows)
    market_noise = rng.normal(0.0004, 0.014, rows)
    seasonal = 0.003 * np.sin(np.linspace(0, 22, rows))
    close = 100 * np.exp(np.cumsum(market_noise + seasonal / 252))
    open_ = close * (1 + rng.normal(0, 0.004, rows))
    spread = np.abs(rng.normal(0.008, 0.004, rows))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(1_000_000, 6_000_000, rows)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower().replace(" ", "_") for column in normalized.columns]

    if "adj_close" in normalized.columns and "close" not in normalized.columns:
        normalized = normalized.rename(columns={"adj_close": "close"})
    if "date" not in normalized.columns:
        normalized = normalized.rename(columns={normalized.columns[0]: "date"})

    missing = [column for column in OHLCV_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(f"Market data is missing required columns: {', '.join(missing)}")

    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized = normalized[["date", *OHLCV_COLUMNS]].dropna().sort_values("date")
    for column in OHLCV_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized.dropna().reset_index(drop=True)
