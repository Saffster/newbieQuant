from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from newbie_quant.backtest import run_backtest
from newbie_quant.data import MarketDataRequest, load_market_data
from newbie_quant.features import add_features
from newbie_quant.model import append_predictions, train_model
from newbie_quant.signals import add_signals


@dataclass(frozen=True)
class PipelineOutput:
    data: pd.DataFrame
    model_metrics: dict[str, float]
    backtest_metrics: dict[str, float]


def run_pipeline(
    csv_path: str | Path | None = None,
    ticker: str = "AAPL",
    model_type: str = "logistic",
    horizon: int = 5,
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.40,
    transaction_cost: float = 0.001,
) -> PipelineOutput:
    raw = load_market_data(csv_path, MarketDataRequest(ticker=ticker))
    featured = add_features(raw, horizon=horizon)
    model_result = train_model(featured, model_type=model_type)
    predicted = append_predictions(featured, model_result.model)
    signaled = add_signals(predicted, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    backtest_data, backtest_metrics = run_backtest(signaled, transaction_cost=transaction_cost)
    return PipelineOutput(
        data=backtest_data,
        model_metrics=model_result.metrics,
        backtest_metrics=backtest_metrics,
    )
