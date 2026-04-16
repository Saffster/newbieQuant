from __future__ import annotations

from newbie_quant.backtest import run_backtest
from newbie_quant.data import generate_sample_ohlcv
from newbie_quant.features import FEATURE_COLUMNS, add_features
from newbie_quant.pipeline import run_pipeline
from newbie_quant.signals import add_signals


def test_feature_engineering_adds_target_and_features() -> None:
    raw = generate_sample_ohlcv(rows=180)
    featured = add_features(raw)

    assert not featured.empty
    assert set(FEATURE_COLUMNS).issubset(featured.columns)
    assert set(featured["target"].unique()).issubset({0, 1})


def test_signal_thresholds_map_to_positions() -> None:
    raw = generate_sample_ohlcv(rows=80)
    featured = add_features(raw)
    featured["prob_up"] = [0.7 if index % 3 == 0 else 0.3 if index % 3 == 1 else 0.5 for index in range(len(featured))]

    signaled = add_signals(featured)

    assert {"BUY", "SELL", "HOLD"}.issubset(set(signaled["signal"]))
    assert set(signaled["position"].unique()).issubset({-1, 0, 1})


def test_backtest_returns_core_metrics() -> None:
    raw = generate_sample_ohlcv(rows=100)
    featured = add_features(raw)
    featured["position"] = 1

    _, metrics = run_backtest(featured)

    assert {"cumulative_return", "sharpe", "max_drawdown", "trades"}.issubset(metrics)


def test_pipeline_runs_end_to_end() -> None:
    output = run_pipeline(model_type="logistic")

    assert not output.data.empty
    assert "prob_up" in output.data.columns
    assert "equity_curve" in output.data.columns
    assert "accuracy" in output.model_metrics
    assert "cumulative_return" in output.backtest_metrics
