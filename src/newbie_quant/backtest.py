from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(frame: pd.DataFrame, transaction_cost: float = 0.001) -> tuple[pd.DataFrame, dict[str, float]]:
    required = {"close", "position"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Backtest frame is missing required columns: {', '.join(sorted(missing))}")

    results = frame.copy()
    results["market_return"] = results["close"].pct_change().fillna(0)
    results["strategy_position"] = results["position"].shift(1).fillna(0)
    trades = results["strategy_position"].diff().abs().fillna(0)
    results["strategy_return"] = results["strategy_position"] * results["market_return"] - trades * transaction_cost
    results["equity_curve"] = (1 + results["strategy_return"]).cumprod()
    results["buy_hold_curve"] = (1 + results["market_return"]).cumprod()

    metrics = {
        "cumulative_return": float(results["equity_curve"].iloc[-1] - 1),
        "buy_hold_return": float(results["buy_hold_curve"].iloc[-1] - 1),
        "sharpe": _sharpe(results["strategy_return"]),
        "max_drawdown": _max_drawdown(results["equity_curve"]),
        "win_rate": float((results["strategy_return"] > 0).mean()),
        "trades": float((trades > 0).sum()),
    }
    return results, metrics


def _sharpe(returns: pd.Series) -> float:
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(252) * returns.mean() / std)


def _max_drawdown(equity_curve: pd.Series) -> float:
    drawdown = equity_curve / equity_curve.cummax() - 1
    return float(drawdown.min())
