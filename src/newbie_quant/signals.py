from __future__ import annotations

import numpy as np
import pandas as pd


def add_signals(frame: pd.DataFrame, buy_threshold: float = 0.60, sell_threshold: float = 0.40) -> pd.DataFrame:
    if sell_threshold >= buy_threshold:
        raise ValueError("sell_threshold must be lower than buy_threshold")
    if "prob_up" not in frame.columns:
        raise ValueError("frame must include a prob_up column")

    signaled = frame.copy()
    signaled["signal"] = np.select(
        [signaled["prob_up"] > buy_threshold, signaled["prob_up"] < sell_threshold],
        ["BUY", "SELL"],
        default="HOLD",
    )
    signaled["position"] = np.select(
        [signaled["signal"] == "BUY", signaled["signal"] == "SELL"],
        [1, -1],
        default=0,
    )
    return signaled
