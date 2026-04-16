from __future__ import annotations

import argparse
from pathlib import Path

from newbie_quant.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a quant signal model and run a backtest.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV with date, open, high, low, close, volume.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker to download when yfinance is installed.")
    parser.add_argument("--model", choices=["logistic", "random_forest"], default="logistic")
    parser.add_argument("--horizon", type=int, default=5, help="Future trading-day horizon for the target.")
    parser.add_argument("--buy-threshold", type=float, default=0.60)
    parser.add_argument("--sell-threshold", type=float, default=0.40)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=Path("data/backtest_results.csv"))
    args = parser.parse_args()

    output = run_pipeline(
        csv_path=args.csv,
        ticker=args.ticker,
        model_type=args.model,
        horizon=args.horizon,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        transaction_cost=args.transaction_cost,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.data.to_csv(args.output, index=False)
    _print_metrics("Model", output.model_metrics)
    _print_metrics("Backtest", output.backtest_metrics)
    print(f"Saved rows: {len(output.data)}")
    print(f"Output: {args.output}")


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title} metrics")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
