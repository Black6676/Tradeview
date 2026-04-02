import pandas as pd
import numpy as np
from algorithm import run_analysis, record_trade_result  # import your updated file

def simple_backtest(candles, symbol="XAUUSD", window=500, step=100):
    df_full = pd.DataFrame(candles)
    equity = [1000]
    trades = []
    equity_curve = []

    for i in range(window, len(df_full), step):
        window_candles = df_full.iloc[i-window:i].to_dict('records')
        result = run_analysis(window_candles, symbol=symbol, timeframe="1h")
        
        # Simulate fill on next candle (open) + 0.5 pip slippage approx
        if result["signals"]:
            sig = result["signals"][-1]  # latest
            entry = sig["price"] + (0.0005 if sig["type"]=="buy" else -0.0005)  # rough spread
            # In real: track open position, check SL/TP on subsequent candles
            # For simplicity here: assume R:R outcome based on random or historical
            result_sim = "win" if np.random.rand() < 0.55 else "loss"  # placeholder - replace with actual simulation
            record_trade_result(sig, result_sim)
            pnl = sig["lot"] * (sig["tp"] - entry) * 100 if result_sim == "win" else sig["lot"] * (entry - sig["sl"]) * -100
            equity.append(equity[-1] + pnl)
            trades.append(sig)
        else:
            equity.append(equity[-1])

        equity_curve.append(equity[-1])

    # Metrics
    returns = pd.Series(equity).pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0
    max_dd = (pd.Series(equity).cummax() - pd.Series(equity)).max()
    win_rate = len([t for t in trades if t.get("result")=="win"]) / len(trades) if trades else 0

    print(f"Backtest Results for {symbol}:")
    print(f"Final Equity: ${equity[-1]:.2f} | Sharpe: {sharpe:.2f} | Max DD: ${max_dd:.2f} | Win Rate: {win_rate:.1%} | Trades: {len(trades)}")
    return {"equity": equity, "trades": trades, "sharpe": sharpe}

# Usage example:
# candles = pd.read_csv("your_data.csv").to_dict('records')
# simple_backtest(candles, "XAUUSD")