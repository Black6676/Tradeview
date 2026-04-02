"""
Walk-Forward Backtest
Tests the algorithm on rolling windows of historical data.
Each window is unseen at the time of signal generation.
"""
import pandas as pd
import numpy as np
from algorithm import run_analysis, record_trade_result, compute_atr


def simulate_trade_outcome(sig, future_candles):
    """
    Check if SL or TP was hit on candles AFTER the signal.
    Returns ('win', pnl) or ('loss', pnl) or ('open', 0).
    PnL is in R-multiples (win=+2R, loss=-1R).
    """
    for candle in future_candles:
        high  = float(candle["high"])
        low   = float(candle["low"])

        if sig["type"] == "buy":
            if low  <= sig["sl"]: return "loss", -1.0
            if high >= sig["tp"]: return "win",  +2.0
        else:
            if high >= sig["sl"]: return "loss", -1.0
            if low  <= sig["tp"]: return "win",  +2.0

    return "open", 0.0


def run_walkforward(candles, symbol="EURUSD", timeframe="1h",
                    window=500, step=100, account_balance=1000):
    """
    Walk-forward backtest.
    - window : number of candles used for each analysis pass
    - step   : how many candles to advance before the next pass
    - Uses actual forward candles to determine SL/TP outcome (no random simulation)
    """
    df_full = pd.DataFrame(candles)
    for col in ["open", "high", "low", "close"]:
        df_full[col] = df_full[col].astype(float)
    df_full["time"] = df_full["time"].astype(int)
    df_full = df_full.sort_values("time").reset_index(drop=True)

    equity        = account_balance
    equity_curve  = [{"window": 0, "equity": equity}]
    all_trades    = []
    seen_signals  = set()   # prevent duplicate signals across windows

    total_windows = (len(df_full) - window) // step

    for w_idx, i in enumerate(range(window, len(df_full), step)):
        window_slice   = df_full.iloc[i - window:i]
        window_candles = window_slice.to_dict("records")

        try:
            result = run_analysis(window_candles, symbol=symbol, timeframe=timeframe)
        except Exception as e:
            print(f"[WF] Window {w_idx+1}/{total_windows} error: {e}")
            continue

        signals = result.get("signals", [])
        if not signals:
            equity_curve.append({"window": w_idx + 1, "equity": equity})
            continue

        sig = signals[-1]   # most recent signal in this window

        # Skip if we've already processed this signal
        sig_key = (sig["time"], sig["type"])
        if sig_key in seen_signals:
            equity_curve.append({"window": w_idx + 1, "equity": equity})
            continue
        seen_signals.add(sig_key)

        # Simulate using actual forward candles (up to 50 candles ahead)
        forward_slice   = df_full.iloc[i:min(i + 50, len(df_full))]
        forward_candles = forward_slice.to_dict("records")

        outcome, pnl_r = simulate_trade_outcome(sig, forward_candles)

        # Convert R to dollar PnL using lot size
        sl_distance = abs(sig["price"] - sig["sl"])
        risk_dollars = account_balance * 0.01  # 1% risk
        pnl_dollars  = pnl_r * risk_dollars if outcome != "open" else 0

        sig["result"]      = outcome
        sig["pnl_r"]       = pnl_r
        sig["pnl_dollars"] = round(pnl_dollars, 2)
        sig["window"]      = w_idx + 1

        all_trades.append(sig)
        equity += pnl_dollars

        # Feed into adaptive system
        if outcome != "open":
            record_trade_result(sig, outcome)

        equity_curve.append({"window": w_idx + 1, "equity": round(equity, 2)})

    # ── Metrics ────────────────────────────────────────────────
    closed_trades = [t for t in all_trades if t["result"] != "open"]
    wins          = [t for t in closed_trades if t["result"] == "win"]
    losses        = [t for t in closed_trades if t["result"] == "loss"]
    total         = len(closed_trades)
    win_count     = len(wins)
    win_rate      = round(win_count / total * 100, 1) if total > 0 else 0
    total_r       = round(sum(t["pnl_r"] for t in closed_trades), 2)
    gross_profit  = sum(t["pnl_r"] for t in wins)
    gross_loss    = abs(sum(t["pnl_r"] for t in losses))
    profit_factor = round(gross_profit / gross_loss if gross_loss > 0 else gross_profit, 2)

    equity_series = pd.Series([e["equity"] for e in equity_curve])
    returns       = equity_series.pct_change().dropna()
    sharpe        = round(returns.mean() / returns.std() * np.sqrt(252), 2) if len(returns) > 1 and returns.std() > 0 else 0
    max_dd        = round(float((equity_series.cummax() - equity_series).max()), 2)
    net_pnl       = round(equity - account_balance, 2)

    stats = {
        "symbol":        symbol,
        "timeframe":     timeframe,
        "total_trades":  total,
        "wins":          win_count,
        "losses":        len(losses),
        "win_rate":      win_rate,
        "total_r":       total_r,
        "profit_factor": profit_factor,
        "sharpe":        sharpe,
        "max_drawdown":  max_dd,
        "net_pnl":       net_pnl,
        "final_equity":  round(equity, 2),
        "windows_tested": total_windows,
    }

    print(f"\n[Walk-Forward Results] {symbol} {timeframe}")
    print(f"  Trades: {total} | Win Rate: {win_rate}% | Total R: {total_r}")
    print(f"  Profit Factor: {profit_factor} | Sharpe: {sharpe} | Max DD: ${max_dd}")
    print(f"  Net P&L: ${net_pnl} | Final Equity: ${equity:.2f}")

    return {
        "stats":        stats,
        "equity_curve": equity_curve,
        "trades":       all_trades,
    }
