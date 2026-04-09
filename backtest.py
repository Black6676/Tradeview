import pandas as pd
import numpy as np
from datetime import datetime, timezone
from algorithm import (
    compute_atr, get_htf_bias, detect_entry_signals,
    apply_trade_management, record_trade_result,
    is_trading_session, ACCOUNT_BALANCE, CONFIDENCE_THRESHOLD
)


def run_backtest(candles, symbol="EURUSD"):
    df = pd.DataFrame(candles)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = df["time"].astype(int)
    df = df.sort_values("time").reset_index(drop=True)

    atr          = compute_atr(df)
    htf_bias_map = get_htf_bias(df)

    print(f"\n=== BACKTEST: {symbol} | Candles: {len(df)} | Threshold: {CONFIDENCE_THRESHOLD} ===")

    trades        = []
    in_trade      = False
    current_trade = None

    for i in range(200, len(df)):
        price  = float(df["close"].iloc[i])
        high_i = float(df["high"].iloc[i])
        low_i  = float(df["low"].iloc[i])
        ts     = int(df["time"].iloc[i])

        # ── Manage open trade ──────────────────────────────────
        if in_trade and current_trade:
            current_trade = apply_trade_management(current_trade, price)

            if current_trade["type"] == "buy":
                if low_i <= current_trade["sl"]:
                    current_trade.update({"exit_price": current_trade["sl"],
                                          "exit_time": ts, "result": "loss", "pnl_r": -1.0})
                    trades.append(current_trade)
                    record_trade_result(current_trade, "loss")
                    in_trade = False; current_trade = None; continue
                elif high_i >= current_trade["tp"]:
                    current_trade.update({"exit_price": current_trade["tp"],
                                          "exit_time": ts, "result": "win", "pnl_r": 2.0})
                    trades.append(current_trade)
                    record_trade_result(current_trade, "win")
                    in_trade = False; current_trade = None; continue
            else:
                if high_i >= current_trade["sl"]:
                    current_trade.update({"exit_price": current_trade["sl"],
                                          "exit_time": ts, "result": "loss", "pnl_r": -1.0})
                    trades.append(current_trade)
                    record_trade_result(current_trade, "loss")
                    in_trade = False; current_trade = None; continue
                elif low_i <= current_trade["tp"]:
                    current_trade.update({"exit_price": current_trade["tp"],
                                          "exit_time": ts, "result": "win", "pnl_r": 2.0})
                    trades.append(current_trade)
                    record_trade_result(current_trade, "win")
                    in_trade = False; current_trade = None; continue

        if in_trade:
            continue

        # ── Look for new signal ────────────────────────────────
        window_atr = atr.iloc[:i + 1]
        signals    = detect_entry_signals(df.iloc[:i + 1], window_atr, htf_bias_map,
                                          for_display=False)
        if not signals:
            continue

        sig = signals[-1]
        # Only enter on the most recent signal candle
        if sig["time"] != ts:
            continue

        current_trade = {
            "type":        sig["type"],
            "entry_price": sig["price"],
            "entry_time":  ts,
            "sl":          sig["sl"],
            "tp":          sig["tp"],
            "rr":          sig.get("rr", 2.0),
            "rsi":         sig.get("rsi", 50),
            "lot":         sig.get("lot", 0.01),
            "atr":         sig.get("atr", 0),
            "confidence":  sig.get("confidence", 0),
            "htf":         sig.get("htf", "neutral"),
        }
        in_trade = True
        dt_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"  → {sig['type'].upper()} @ {sig['price']:.5f} | "
              f"SL {sig['sl']:.5f} | TP {sig['tp']:.5f} | "
              f"Conf {sig.get('confidence',0)}% | {dt_str}")

    # Close any still-open trade
    if in_trade and current_trade:
        current_trade.update({"exit_price": float(df["close"].iloc[-1]),
                               "exit_time":  int(df["time"].iloc[-1]),
                               "result":     "open", "pnl_r": 0.0})
        trades.append(current_trade)

    # ── Statistics ─────────────────────────────────────────────
    closed       = [t for t in trades if t["result"] != "open"]
    wins         = [t for t in closed if t["result"] == "win"]
    losses       = [t for t in closed if t["result"] == "loss"]
    total        = len(closed)
    win_count    = len(wins)
    loss_count   = len(losses)
    win_rate     = round(win_count / total * 100, 1) if total > 0 else 0
    total_r      = round(sum(t["pnl_r"] for t in closed), 2)
    avg_win_r    = round(sum(t["pnl_r"] for t in wins)   / win_count  if win_count  else 0, 2)
    avg_loss_r   = round(sum(t["pnl_r"] for t in losses) / loss_count if loss_count else 0, 2)
    gross_profit = sum(t["pnl_r"] for t in wins)
    gross_loss   = abs(sum(t["pnl_r"] for t in losses))
    profit_factor = round(gross_profit / gross_loss if gross_loss > 0 else gross_profit, 2)

    # Equity curve
    equity, cumulative = [], 0.0
    for t in closed:
        cumulative += t["pnl_r"]
        equity.append({"time": t["exit_time"], "value": round(cumulative, 2)})

    # Max drawdown
    peak, max_dd = 0.0, 0.0
    for e in equity:
        if e["value"] > peak: peak = e["value"]
        dd = peak - e["value"]
        if dd > max_dd: max_dd = dd

    stats = {
        "total_trades":  total,
        "wins":          win_count,
        "losses":        loss_count,
        "win_rate":      win_rate,
        "total_r":       total_r,
        "avg_win_r":     avg_win_r,
        "avg_loss_r":    avg_loss_r,
        "profit_factor": profit_factor,
        "max_drawdown":  round(max_dd, 2),
        "open_trades":   len([t for t in trades if t["result"] == "open"]),
    }

    print(f"\n=== RESULTS ===")
    print(f"  Trades: {total} | Win Rate: {win_rate}% | "
          f"Total R: {total_r} | PF: {profit_factor} | Max DD: {max_dd}R")

    trade_log = [{
        "type":        t["type"],
        "entry_time":  t["entry_time"],
        "entry_price": t["entry_price"],
        "exit_time":   t.get("exit_time"),
        "exit_price":  t.get("exit_price"),
        "sl":          t["sl"],
        "tp":          t["tp"],
        "result":      t["result"],
        "pnl_r":       t["pnl_r"],
        "rsi":         t.get("rsi", 0),
        "lot":         t.get("lot", 0.01),
        "confidence":  t.get("confidence", 0),
    } for t in trades]

    return {"stats": stats, "equity": equity, "trades": trade_log}
