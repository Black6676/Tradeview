import pandas as pd
import numpy as np
from algorithm import (compute_ema, compute_rsi, compute_atr,
                       detect_order_blocks, get_htf_bias,
                       detect_swings, detect_structure_breaks, detect_liquidity,
                       is_trading_session, lot_size)


def run_backtest(candles, symbol="EURUSD"):
    df = pd.DataFrame(candles)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = df["time"].astype(int)
    df = df.sort_values("time").reset_index(drop=True)

    closes = df["close"]
    highs  = df["high"]
    lows   = df["low"]
    times  = df["time"].values

    ema200       = compute_ema(closes, 200)
    rsi          = compute_rsi(closes, 14)
    atr          = compute_atr(df)
    htf_bias_map = get_htf_bias(df)

    bull_obs, bear_obs  = detect_order_blocks(df)
    all_obs             = bull_obs + bear_obs

    swing_highs, swing_lows = detect_swings(df)
    structure               = detect_structure_breaks(swing_highs, swing_lows)
    liquidity               = detect_liquidity(df)

    trades, in_trade, current_trade = [], False, None

    for i in range(200, len(df)):
        price   = float(closes.iloc[i])
        high_i  = float(highs.iloc[i])
        low_i   = float(lows.iloc[i])
        ts      = int(times[i])
        e200    = float(ema200.iloc[i])
        rsi_val = float(rsi.iloc[i])
        atr_val = float(atr.iloc[i])

        candle_date  = pd.Timestamp(ts, unit="s").date()
        htf          = htf_bias_map.get(candle_date, "neutral")
        has_bull_bos = any(b["type"] == "bullish_bos" and i - 50 < b["idx"] < i for b in structure)
        has_bear_bos = any(b["type"] == "bearish_bos" and i - 50 < b["idx"] < i for b in structure)
        has_liq      = any(i - 20 < l["idx"] < i for l in liquidity)

        # ── Manage open trade ──────────────────────────────────
        if in_trade and current_trade:
            t = current_trade
            if t["type"] == "buy":
                if low_i <= t["sl"]:
                    t.update({"exit_price": t["sl"], "exit_time": ts,
                               "result": "loss", "pnl_r": -1.0})
                    trades.append(t); in_trade, current_trade = False, None; continue
                elif high_i >= t["tp"]:
                    t.update({"exit_price": t["tp"], "exit_time": ts,
                               "result": "win", "pnl_r": 2.0})
                    trades.append(t); in_trade, current_trade = False, None; continue
            elif t["type"] == "sell":
                if high_i >= t["sl"]:
                    t.update({"exit_price": t["sl"], "exit_time": ts,
                               "result": "loss", "pnl_r": -1.0})
                    trades.append(t); in_trade, current_trade = False, None; continue
                elif low_i <= t["tp"]:
                    t.update({"exit_price": t["tp"], "exit_time": ts,
                               "result": "win", "pnl_r": 2.0})
                    trades.append(t); in_trade, current_trade = False, None; continue

        if in_trade:
            continue

        # ── Look for new entry ─────────────────────────────────
        for ob in all_obs:
            ob_rows = df[df["time"] == ob["time"]]
            if ob_rows.empty or ob_rows.index[0] >= i:
                continue

            in_zone = ob["bottom"] <= price <= ob["top"]

            if ob["type"] == "bullish" and htf == "bullish" and in_zone and price > e200 and is_trading_session(ts):
                score = 20
                if has_bull_bos: score += 25
                if has_liq:      score += 20
                if 45 <= rsi_val <= 70: score += 20
                if score >= 35:
                    sl = round(ob["bottom"] - atr_val * 1.5, 5)
                    tp = round(price + (price - sl) * 2.0, 5)
                    current_trade = {
                        "type": "buy", "entry_price": price, "entry_time": ts,
                        "sl": sl, "tp": tp, "rr": 2.0,
                        "rsi": round(rsi_val, 1), "htf": htf,
                        "lot": lot_size(round(price - sl, 5)),
                    }
                    in_trade = True; break

            elif ob["type"] == "bearish" and htf == "bearish" and in_zone and price < e200 and is_trading_session(ts):
                score = 20
                if has_bear_bos: score += 25
                if has_liq:      score += 20
                if 30 <= rsi_val <= 55: score += 20
                if score >= 35:
                    sl = round(ob["top"] + atr_val * 1.5, 5)
                    tp = round(price - (sl - price) * 2.0, 5)
                    current_trade = {
                        "type": "sell", "entry_price": price, "entry_time": ts,
                        "sl": sl, "tp": tp, "rr": 2.0,
                        "rsi": round(rsi_val, 1), "htf": htf,
                        "lot": lot_size(round(sl - price, 5)),
                    }
                    in_trade = True; break

    # Close any still-open trade at last price
    if in_trade and current_trade:
        current_trade.update({
            "exit_price": float(closes.iloc[-1]),
            "exit_time":  int(times[-1]),
            "result":     "open",
            "pnl_r":      0.0,
        })
        trades.append(current_trade)

    # ── Statistics ─────────────────────────────────────────────
    closed        = [t for t in trades if t["result"] != "open"]
    wins          = [t for t in closed if t["result"] == "win"]
    losses        = [t for t in closed if t["result"] == "loss"]
    total         = len(closed)
    win_count     = len(wins)
    loss_count    = len(losses)
    win_rate      = round(win_count / total * 100, 1) if total > 0 else 0
    total_r       = round(sum(t["pnl_r"] for t in closed), 2)
    avg_win_r     = round(sum(t["pnl_r"] for t in wins)   / win_count  if win_count  else 0, 2)
    avg_loss_r    = round(sum(t["pnl_r"] for t in losses) / loss_count if loss_count else 0, 2)
    gross_profit  = sum(t["pnl_r"] for t in wins)
    gross_loss    = abs(sum(t["pnl_r"] for t in losses))
    profit_factor = round(gross_profit / gross_loss if gross_loss > 0 else gross_profit, 2)

    equity, cumulative = [], 0.0
    for t in closed:
        cumulative += t["pnl_r"]
        equity.append({"time": t["exit_time"], "value": round(cumulative, 2)})

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
        "rsi":         t["rsi"],
        "lot":         t.get("lot", 0.01),
    } for t in trades]

    return {"stats": stats, "equity": equity, "trades": trade_log}