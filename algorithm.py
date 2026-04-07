import pandas as pd
import numpy as np

try:
    from scipy.signal import argrelextrema
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════

def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def compute_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ══════════════════════════════════════════════════════════════
# RISK MANAGEMENT & SESSION FILTER
# ══════════════════════════════════════════════════════════════

ACCOUNT_BALANCE      = 1000
RISK_PER_TRADE       = 0.01
MAX_TRADES_PER_DAY   = 3
CONFIDENCE_THRESHOLD = 65

_trade_history = []


def is_trading_session(ts):
    """
    Active market hours: 07:00–20:00 UTC.
    Covers London open through NY close.
    Excludes Asian session low-liquidity dead zone (20:00–07:00).
    """
    from datetime import datetime, timezone
    hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour
    return 7 <= hour <= 20


def lot_size(sl_distance, account_balance=None, risk_pct=None):
    balance  = account_balance or ACCOUNT_BALANCE
    risk_pct = risk_pct or RISK_PER_TRADE
    risk     = balance * risk_pct
    if sl_distance <= 0:
        return 0.01
    lot = risk / (sl_distance * 100)
    return round(max(lot, 0.01), 2)


def apply_trade_management(trade, current_price):
    entry = trade["entry_price"]
    sl    = trade["sl"]
    tp    = trade["tp"]
    risk  = abs(entry - sl)
    atr   = trade.get("atr", 0)
    if trade["type"] == "buy":
        if current_price - entry >= risk:
            trade["sl"] = max(sl, entry)
        trade["sl"] = max(trade["sl"], current_price - 1.5 * atr)
    elif trade["type"] == "sell":
        if entry - current_price >= risk:
            trade["sl"] = min(sl, entry)
        trade["sl"] = min(trade["sl"], current_price + 1.5 * atr)
    return trade


def adapt_strategy():
    global CONFIDENCE_THRESHOLD
    if len(_trade_history) < 10:
        return
    wins = sum(1 for t in _trade_history if t.get("result") == "win")
    rate = wins / len(_trade_history)
    if rate < 0.4:
        CONFIDENCE_THRESHOLD = min(90, CONFIDENCE_THRESHOLD + 5)
    elif rate > 0.6:
        CONFIDENCE_THRESHOLD = max(60, CONFIDENCE_THRESHOLD - 5)


def record_trade_result(signal, result):
    _trade_history.append({**signal, "result": result})
    if len(_trade_history) > 50:
        _trade_history.pop(0)
    adapt_strategy()


# ══════════════════════════════════════════════════════════════
# HTF BIAS
# ══════════════════════════════════════════════════════════════

def get_htf_bias(df):
    df2 = df.copy()
    df2["datetime"] = pd.to_datetime(df2["time"], unit="s", utc=True)
    df2 = df2.set_index("datetime")
    h4 = df2[["open","high","low","close"]].resample("4h").agg({
        "open":"first","high":"max","low":"min","close":"last"
    }).dropna()
    if len(h4) < 25:
        return {}
    h4_ema20 = compute_ema(h4["close"], 20)
    h4_ema50 = compute_ema(h4["close"], 50)
    bias_map = {}
    for dt in h4.index:
        close = h4.loc[dt, "close"]
        e20   = h4_ema20.loc[dt]
        e50   = h4_ema50.loc[dt]
        bias  = "bullish" if close > e20 > e50 else "bearish" if close < e20 < e50 else "neutral"
        for h in range(4):
            key = (dt + pd.Timedelta(hours=h)).date()
            bias_map[key] = bias
    return bias_map


# ══════════════════════════════════════════════════════════════
# MARKET STRUCTURE
# ══════════════════════════════════════════════════════════════

def detect_swings(df, lookback=5):
    highs = df["high"].values
    lows  = df["low"].values
    if SCIPY_AVAILABLE:
        try:
            order = max(2, lookback)
            sh_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
            sl_idx = argrelextrema(lows,  np.less_equal,    order=order)[0]
            swing_highs = [(i, highs[i]) for i in sh_idx if lookback <= i < len(df) - lookback]
            swing_lows  = [(i, lows[i])  for i in sl_idx if lookback <= i < len(df) - lookback]
            return swing_highs, swing_lows
        except Exception:
            pass
    swing_highs, swing_lows = [], []
    for i in range(lookback, len(df) - lookback):
        if highs[i] == max(highs[i - lookback:i + lookback + 1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i - lookback:i + lookback + 1]):
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


def detect_structure_breaks(swing_highs, swing_lows):
    breaks = []
    for i in range(1, len(swing_highs)):
        if swing_highs[i][1] > swing_highs[i - 1][1]:
            breaks.append({"type": "bullish_bos", "idx": swing_highs[i][0], "level": swing_highs[i][1]})
    for i in range(1, len(swing_lows)):
        if swing_lows[i][1] < swing_lows[i - 1][1]:
            breaks.append({"type": "bearish_bos", "idx": swing_lows[i][0], "level": swing_lows[i][1]})
    return breaks


# ══════════════════════════════════════════════════════════════
# LIQUIDITY & FVG
# ══════════════════════════════════════════════════════════════

def detect_liquidity(df, threshold=0.0005):
    highs      = df["high"].values
    lows       = df["low"].values
    avg_price  = float(df["close"].mean())
    scaled_thr = threshold * max(1.0, avg_price / 2.0)
    liquidity  = []
    for i in range(1, len(df)):
        if abs(highs[i] - highs[i - 1]) < scaled_thr:
            liquidity.append({"type": "equal_highs", "idx": i, "level": highs[i]})
        if abs(lows[i]  - lows[i - 1])  < scaled_thr:
            liquidity.append({"type": "equal_lows",  "idx": i, "level": lows[i]})
    return liquidity


def detect_fvg(df):
    """Fair Value Gaps — stores candle index for accurate recency checks."""
    fvgs = []
    for i in range(2, len(df) - 1):
        if df["low"].iloc[i] > df["high"].iloc[i - 2]:
            fvgs.append({"idx": i, "time": int(df["time"].iloc[i - 1]),
                         "top": float(df["low"].iloc[i]), "bottom": float(df["high"].iloc[i - 2]),
                         "type": "bullish"})
        elif df["high"].iloc[i] < df["low"].iloc[i - 2]:
            fvgs.append({"idx": i, "time": int(df["time"].iloc[i - 1]),
                         "top": float(df["low"].iloc[i - 2]), "bottom": float(df["high"].iloc[i]),
                         "type": "bearish"})
    return fvgs


# ══════════════════════════════════════════════════════════════
# ORDER BLOCKS  (volume + mitigation fix)
# ══════════════════════════════════════════════════════════════

def detect_order_blocks(df, lookback=10):
    atr    = compute_atr(df)
    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    times  = df["time"].values
    volumes = df["volume"].values if "volume" in df.columns and df["volume"].sum() > 0 else np.ones(len(df))
    bull_obs, bear_obs = [], []

    for i in range(lookback, len(df) - 1):
        body       = abs(closes[i] - opens[i])
        threshold  = 1.2 * atr.iloc[i]
        vol_avg    = np.mean(volumes[max(0, i - lookback):i + 1])
        vol_strong = volumes[i] > vol_avg * 1.2 if vol_avg > 0 else True

        if closes[i] > opens[i] and body > threshold and vol_strong:
            for j in range(i - 1, max(i - lookback, 0) - 1, -1):
                if closes[j] < opens[j]:
                    ob_top = float(max(opens[j], closes[j]))
                    ob_bot = float(min(opens[j], closes[j]))
                    mitigated = any(ob_bot <= p <= ob_top for p in closes[j + 1:i])
                    if mitigated or (i - j <= 5):
                        bull_obs.append({"time": int(times[j]), "top": round(ob_top, 5),
                            "bottom": round(ob_bot, 5), "type": "bullish",
                            "high": round(float(highs[j]), 5), "low": round(float(lows[j]), 5),
                            "atr": round(float(atr.iloc[j]), 5), "signal_idx": i})
                    break

        if closes[i] < opens[i] and body > threshold and vol_strong:
            for j in range(i - 1, max(i - lookback, 0) - 1, -1):
                if closes[j] > opens[j]:
                    ob_top = float(max(opens[j], closes[j]))
                    ob_bot = float(min(opens[j], closes[j]))
                    mitigated = any(ob_bot <= p <= ob_top for p in closes[j + 1:i])
                    if mitigated or (i - j <= 5):
                        bear_obs.append({"time": int(times[j]), "top": round(ob_top, 5),
                            "bottom": round(ob_bot, 5), "type": "bearish",
                            "high": round(float(highs[j]), 5), "low": round(float(lows[j]), 5),
                            "atr": round(float(atr.iloc[j]), 5), "signal_idx": i})
                    break

    seen, unique_bull = set(), []
    for ob in bull_obs:
        if ob["time"] not in seen:
            seen.add(ob["time"]); unique_bull.append(ob)
    seen, unique_bear = set(), []
    for ob in bear_obs:
        if ob["time"] not in seen:
            seen.add(ob["time"]); unique_bear.append(ob)
    return unique_bull, unique_bear


# ══════════════════════════════════════════════════════════════
# CONFIDENCE SCORE
# ══════════════════════════════════════════════════════════════

def compute_confidence(signal, trend, rsi_val, htf_bias):
    score = 0
    if signal["type"] == "buy"  and trend == "bullish" and htf_bias == "bullish": score += 40
    elif signal["type"] == "sell" and trend == "bearish" and htf_bias == "bearish": score += 40
    elif (signal["type"] == "buy" and trend == "bullish") or \
         (signal["type"] == "sell" and trend == "bearish"): score += 20
    if signal["type"] == "buy"  and 55 < rsi_val <= 70: score += 30
    elif signal["type"] == "buy"  and 45 <= rsi_val <= 55: score += 15
    elif signal["type"] == "sell" and 30 <= rsi_val < 45: score += 30
    elif signal["type"] == "sell" and 45 <= rsi_val <= 55: score += 15
    if signal.get("rr", 0) >= 2: score += 30
    return min(score, 100)


# ══════════════════════════════════════════════════════════════
# SIGNAL ENGINE  (score-based + FVG)
# ══════════════════════════════════════════════════════════════

def detect_entry_signals(df, atr_series, htf_bias_map, for_display=True):
    closes = df["close"]
    ema200 = compute_ema(closes, 200)
    rsi    = compute_rsi(closes, 14)
    bull_obs, bear_obs = detect_order_blocks(df)
    all_obs = bull_obs + bear_obs
    swing_highs, swing_lows = detect_swings(df)
    structure = detect_structure_breaks(swing_highs, swing_lows)
    liquidity = detect_liquidity(df)
    fvgs      = detect_fvg(df)
    times     = df["time"].values
    signals   = []

    for i in range(200, len(df)):
        price   = float(closes.iloc[i])
        ts      = int(times[i])
        e200    = float(ema200.iloc[i])
        rsi_val = float(rsi.iloc[i])
        atr_val = float(atr_series.iloc[i])

        if not is_trading_session(ts):
            continue

        candle_date  = pd.Timestamp(ts, unit="s").date()
        htf          = htf_bias_map.get(candle_date, "neutral")
        has_bull_bos = any(b["type"] == "bullish_bos" and i - 50 < b["idx"] < i for b in structure)
        has_bear_bos = any(b["type"] == "bearish_bos" and i - 50 < b["idx"] < i for b in structure)
        has_liq      = any(i - 20 < l["idx"] < i for l in liquidity)
        has_bull_fvg = any(i - 30 < f["idx"] < i for f in fvgs if f["type"] == "bullish")
        has_bear_fvg = any(i - 30 < f["idx"] < i for f in fvgs if f["type"] == "bearish")

        for ob in all_obs:
            ob_rows = df[df["time"] == ob["time"]]
            if ob_rows.empty or ob_rows.index[0] >= i:
                continue
            in_zone = ob["bottom"] <= price <= ob["top"]

            if ob["type"] == "bullish" and htf == "bullish" and in_zone and price > e200:
                score = 20
                if has_bull_bos: score += 25
                if has_liq:      score += 20
                if has_bull_fvg: score += 15
                if 45 <= rsi_val <= 70: score += 15
                if score >= 35:
                    sl = round(ob["bottom"] - atr_val * 1.5, 5)
                    tp = round(price + (price - sl) * 2.0, 5)
                    sig = {"time": ts, "type": "buy", "price": round(price, 5),
                           "rsi": round(rsi_val, 1), "sl": sl, "tp": tp, "rr": 2.0,
                           "atr": round(atr_val, 5), "htf": htf,
                           "lot": lot_size(round(price - sl, 5))}
                    sig["confidence"] = compute_confidence(sig, "bullish", rsi_val, htf)
                    if sig["confidence"] >= CONFIDENCE_THRESHOLD:
                        signals.append(sig)

            elif ob["type"] == "bearish" and htf == "bearish" and in_zone and price < e200:
                score = 20
                if has_bear_bos: score += 25
                if has_liq:      score += 20
                if has_bear_fvg: score += 15
                if 30 <= rsi_val <= 55: score += 15
                if score >= 35:
                    sl = round(ob["top"] + atr_val * 1.5, 5)
                    tp = round(price - (sl - price) * 2.0, 5)
                    sig = {"time": ts, "type": "sell", "price": round(price, 5),
                           "rsi": round(rsi_val, 1), "sl": sl, "tp": tp, "rr": 2.0,
                           "atr": round(atr_val, 5), "htf": htf,
                           "lot": lot_size(round(sl - price, 5))}
                    sig["confidence"] = compute_confidence(sig, "bearish", rsi_val, htf)
                    if sig["confidence"] >= CONFIDENCE_THRESHOLD:
                        signals.append(sig)

    deduped, last_ts = [], 0
    gap = 5 * 3600 if for_display else 3 * 3600
    for s in sorted(signals, key=lambda x: x["time"]):
        if s["time"] - last_ts > gap:
            deduped.append(s); last_ts = s["time"]
    return deduped[-10:] if for_display else deduped


# ══════════════════════════════════════════════════════════════
# TRADE EXECUTOR
# ══════════════════════════════════════════════════════════════

def execute_trade(signal, symbol="XAUUSD", lot=0.01, use_mt5=False):
    print(f"[TradeView] {signal['type'].upper()} @ {signal['price']} | "
          f"SL: {signal['sl']} | TP: {signal['tp']} | "
          f"Lot: {lot} | Confidence: {signal.get('confidence','—')}%")
    if use_mt5:
        return execute_trade_mt5(signal, symbol=symbol, lot=lot)
    print("[TradeView] Simulation mode — set use_mt5=True for live orders")
    return None


# ══════════════════════════════════════════════════════════════
# MT5 INTEGRATION  (Windows only)
# ══════════════════════════════════════════════════════════════

def fetch_live_data_mt5(symbol="XAUUSD", timeframe=None, n=300):
    try:
        import MetaTrader5 as mt5
        tf = timeframe if timeframe is not None else mt5.TIMEFRAME_M5
        if not mt5.initialize():
            return []
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
        mt5.shutdown()
        if rates is None or len(rates) == 0:
            return []
        return [{"time": int(r["time"]), "open": float(r["open"]), "high": float(r["high"]),
                 "low": float(r["low"]), "close": float(r["close"]),
                 "volume": float(r["tick_volume"]) if "tick_volume" in r.dtype.names else 0.0} for r in rates]
    except ImportError:
        print("[MT5] Run: pip install MetaTrader5")
        return []
    except Exception as e:
        print(f"[MT5] Error: {e}")
        return []


def execute_trade_mt5(signal, symbol="XAUUSD", lot=0.01):
    """Routes to mt5_connection.py for clean separation of concerns."""
    try:
        from mt5_connection import connect, place_order, disconnect
        if not connect():
            print("[MT5] Could not connect — trade not sent")
            return None
        result = place_order(signal, symbol=symbol, lot=lot)
        disconnect()
        return result
    except ImportError:
        print("[MT5] mt5_connection.py not found or MetaTrader5 not installed")
        return None
    except Exception as e:
        print(f"[MT5] Error: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# AI NARRATIVE & SUMMARY
# ══════════════════════════════════════════════════════════════

def generate_ai_analysis(df, signals):
    if len(df) < 50:
        return "Not enough data for analysis."
    close      = df["close"]
    ema50      = compute_ema(close, 50)
    ema200     = compute_ema(close, 200)
    last_price = float(close.iloc[-1])
    trend      = "bullish" if ema50.iloc[-1] > ema200.iloc[-1] else \
                 "bearish" if ema50.iloc[-1] < ema200.iloc[-1] else "sideways"
    rsi_val    = float(compute_rsi(close).iloc[-1])
    momentum   = "strong bullish" if rsi_val > 60 else "strong bearish" if rsi_val < 40 else "weak / ranging"
    atr_val    = float(compute_atr(df).iloc[-1])
    last_sig   = signals[-1]["type"] if signals else "none"
    return (f"Market Analysis:\n- Price: {round(last_price, 5)}\n- Trend: {trend}\n"
            f"- Momentum: {momentum} (RSI {round(rsi_val,1)})\n"
            f"- Volatility (ATR): {round(atr_val, 5)}\n"
            f"Structure: {trend} with {momentum} momentum. Last signal: {last_sig}.\n"
            f"Strategy: Trade with the dominant trend. Avoid counter-trend entries "
            f"unless strong reversal confluence appears.")


def generate_summary(bias, htf_bias, last_rsi, last_close, ema20, ema50, ema200,
                     order_blocks, signals, symbol, timeframe):
    sym_label = {"EURUSD": "EUR/USD", "XAUUSD": "XAU/USD (Gold)", "USDJPY": "USD/JPY"}.get(symbol, symbol)
    trend = "Above EMA200 — bullish macro." if last_close > ema200 else "Below EMA200 — bearish macro."
    if last_close > ema50 > ema200:   trend += " EMAs stacked bullishly."
    elif last_close < ema50 < ema200: trend += " EMAs stacked bearishly."
    else:                              trend += " EMAs mixed."
    trend += f" 4H HTF: {htf_bias.upper()}."
    if last_rsi > 70:   rsi_desc = f"RSI {last_rsi:.1f} — overbought."
    elif last_rsi < 30: rsi_desc = f"RSI {last_rsi:.1f} — oversold."
    elif last_rsi > 55: rsi_desc = f"RSI {last_rsi:.1f} — bullish momentum."
    elif last_rsi < 45: rsi_desc = f"RSI {last_rsi:.1f} — bearish momentum."
    else:               rsi_desc = f"RSI {last_rsi:.1f} — neutral."
    bull_obs = [o for o in order_blocks if o["type"] == "bullish"]
    bear_obs = [o for o in order_blocks if o["type"] == "bearish"]
    ob_desc  = f"{len(bull_obs)} bullish OB(s), {len(bear_obs)} bearish OB(s)."
    if bull_obs: ob_desc += f" Demand: {bull_obs[-1]['bottom']}–{bull_obs[-1]['top']}."
    if bear_obs: ob_desc += f" Supply: {bear_obs[-1]['bottom']}–{bear_obs[-1]['top']}."
    if signals:
        s = signals[-1]
        sig_desc = (f"Latest: {s['type'].upper()} @ {s['price']} · SL {s['sl']} · "
                    f"TP {s['tp']} · 1:{s['rr']} R:R · {s.get('confidence','—')}% conf.")
    else:
        sig_desc = "No confirmed signals. Waiting for full confluence."
    if bias == "bullish" and htf_bias == "bullish":
        rec = "BULLISH — HTF confirmed. Buy from bullish OBs with BOS + FVG + liquidity."
    elif bias == "bearish" and htf_bias == "bearish":
        rec = "BEARISH — HTF confirmed. Sell from bearish OBs with BOS + FVG + liquidity."
    elif htf_bias == "neutral":
        rec = "HTF NEUTRAL — wait for direction before entering."
    else:
        rec = f"LTF ({bias}) conflicts with HTF ({htf_bias}). Wait for alignment."
    return {"symbol": sym_label, "timeframe": timeframe, "trend": trend,
            "rsi_desc": rsi_desc, "ob_desc": ob_desc, "sig_desc": sig_desc,
            "rec": rec, "bias": bias, "htf_bias": htf_bias}


# ══════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════

def run_analysis(candles, symbol="EURUSD", timeframe="1h"):
    global CONFIDENCE_THRESHOLD

    df = pd.DataFrame(candles)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = df["time"].astype(int)
    df = df.sort_values("time").reset_index(drop=True)

    closes = df["close"]
    times  = df["time"].values
    atr    = compute_atr(df)

    # Dynamic confidence threshold
    base_threshold   = 60 if symbol == "XAUUSD" else 65
    vol_factor       = 5 if atr.iloc[-1] > atr.mean() * 1.3 else 0
    CONFIDENCE_THRESHOLD = max(55, min(80, base_threshold + vol_factor))

    ema20  = compute_ema(closes, 20)
    ema50  = compute_ema(closes, 50)
    ema200 = compute_ema(closes, 200)
    rsi    = compute_rsi(closes, 14)

    ema_lines = {
        "ema20":  [{"time": int(times[i]), "value": round(float(ema20.iloc[i]),  5)} for i in range(len(df))],
        "ema50":  [{"time": int(times[i]), "value": round(float(ema50.iloc[i]),  5)} for i in range(len(df))],
        "ema200": [{"time": int(times[i]), "value": round(float(ema200.iloc[i]), 5)} for i in range(len(df))],
    }
    rsi_line = [{"time": int(times[i]), "value": round(float(rsi.iloc[i]), 2)} for i in range(len(df))]

    htf_bias_map       = get_htf_bias(df)
    bull_obs, bear_obs = detect_order_blocks(df)
    display_obs        = bull_obs[-8:] + bear_obs[-8:]
    signals            = detect_entry_signals(df, atr, htf_bias_map, for_display=True)

    last_close = float(closes.iloc[-1])
    last_e200  = float(ema200.iloc[-1])
    last_e50   = float(ema50.iloc[-1])
    last_rsi   = float(rsi.iloc[-1])

    bias = "bullish" if last_close > last_e200 and last_close > last_e50 else \
           "bearish" if last_close < last_e200 and last_close < last_e50 else "neutral"

    last_date = pd.Timestamp(int(times[-1]), unit="s").date()
    htf_bias  = htf_bias_map.get(last_date, "neutral")

    summary     = generate_summary(bias, htf_bias, last_rsi, last_close,
                                   float(ema20.iloc[-1]), float(ema50.iloc[-1]), last_e200,
                                   display_obs, signals, symbol, timeframe)
    ai_analysis = generate_ai_analysis(df, signals)

    if signals:
        best = max(signals, key=lambda x: x.get("confidence", 0))
        if best.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
            execute_trade(best, symbol=symbol, lot=best.get("lot", 0.01))

    return {
        "ema_lines":    ema_lines,
        "rsi":          rsi_line,
        "order_blocks": display_obs,
        "signals":      signals,
        "bias":         bias,
        "htf_bias":     htf_bias,
        "last_rsi":     round(last_rsi, 1),
        "summary":      summary,
        "ai_analysis":  ai_analysis,
    }