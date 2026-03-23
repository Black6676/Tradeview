import pandas as pd
import numpy as np


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
# HTF BIAS  (4H resampling — works with 500 x 1H bars)
# ══════════════════════════════════════════════════════════════

def get_htf_bias(df):
    """
    Resample input data to 4H candles and derive trend bias
    from EMA20 vs EMA50 relationship.
    Returns dict: {date -> 'bullish'|'bearish'|'neutral'}
    """
    df2 = df.copy()
    df2["datetime"] = pd.to_datetime(df2["time"], unit="s", utc=True)
    df2 = df2.set_index("datetime")

    h4 = df2[["open","high","low","close"]].resample("4h").agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
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
        if close > e20 and e20 > e50:
            bias = "bullish"
        elif close < e20 and e20 < e50:
            bias = "bearish"
        else:
            bias = "neutral"
        # Tag every hour in this 4H window with this bias
        for h in range(4):
            key = (dt + pd.Timedelta(hours=h)).date()
            bias_map[key] = bias

    return bias_map


# ══════════════════════════════════════════════════════════════
# MARKET STRUCTURE  (merged from your version)
# ══════════════════════════════════════════════════════════════

def detect_swings(df, lookback=5):
    """Identify swing highs and swing lows."""
    highs = df["high"].values
    lows  = df["low"].values
    swing_highs, swing_lows = [], []

    for i in range(lookback, len(df) - lookback):
        if highs[i] == max(highs[i - lookback:i + lookback + 1]):
            swing_highs.append((i, highs[i]))
        if lows[i]  == min(lows[i  - lookback:i + lookback + 1]):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def detect_structure_breaks(swing_highs, swing_lows):
    """
    Break of Structure (BOS):
      Bullish BOS = new swing high above previous swing high
      Bearish BOS = new swing low below previous swing low
    """
    breaks = []

    for i in range(1, len(swing_highs)):
        if swing_highs[i][1] > swing_highs[i - 1][1]:
            breaks.append({"type": "bullish_bos", "idx": swing_highs[i][0],
                           "level": swing_highs[i][1]})

    for i in range(1, len(swing_lows)):
        if swing_lows[i][1] < swing_lows[i - 1][1]:
            breaks.append({"type": "bearish_bos", "idx": swing_lows[i][0],
                           "level": swing_lows[i][1]})

    return breaks


# ══════════════════════════════════════════════════════════════
# LIQUIDITY DETECTION  (merged from your version)
# ══════════════════════════════════════════════════════════════

def detect_liquidity(df, threshold=0.0005):
    """
    Equal highs / equal lows = liquidity pools.
    Threshold scales with price for Gold (XAU).
    """
    highs = df["high"].values
    lows  = df["low"].values
    avg_price = float(df["close"].mean())

    # Scale threshold for higher-priced instruments (Gold ~2000 vs Forex ~1.1)
    scaled_threshold = threshold * max(1.0, avg_price / 2.0)

    liquidity = []
    for i in range(1, len(df)):
        if abs(highs[i] - highs[i - 1]) < scaled_threshold:
            liquidity.append({"type": "equal_highs", "idx": i,
                              "level": highs[i]})
        if abs(lows[i]  - lows[i - 1])  < scaled_threshold:
            liquidity.append({"type": "equal_lows",  "idx": i,
                              "level": lows[i]})

    return liquidity


# ══════════════════════════════════════════════════════════════
# ORDER BLOCKS
# ══════════════════════════════════════════════════════════════

def detect_order_blocks(df, lookback=10):
    """
    Full dataset scan for bullish and bearish order blocks.
    Impulse threshold: 1.2× ATR.
    Returns (bull_obs, bear_obs) separately so callers can filter.
    """
    atr    = compute_atr(df)
    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    times  = df["time"].values
    bull_obs, bear_obs = [], []

    for i in range(lookback, len(df) - 1):
        body      = abs(closes[i] - opens[i])
        threshold = 1.2 * atr.iloc[i]

        if closes[i] > opens[i] and body > threshold:
            for j in range(i - 1, max(i - lookback, 0) - 1, -1):
                if closes[j] < opens[j]:
                    bull_obs.append({
                        "time":   int(times[j]),
                        "top":    round(float(opens[j]),  5),
                        "bottom": round(float(closes[j]), 5),
                        "type":   "bullish",
                        "high":   round(float(highs[j]),  5),
                        "low":    round(float(lows[j]),   5),
                        "atr":    round(float(atr.iloc[j]), 5),
                        "signal_idx": i,
                    })
                    break

        if closes[i] < opens[i] and body > threshold:
            for j in range(i - 1, max(i - lookback, 0) - 1, -1):
                if closes[j] > opens[j]:
                    bear_obs.append({
                        "time":   int(times[j]),
                        "top":    round(float(closes[j]), 5),
                        "bottom": round(float(opens[j]),  5),
                        "type":   "bearish",
                        "high":   round(float(highs[j]),  5),
                        "low":    round(float(lows[j]),   5),
                        "atr":    round(float(atr.iloc[j]), 5),
                        "signal_idx": i,
                    })
                    break

    # Deduplicate by OB candle timestamp
    seen, unique_bull = set(), []
    for ob in bull_obs:
        if ob["time"] not in seen:
            seen.add(ob["time"])
            unique_bull.append(ob)

    seen, unique_bear = set(), []
    for ob in bear_obs:
        if ob["time"] not in seen:
            seen.add(ob["time"])
            unique_bear.append(ob)

    return unique_bull, unique_bear


# ══════════════════════════════════════════════════════════════
# SIGNAL ENGINE  (HTF + BOS + Liquidity + OB + EMA + RSI)
# ══════════════════════════════════════════════════════════════

def detect_entry_signals(df, atr_series, htf_bias_map, for_display=True):
    """
    Full confluence engine:
      1. HTF 4H bias must align with trade direction
      2. Price inside order block zone
      3. Price on correct side of EMA200
      4. RSI within valid band (45-70 buy / 30-55 sell)
      5. Break of Structure must have occurred (market shifted)
      6. Liquidity sweep present (price took out a pool before entry)
      7. SL = 1.5× ATR below/above OB (prevents immediate stop-out)
    """
    closes = df["close"]
    ema200 = compute_ema(closes, 200)
    rsi    = compute_rsi(closes, 14)

    bull_obs, bear_obs = detect_order_blocks(df)
    all_obs = bull_obs + bear_obs

    swing_highs, swing_lows = detect_swings(df)
    structure = detect_structure_breaks(swing_highs, swing_lows)
    liquidity = detect_liquidity(df)

    times    = df["time"].values
    signals  = []

    for i in range(200, len(df)):
        price   = float(closes.iloc[i])
        ts      = int(times[i])
        e200    = float(ema200.iloc[i])
        rsi_val = float(rsi.iloc[i])
        atr_val = float(atr_series.iloc[i])

        # HTF bias check
        candle_date = pd.Timestamp(ts, unit="s").date()
        htf = htf_bias_map.get(candle_date, "neutral")

        # BOS must be recent (within last 50 candles) to be relevant
        has_bull_bos = any(b["type"] == "bullish_bos" and i - 50 < b["idx"] < i for b in structure)
        has_bear_bos = any(b["type"] == "bearish_bos" and i - 50 < b["idx"] < i for b in structure)
        # Liquidity sweep must be recent (within last 20 candles)
        has_liq      = any(i - 20 < l["idx"] < i for l in liquidity)

        for ob in all_obs:
            ob_rows = df[df["time"] == ob["time"]]
            if ob_rows.empty or ob_rows.index[0] >= i:
                continue

            in_zone = ob["bottom"] <= price <= ob["top"]

            # ── BUY confluence ─────────────────────────────────
            if ob["type"] == "bullish" and htf == "bullish":
                if (in_zone and price > e200
                        and 45 <= rsi_val <= 70
                        and has_bull_bos
                        and has_liq):
                    sl = round(ob["bottom"] - atr_val * 1.5, 5)
                    tp = round(price + (price - sl) * 2.0, 5)
                    signals.append({
                        "time":  ts, "type": "buy",
                        "price": round(price, 5),
                        "rsi":   round(rsi_val, 1),
                        "sl": sl, "tp": tp, "rr": 2.0,
                        "atr":   round(atr_val, 5),
                        "htf":   htf,
                    })

            # ── SELL confluence ────────────────────────────────
            elif ob["type"] == "bearish" and htf == "bearish":
                if (in_zone and price < e200
                        and 30 <= rsi_val <= 55
                        and has_bear_bos
                        and has_liq):
                    sl = round(ob["top"] + atr_val * 1.5, 5)
                    tp = round(price - (sl - price) * 2.0, 5)
                    signals.append({
                        "time":  ts, "type": "sell",
                        "price": round(price, 5),
                        "rsi":   round(rsi_val, 1),
                        "sl": sl, "tp": tp, "rr": 2.0,
                        "atr":   round(atr_val, 5),
                        "htf":   htf,
                    })

    # Deduplicate — minimum gap between signals
    deduped, last_ts = [], 0
    for s in sorted(signals, key=lambda x: x["time"]):
        gap = 5 * 3600 if for_display else 3 * 3600
        if s["time"] - last_ts > gap:
            deduped.append(s)
            last_ts = s["time"]

    return deduped[-10:] if for_display else deduped


# ══════════════════════════════════════════════════════════════
# PLAIN ENGLISH SUMMARY
# ══════════════════════════════════════════════════════════════

def generate_summary(bias, htf_bias, last_rsi, last_close, ema20, ema50, ema200,
                     order_blocks, signals, symbol, timeframe):
    sym_label = {"EURUSD": "EUR/USD", "XAUUSD": "XAU/USD (Gold)",
                 "USDJPY": "USD/JPY"}.get(symbol, symbol)

    # Trend
    if last_close > ema200:
        trend = "Price is above EMA200 — bullish macro structure."
    else:
        trend = "Price is below EMA200 — bearish macro structure."

    if last_close > ema50 > ema200:
        trend += " EMA20/50 stacked bullishly — strong uptrend."
    elif last_close < ema50 < ema200:
        trend += " EMA20/50 stacked bearishly — strong downtrend."
    else:
        trend += " EMAs mixed — possible consolidation."
    trend += f" 4H HTF bias: {htf_bias.upper()}."

    # RSI
    if last_rsi > 70:
        rsi_desc = f"RSI {last_rsi:.1f} — overbought."
    elif last_rsi < 30:
        rsi_desc = f"RSI {last_rsi:.1f} — oversold."
    elif last_rsi > 55:
        rsi_desc = f"RSI {last_rsi:.1f} — bullish momentum."
    elif last_rsi < 45:
        rsi_desc = f"RSI {last_rsi:.1f} — bearish momentum."
    else:
        rsi_desc = f"RSI {last_rsi:.1f} — neutral momentum."

    # Order blocks
    bull_obs = [o for o in order_blocks if o["type"] == "bullish"]
    bear_obs = [o for o in order_blocks if o["type"] == "bearish"]
    ob_desc  = f"{len(bull_obs)} bullish OB(s), {len(bear_obs)} bearish OB(s) detected."
    if bull_obs:
        nb = bull_obs[-1]
        ob_desc += f" Nearest demand: {nb['bottom']} – {nb['top']}."
    if bear_obs:
        nb = bear_obs[-1]
        ob_desc += f" Nearest supply: {nb['bottom']} – {nb['top']}."

    # Signal
    if signals:
        last     = signals[-1]
        sig_type = last["type"].upper()
        sig_desc = (f"Latest signal: {sig_type} at {last['price']} · "
                    f"SL {last['sl']} · TP {last['tp']} · "
                    f"1:{last['rr']} R:R · HTF {last.get('htf','—')} confirmed.")
    else:
        sig_desc = ("No HTF-confirmed signals on this timeframe. "
                    "Waiting for OB + BOS + liquidity + EMA + RSI confluence.")

    # Recommendation
    if bias == "bullish" and htf_bias == "bullish":
        rec = "Bias BULLISH — confirmed by 4H HTF. Look for buys from bullish OBs with BOS + liquidity sweep confirmation."
    elif bias == "bearish" and htf_bias == "bearish":
        rec = "Bias BEARISH — confirmed by 4H HTF. Look for sells from bearish OBs with BOS + liquidity sweep confirmation."
    elif htf_bias == "neutral":
        rec = "4H HTF is NEUTRAL. No trades until higher timeframe gives a clear direction."
    else:
        rec = (f"LTF bias ({bias}) conflicts with 4H HTF ({htf_bias}). "
               "Wait for alignment before entering.")

    return {
        "symbol": sym_label, "timeframe": timeframe,
        "trend": trend, "rsi_desc": rsi_desc,
        "ob_desc": ob_desc, "sig_desc": sig_desc,
        "rec": rec, "bias": bias, "htf_bias": htf_bias,
    }



# ══════════════════════════════════════════════════════════════
# AI ANALYSIS NARRATIVE  (your addition)
# ══════════════════════════════════════════════════════════════

def generate_ai_analysis(df, signals):
    """
    Generates a plain-English market narrative combining trend,
    momentum, volatility and recent signal bias.
    """
    if len(df) < 50:
        return "Not enough data for analysis."

    close      = df["close"]
    ema50      = compute_ema(close, 50)
    ema200     = compute_ema(close, 200)
    last_price = float(close.iloc[-1])

    # Trend
    if ema50.iloc[-1] > ema200.iloc[-1]:
        trend = "bullish"
    elif ema50.iloc[-1] < ema200.iloc[-1]:
        trend = "bearish"
    else:
        trend = "sideways"

    # Momentum
    rsi     = compute_rsi(close)
    rsi_val = float(rsi.iloc[-1])
    if rsi_val > 60:
        momentum = "strong bullish momentum"
    elif rsi_val < 40:
        momentum = "strong bearish momentum"
    else:
        momentum = "weak / ranging momentum"

    # Volatility
    atr = compute_atr(df)
    vol = float(atr.iloc[-1])

    # Signal bias
    last_signal = signals[-1]["type"] if signals else "none"

    analysis = (
        f"Market Analysis:\n"
        f"- Current price: {round(last_price, 5)}\n"
        f"- Trend: {trend}\n"
        f"- Momentum: {momentum} (RSI: {round(rsi_val, 1)})\n"
        f"- Volatility (ATR): {round(vol, 5)}\n"
        f"Structure suggests a {trend} environment with {momentum}. "
        f"Recent signals indicate: {last_signal} bias.\n"
        f"Strategy: Look for confirmations in line with the dominant trend "
        f"and avoid counter-trend trades unless strong reversals occur."
    )
    return analysis.strip()



# ══════════════════════════════════════════════════════════════
# AI NARRATIVE ANALYSIS  (added by user)
# ══════════════════════════════════════════════════════════════

def generate_ai_analysis(df, signals):
    """
    Generate a plain-English market narrative covering trend,
    momentum, volatility and signal bias.
    """
    if len(df) < 50:
        return "Not enough data for analysis."

    close  = df["close"]
    ema50  = compute_ema(close, 50)
    ema200 = compute_ema(close, 200)
    last_price = float(close.iloc[-1])

    # Trend
    if ema50.iloc[-1] > ema200.iloc[-1]:
        trend = "bullish"
    elif ema50.iloc[-1] < ema200.iloc[-1]:
        trend = "bearish"
    else:
        trend = "sideways"

    # Momentum
    rsi     = compute_rsi(close)
    rsi_val = float(rsi.iloc[-1])
    if rsi_val > 60:
        momentum = "strong bullish momentum"
    elif rsi_val < 40:
        momentum = "strong bearish momentum"
    else:
        momentum = "weak / ranging momentum"

    # Volatility
    atr = compute_atr(df)
    vol = float(atr.iloc[-1])

    # Signal bias
    last_signal = signals[-1]["type"] if signals else "none"

    # Narrative
    analysis = (
        f"Market Analysis:\n"
        f"- Current price: {round(last_price, 5)}\n"
        f"- Trend: {trend}\n"
        f"- Momentum: {momentum} (RSI: {round(rsi_val, 1)})\n"
        f"- Volatility (ATR): {round(vol, 5)}\n"
        f"Structure suggests a {trend} environment with {momentum}. "
        f"Recent signals indicate: {last_signal} bias.\n"
        f"Strategy: Look for confirmations in line with the dominant trend "
        f"and avoid counter-trend trades unless strong reversals occur."
    )
    return analysis.strip()

# ══════════════════════════════════════════════════════════════
# MAIN RUNNER  (called by app.py and scanner.py)
# ══════════════════════════════════════════════════════════════

def run_analysis(candles, symbol="EURUSD", timeframe="1h"):
    df = pd.DataFrame(candles)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = df["time"].astype(int)
    df = df.sort_values("time").reset_index(drop=True)

    closes = df["close"]
    times  = df["time"].values
    atr    = compute_atr(df)

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

    htf_bias_map         = get_htf_bias(df)
    bull_obs, bear_obs   = detect_order_blocks(df)
    display_obs          = bull_obs[-8:] + bear_obs[-8:]
    signals              = detect_entry_signals(df, atr, htf_bias_map, for_display=True)

    last_close = float(closes.iloc[-1])
    last_e200  = float(ema200.iloc[-1])
    last_e50   = float(ema50.iloc[-1])
    last_rsi   = float(rsi.iloc[-1])

    if last_close > last_e200 and last_close > last_e50:
        bias = "bullish"
    elif last_close < last_e200 and last_close < last_e50:
        bias = "bearish"
    else:
        bias = "neutral"

    last_date = pd.Timestamp(int(times[-1]), unit="s").date()
    htf_bias  = htf_bias_map.get(last_date, "neutral")

    summary = generate_summary(
        bias, htf_bias, last_rsi, last_close,
        float(ema20.iloc[-1]), float(ema50.iloc[-1]), last_e200,
        display_obs, signals, symbol, timeframe
    )

    ai_analysis = generate_ai_analysis(df, signals)

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
