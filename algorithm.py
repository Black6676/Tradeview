import pandas as pd
import numpy as np
from datetime import datetime, timezone

try:
    from scipy.signal import argrelextrema
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# ══════════════════════════════════════════════════════════════
# GLOBAL SETTINGS
# ══════════════════════════════════════════════════════════════

ACCOUNT_BALANCE      = 1000
RISK_PER_TRADE       = 0.01
MAX_TRADES_PER_DAY   = 3
CONFIDENCE_THRESHOLD = 60
MAX_DAILY_LOSS_PCT   = 0.03

# Vantage MT5 credentials
VANTAGE_LOGIN    = 68336677
VANTAGE_PASSWORD = "Qwerty@12345"
VANTAGE_SERVER   = "RoboForex-Pro"

# Symbol-specific settings
SYMBOL_SETTINGS = {
    "XAUUSD": {"pip_value": 1.0,   "point_mult": 100,  "min_lot": 0.01, "spread_max_pct_atr": 0.15},
    "EURUSD": {"pip_value": 10.0,  "point_mult": 10000, "min_lot": 0.01, "spread_max_pct_atr": 0.10},
    "USDJPY": {"pip_value": 1000.0, "point_mult": 100,  "min_lot": 0.01, "spread_max_pct_atr": 0.10},
    "GBPUSD": {"pip_value": 10.0,  "point_mult": 10000, "min_lot": 0.01, "spread_max_pct_atr": 0.10},
}

_trade_history = []
_daily_stats   = {"date": None, "trades": 0, "loss": 0.0}


# ══════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════

def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def compute_sma(series, period):
    return series.rolling(window=period).mean()


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


def compute_adx(df, period=14):
    """Average Directional Index — trend strength. >25 trending, <20 ranging."""
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm  = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(span=period, adjust=False).mean().fillna(0)


def compute_bollinger_bands(series, period=20, std_dev=2.0):
    sma = compute_sma(series, period)
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


# ══════════════════════════════════════════════════════════════
# SESSION & KILL-ZONE FILTER
# ══════════════════════════════════════════════════════════════

def get_session_info(ts):
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    hour_utc = dt.hour
    weekday = dt.weekday()
    is_london = 7 <= hour_utc <= 10
    is_ny = 12 <= hour_utc <= 15
    is_london_close = 15 <= hour_utc <= 17
    is_asian = 0 <= hour_utc < 7
    is_weekend = weekday >= 5
    return {
        "hour": hour_utc, "weekday": weekday,
        "is_london": is_london, "is_ny": is_ny,
        "is_london_close": is_london_close, "is_asian": is_asian,
        "is_weekend": is_weekend, "is_killzone": is_london or is_ny,
        "session_name": "london" if is_london else ("ny" if is_ny else ("london_close" if is_london_close else ("asian" if is_asian else "other")))
    }


def is_trading_session(ts, require_killzone=True):
    info = get_session_info(ts)
    if info["is_weekend"]:
        return False
    if info["is_asian"]:
        return False
    if require_killzone and not info["is_killzone"]:
        return False
    return True


# ══════════════════════════════════════════════════════════════
# RISK MANAGEMENT
# ══════════════════════════════════════════════════════════════

def lot_size(sl_distance, symbol="XAUUSD", account_balance=None, risk_pct=None):
    balance  = account_balance or ACCOUNT_BALANCE
    risk_pct = risk_pct or RISK_PER_TRADE
    risk_amt = balance * risk_pct
    settings = SYMBOL_SETTINGS.get(symbol, SYMBOL_SETTINGS["EURUSD"])
    if sl_distance <= 0:
        return settings["min_lot"]
    lot = risk_amt / (sl_distance * settings["pip_value"] * settings["point_mult"])
    return round(max(lot, settings["min_lot"]), 2)


def check_daily_limit():
    today = datetime.now(timezone.utc).date()
    if _daily_stats["date"] != today:
        _daily_stats["date"] = today
        _daily_stats["trades"] = 0
        _daily_stats["loss"] = 0.0
    if _daily_stats["loss"] <= -ACCOUNT_BALANCE * MAX_DAILY_LOSS_PCT:
        return False
    if _daily_stats["trades"] >= MAX_TRADES_PER_DAY:
        return False
    return True


def record_trade_result(signal, result, pnl=0.0):
    _trade_history.append({**signal, "result": result, "pnl": pnl})
    if len(_trade_history) > 100:
        _trade_history.pop(0)
    today = datetime.now(timezone.utc).date()
    if _daily_stats["date"] != today:
        _daily_stats["date"] = today
        _daily_stats["trades"] = 0
        _daily_stats["loss"] = 0.0
    _daily_stats["trades"] += 1
    if pnl < 0:
        _daily_stats["loss"] += pnl
    adapt_strategy()


def adapt_strategy():
    global CONFIDENCE_THRESHOLD
    if len(_trade_history) < 15:
        return
    recent = _trade_history[-15:]
    wins = sum(1 for t in recent if t.get("result") == "win")
    rate = wins / len(recent)
    losses = [t["pnl"] for t in recent if t.get("pnl", 0) < 0]
    avg_loss = abs(np.mean(losses)) if losses else 0
    if rate < 0.35 or avg_loss > ACCOUNT_BALANCE * RISK_PER_TRADE * 2:
        CONFIDENCE_THRESHOLD = min(85, CONFIDENCE_THRESHOLD + 3)
    elif rate > 0.60 and avg_loss < ACCOUNT_BALANCE * RISK_PER_TRADE * 1.2:
        CONFIDENCE_THRESHOLD = max(55, CONFIDENCE_THRESHOLD - 3)


def apply_trade_management(trade, current_price, current_atr=0):
    entry = trade["entry_price"]
    sl    = trade["sl"]
    tp    = trade["tp"]
    risk  = abs(entry - sl)
    atr   = current_atr or trade.get("atr", 0)
    trade.setdefault("partial_tp_hit", False)
    trade.setdefault("breakeven_set", False)
    if trade["type"] == "buy":
        if not trade["breakeven_set"] and current_price - entry >= risk * 0.8:
            trade["sl"] = max(sl, entry)
            trade["breakeven_set"] = True
        if not trade["partial_tp_hit"] and current_price - entry >= risk * 1.5:
            trade["partial_tp_hit"] = True
            trade["close_50_percent"] = True
        if current_price - entry >= risk * 2.0 and atr > 0:
            trade["sl"] = max(trade["sl"], current_price - 2.0 * atr)
    elif trade["type"] == "sell":
        if not trade["breakeven_set"] and entry - current_price >= risk * 0.8:
            trade["sl"] = min(sl, entry)
            trade["breakeven_set"] = True
        if not trade["partial_tp_hit"] and entry - current_price >= risk * 1.5:
            trade["partial_tp_hit"] = True
            trade["close_50_percent"] = True
        if entry - current_price >= risk * 2.0 and atr > 0:
            trade["sl"] = min(trade["sl"], current_price + 2.0 * atr)
    return trade


# ══════════════════════════════════════════════════════════════
# MARKET REGIME DETECTION
# ══════════════════════════════════════════════════════════════

def detect_market_regime(df):
    if len(df) < 50:
        return "unknown"
    adx = compute_adx(df).iloc[-1]
    upper, mid, lower = compute_bollinger_bands(df["close"], 20, 2)
    bb_width = (upper.iloc[-1] - lower.iloc[-1]) / mid.iloc[-1] if mid.iloc[-1] != 0 else 0
    ema20 = compute_ema(df["close"], 20).iloc[-1]
    ema50 = compute_ema(df["close"], 50).iloc[-1]
    if adx > 25:
        return "trending_up" if ema20 > ema50 else "trending_down"
    elif adx < 18 and bb_width < 0.02:
        return "ranging"
    elif bb_width > 0.05:
        return "volatile"
    else:
        return "choppy"


# ══════════════════════════════════════════════════════════════
# SWING DETECTION
# ══════════════════════════════════════════════════════════════

def detect_swings(df, lookback=5):
    highs = df["high"].values
    lows  = df["low"].values
    if SCIPY_AVAILABLE:
        try:
            order = max(2, lookback)
            sh_idx = argrelextrema(highs, np.greater, order=order)[0]
            sl_idx = argrelextrema(lows,  np.less,    order=order)[0]
            swing_highs = [(int(i), float(highs[i])) for i in sh_idx if lookback <= i < len(df) - lookback]
            swing_lows  = [(int(i), float(lows[i]))  for i in sl_idx if lookback <= i < len(df) - lookback]
            return swing_highs, swing_lows
        except Exception:
            pass
    swing_highs, swing_lows = [], []
    for i in range(lookback, len(df) - lookback):
        if highs[i] > max(highs[i - lookback:i]) and highs[i] > max(highs[i + 1:i + lookback + 1]):
            swing_highs.append((i, float(highs[i])))
        if lows[i] < min(lows[i - lookback:i]) and lows[i] < min(lows[i + 1:i + lookback + 1]):
            swing_lows.append((i, float(lows[i])))
    return swing_highs, swing_lows


# ══════════════════════════════════════════════════════════════
# MARKET STRUCTURE — BOS + CHoCH + MSS
# ══════════════════════════════════════════════════════════════

def _is_displacement_candle(df, idx, atr_mult=1.5):
    if idx >= len(df):
        return False
    atr = compute_atr(df).iloc[idx]
    if atr == 0 or pd.isna(atr):
        return False
    o, h, l, c = df["open"].iloc[idx], df["high"].iloc[idx], df["low"].iloc[idx], df["close"].iloc[idx]
    body = abs(c - o)
    wick = (h - l) - body
    return body >= atr * atr_mult and wick <= body * 0.3


def detect_structure(swing_highs, swing_lows, df, lookback=5):
    events = []
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return events
    all_swings = sorted([(i, "high", p) for i, p in swing_highs] + [(i, "low", p) for i, p in swing_lows])
    for idx in range(2, len(all_swings)):
        prev2 = all_swings[idx - 2]
        prev1 = all_swings[idx - 1]
        curr  = all_swings[idx]
        if curr[1] == "high" and prev2[1] == "high":
            if curr[2] > prev2[2]:
                is_disp = _is_displacement_candle(df, curr[0])
                events.append({"type": "bullish_bos", "idx": curr[0], "level": curr[2],
                               "displacement": is_disp, "strength": "strong" if is_disp else "weak"})
        if curr[1] == "low" and prev2[1] == "low":
            if curr[2] < prev2[2]:
                is_disp = _is_displacement_candle(df, curr[0])
                events.append({"type": "bearish_bos", "idx": curr[0], "level": curr[2],
                               "displacement": is_disp, "strength": "strong" if is_disp else "weak"})
        if curr[1] == "high" and prev1[1] == "low" and prev2[1] == "high":
            if curr[2] > prev2[2]:
                is_disp = _is_displacement_candle(df, curr[0])
                events.append({"type": "bullish_choch" if not is_disp else "bullish_mss",
                               "idx": curr[0], "level": curr[2], "displacement": is_disp})
        if curr[1] == "low" and prev1[1] == "high" and prev2[1] == "low":
            if curr[2] < prev2[2]:
                is_disp = _is_displacement_candle(df, curr[0])
                events.append({"type": "bearish_choch" if not is_disp else "bearish_mss",
                               "idx": curr[0], "level": curr[2], "displacement": is_disp})
    return events


# ══════════════════════════════════════════════════════════════
# LIQUIDITY & SWEEPS
# ══════════════════════════════════════════════════════════════

def detect_liquidity(df, threshold=0.0005):
    highs = df["high"].values
    lows  = df["low"].values
    avg_price = float(df["close"].mean())
    scaled = threshold * max(1.0, avg_price / 2.0)
    liq = []
    for i in range(1, len(df)):
        if abs(highs[i] - highs[i - 1]) < scaled:
            liq.append({"type": "equal_highs", "idx": i, "level": float(highs[i]), "swept": False})
        if abs(lows[i]  - lows[i - 1])  < scaled:
            liq.append({"type": "equal_lows",  "idx": i, "level": float(lows[i]), "swept": False})
    return liq


def detect_liquidity_sweeps(df, liquidity, lookforward=10):
    highs = df["high"].values
    lows  = df["low"].values
    closes = df["close"].values
    swept = []
    for liq in liquidity:
        idx = liq["idx"]
        level = liq["level"]
        liq_type = liq["type"]
        for j in range(idx + 1, min(idx + lookforward, len(df))):
            if liq_type == "equal_highs":
                if highs[j] > level and closes[j] < level:
                    if j + 1 < len(df) and closes[j+1] < closes[j]:
                        swept.append({**liq, "swept": True, "sweep_idx": j,
                                      "sweep_type": "buy_side_liquidity_sweep"})
                        break
            else:
                if lows[j] < level and closes[j] > level:
                    if j + 1 < len(df) and closes[j+1] > closes[j]:
                        swept.append({**liq, "swept": True, "sweep_idx": j,
                                      "sweep_type": "sell_side_liquidity_sweep"})
                        break
    return swept


# ══════════════════════════════════════════════════════════════
# FAIR VALUE GAPS
# ══════════════════════════════════════════════════════════════

def detect_fvg(df):
    fvgs = []
    atr = compute_atr(df)
    for i in range(2, len(df) - 1):
        if df["low"].iloc[i] > df["high"].iloc[i-2]:
            gap_size = df["low"].iloc[i] - df["high"].iloc[i-2]
            atr_val = atr.iloc[i]
            quality = min(1.0, gap_size / (atr_val * 0.5)) if atr_val > 0 else 0.5
            fvgs.append({"idx": i, "time": int(df["time"].iloc[i-1]),
                         "top": float(df["low"].iloc[i]),
                         "bottom": float(df["high"].iloc[i-2]),
                         "type": "bullish", "quality": round(float(quality), 2),
                         "gap_size": round(float(gap_size), 5)})
        elif df["high"].iloc[i] < df["low"].iloc[i-2]:
            gap_size = df["low"].iloc[i-2] - df["high"].iloc[i]
            atr_val = atr.iloc[i]
            quality = min(1.0, gap_size / (atr_val * 0.5)) if atr_val > 0 else 0.5
            fvgs.append({"idx": i, "time": int(df["time"].iloc[i-1]),
                         "top": float(df["low"].iloc[i-2]),
                         "bottom": float(df["high"].iloc[i]),
                         "type": "bearish", "quality": round(float(quality), 2),
                         "gap_size": round(float(gap_size), 5)})
    return fvgs


# ══════════════════════════════════════════════════════════════
# ORDER BLOCKS — Strict with displacement + BOS
# ══════════════════════════════════════════════════════════════

def detect_order_blocks(df, lookback=10):
    atr = compute_atr(df)
    o = df["open"].values
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    t = df["time"].values
    bull, bear = [], []
    for i in range(lookback, len(df) - 2):
        if not _is_displacement_candle(df, i, atr_mult=1.2):
            continue
        is_bull = c[i] > o[i]
        is_bear = c[i] < o[i]
        ob_candle = None
        for j in range(i - 1, max(i - lookback, -1), -1):
            if is_bull and c[j] < o[j]:
                ob_candle = j; break
            if is_bear and c[j] > o[j]:
                ob_candle = j; break
        if ob_candle is None:
            continue
        recent_window = df.iloc[max(0, i-20):i]
        if is_bull:
            recent_high = recent_window["high"].max()
            if c[i] <= recent_high:
                continue
        else:
            recent_low = recent_window["low"].min()
            if c[i] >= recent_low:
                continue
        j = ob_candle
        ob_top = float(max(o[j], c[j]))
        ob_bot = float(min(o[j], c[j]))
        mitigated = False
        for k in range(j + 1, i):
            if ob_bot <= c[k] <= ob_top:
                mitigated = True; break
        if mitigated:
            continue
        entry = {
            "time": int(t[j]), "top": round(ob_top, 5), "bottom": round(ob_bot, 5),
            "type": "bullish" if is_bull else "bearish",
            "high": round(float(h[j]), 5), "low": round(float(l[j]), 5),
            "atr": round(float(atr.iloc[j]), 5), "displacement_idx": i,
            "bos_confirmed": True, "fresh": True,
        }
        ob_size = ob_top - ob_bot
        atr_val = atr.iloc[j]
        if atr_val > 0:
            entry["quality"] = round(min(1.0, ob_size / atr_val / 1.5), 2)
        else:
            entry["quality"] = 0.5
        (bull if is_bull else bear).append(entry)
    seen, unique_bull = set(), []
    for ob in bull:
        if ob["time"] not in seen:
            seen.add(ob["time"]); unique_bull.append(ob)
    seen, unique_bear = set(), []
    for ob in bear:
        if ob["time"] not in seen:
            seen.add(ob["time"]); unique_bear.append(ob)
    return unique_bull, unique_bear


def detect_breaker_blocks(df, existing_obs):
    breakers_bull, breakers_bear = [], []
    c = df["close"].values
    for ob in existing_obs:
        ob_rows = df[df["time"] == ob["time"]]
        if ob_rows.empty:
            continue
        ob_idx = ob_rows.index[0]
        for i in range(ob_idx + 1, min(ob_idx + 50, len(df))):
            if ob["type"] == "bullish":
                if c[i] < ob["low"] and _is_displacement_candle(df, i, atr_mult=1.0):
                    breakers_bear.append({"time": ob["time"], "top": ob["top"], "bottom": ob["bottom"],
                                          "type": "bearish_breaker", "original_type": "bullish",
                                          "break_idx": i, "atr": ob["atr"]})
                    break
            else:
                if c[i] > ob["high"] and _is_displacement_candle(df, i, atr_mult=1.0):
                    breakers_bull.append({"time": ob["time"], "top": ob["top"], "bottom": ob["bottom"],
                                          "type": "bullish_breaker", "original_type": "bearish",
                                          "break_idx": i, "atr": ob["atr"]})
                    break
    return breakers_bull, breakers_bear


# ══════════════════════════════════════════════════════════════
# HTF BIAS + Premium/Discount
# ══════════════════════════════════════════════════════════════

def get_htf_bias(df):
    df2 = df.copy()
    df2["datetime"] = pd.to_datetime(df2["time"], unit="s", utc=True)
    df2 = df2.set_index("datetime")
    h4  = df2[["open","high","low","close"]].resample("4h").agg({
        "open":"first","high":"max","low":"min","close":"last"
    }).dropna()
    if len(h4) < 25:
        return {"bias_map": {}, "ranges": {}}
    h4_ema20 = compute_ema(h4["close"], 20)
    h4_ema50 = compute_ema(h4["close"], 50)
    bias_map = {}
    ranges = {}
    for dt in h4.index:
        close = h4.loc[dt, "close"]
        e20   = h4_ema20.loc[dt]
        e50   = h4_ema50.loc[dt]
        if close > e20 > e50:
            bias = "bullish"
        elif close < e20 < e50:
            bias = "bearish"
        else:
            bias = "neutral"
        recent = h4.loc[:dt].tail(20)
        range_high = recent["high"].max()
        range_low  = recent["low"].min()
        eq = (range_high + range_low) / 2
        for h in range(4):
            key = (dt + pd.Timedelta(hours=h)).date()
            bias_map[key] = bias
            ranges[key] = {"high": range_high, "low": range_low, "eq": eq}
    return {"bias_map": bias_map, "ranges": ranges}


def is_premium_discount(price, range_info, direction):
    eq = range_info.get("eq", price)
    if direction == "buy":
        return "discount" if price < eq else ("premium" if price > eq else "equilibrium")
    else:
        return "premium" if price > eq else ("discount" if price < eq else "equilibrium")


# ══════════════════════════════════════════════════════════════
# CONFIDENCE SCORE
# ══════════════════════════════════════════════════════════════

def compute_confidence(signal, trend, rsi_val, htf_bias, regime,
                       liquidity_sweep, ob_quality, fvg_quality=0, premium_discount="equilibrium"):
    score = 0
    if signal["type"] == "buy" and trend == "bullish" and htf_bias == "bullish":
        score += 30
    elif signal["type"] == "sell" and trend == "bearish" and htf_bias == "bearish":
        score += 30
    elif (signal["type"] == "buy" and trend == "bullish") or \
         (signal["type"] == "sell" and trend == "bearish"):
        score += 15
    if regime in ("trending_up", "trending_down"):
        if (signal["type"] == "buy" and regime == "trending_up") or \
           (signal["type"] == "sell" and regime == "trending_down"):
            score += 15
        else:
            score -= 10
    elif regime == "ranging":
        score += 5
    elif regime == "volatile":
        score -= 5
    if signal["type"] == "buy":
        if 55 < rsi_val <= 70:    score += 20
        elif 45 <= rsi_val <= 55: score += 10
        elif rsi_val > 70:        score -= 10
    elif signal["type"] == "sell":
        if 30 <= rsi_val < 45:    score += 20
        elif 45 <= rsi_val <= 55: score += 10
        elif rsi_val < 30:        score -= 10
    rr = signal.get("rr", 0)
    if rr >= 3:     score += 15
    elif rr >= 2:   score += 10
    elif rr >= 1.5: score += 5
    if liquidity_sweep:
        score += 10
    score += int(ob_quality * 5)
    score += int(fvg_quality * 5)
    if premium_discount == "discount" and signal["type"] == "buy":
        score += 5
    elif premium_discount == "premium" and signal["type"] == "sell":
        score += 5
    elif premium_discount == "premium" and signal["type"] == "buy":
        score -= 10
    elif premium_discount == "discount" and signal["type"] == "sell":
        score -= 10
    return max(0, min(100, score))


# ══════════════════════════════════════════════════════════════
# SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════

def detect_entry_signals(df, atr_series, htf_data, symbol="XAUUSD", for_display=True):
    closes = df["close"]
    ema200 = compute_ema(closes, 200)
    ema50  = compute_ema(closes, 50)
    rsi    = compute_rsi(closes, 14)
    regime = detect_market_regime(df)
    bull_obs, bear_obs = detect_order_blocks(df)
    breakers_bull, breakers_bear = detect_breaker_blocks(df, bull_obs + bear_obs)
    all_obs = bull_obs + bear_obs + breakers_bull + breakers_bear
    swing_h, swing_l = detect_swings(df)
    structure = detect_structure(swing_h, swing_l, df)
    liquidity = detect_liquidity(df)
    sweeps = detect_liquidity_sweeps(df, liquidity)
    fvgs = detect_fvg(df)
    times = df["time"].values
    signals = []
    sym_settings = SYMBOL_SETTINGS.get(symbol, SYMBOL_SETTINGS["EURUSD"])
    for i in range(200, len(df)):
        ts = int(times[i])
        session = get_session_info(ts)
        if not session["is_killzone"]:
            continue
        price   = float(closes.iloc[i])
        e200    = float(ema200.iloc[i])
        e50     = float(ema50.iloc[i])
        rsi_val = float(rsi.iloc[i])
        atr_val = float(atr_series.iloc[i])
        candle_spread = float(df["high"].iloc[i] - df["low"].iloc[i])
        if candle_spread > atr_val * sym_settings["spread_max_pct_atr"] * 3:
            continue
        candle_date  = pd.Timestamp(ts, unit="s").date()
        htf          = htf_data.get("bias_map", {}).get(candle_date, "neutral")
        range_info   = htf_data.get("ranges", {}).get(candle_date, {})
        recent_struct = [s for s in structure if i - 50 < s["idx"] <= i]
        has_bull_bos = any(s["type"] == "bullish_bos" and s.get("displacement") for s in recent_struct)
        has_bear_bos = any(s["type"] == "bearish_bos" and s.get("displacement") for s in recent_struct)
        has_bull_mss = any(s["type"] in ("bullish_choch", "bullish_mss") for s in recent_struct)
        has_bear_mss = any(s["type"] in ("bearish_choch", "bearish_mss") for s in recent_struct)
        recent_sweeps = [s for s in sweeps if i - 30 < s.get("sweep_idx", 0) <= i]
        has_bull_sweep = any(s["sweep_type"] == "sell_side_liquidity_sweep" for s in recent_sweeps)
        has_bear_sweep = any(s["sweep_type"] == "buy_side_liquidity_sweep" for s in recent_sweeps)
        recent_fvgs = [f for f in fvgs if i - 40 < f["idx"] < i]
        has_bull_fvg = any(f["type"] == "bullish" for f in recent_fvgs)
        has_bear_fvg = any(f["type"] == "bearish" for f in recent_fvgs)
        best_bull_fvg = max([f["quality"] for f in recent_fvgs if f["type"] == "bullish"], default=0)
        best_bear_fvg = max([f["quality"] for f in recent_fvgs if f["type"] == "bearish"], default=0)
        candidates = []
        for ob in all_obs:
            ob_rows = df[df["time"] == ob["time"]]
            if ob_rows.empty or ob_rows.index[0] >= i:
                continue
            ob_idx = ob_rows.index[0]
            mitigated = False
            for k in range(ob_idx + 1, i + 1):
                if ob["bottom"] <= closes.iloc[k] <= ob["top"]:
                    mitigated = True; break
            if mitigated and ob.get("type") not in ("bullish_breaker", "bearish_breaker"):
                continue
            in_zone = ob["bottom"] <= price <= ob["top"]
            near_zone = ob["bottom"] - atr_val * 0.5 <= price <= ob["top"] + atr_val * 0.5
            if not (in_zone or near_zone):
                continue
            ob_type = ob["type"]
            is_breaker = "breaker" in ob_type
            if ob_type in ("bullish", "bullish_breaker"):
                if htf != "bullish" and not is_breaker:
                    continue
                if price < e200 * 0.998:
                    continue
                score = 15
                if has_bull_bos: score += 20
                if has_bull_mss: score += 25
                if has_bull_sweep: score += 20
                if has_bull_fvg: score += 10
                if 45 <= rsi_val <= 70: score += 10
                if is_breaker: score += 5
                if score >= 35:
                    sl = round(ob["bottom"] - atr_val * 1.2, 5)
                    tp = round(price + (price - sl) * 2.5, 5)
                    rr = round((tp - price) / (price - sl), 2) if price != sl else 1.0
                    sig = {"time": ts, "type": "buy", "price": round(price, 5),
                           "rsi": round(rsi_val, 1), "sl": sl, "tp": tp, "rr": rr,
                           "atr": round(atr_val, 5), "htf": htf,
                           "lot": lot_size(round(price - sl, 5), symbol),
                           "regime": regime, "session": session["session_name"],
                           "ob_quality": ob.get("quality", 0.5), "is_breaker": is_breaker,
                           "raw_score": score}
                    pd_zone = is_premium_discount(price, range_info, "buy")
                    sig["confidence"] = compute_confidence(
                        sig, "bullish", rsi_val, htf, regime,
                        has_bull_sweep, ob.get("quality", 0.5), best_bull_fvg, pd_zone)
                    sig["premium_discount"] = pd_zone
                    try:
                        from ml_model import predict_win_probability
                        sig["ml_prob"] = predict_win_probability(sig, df, i)
                    except Exception:
                        sig["ml_prob"] = sig["confidence"] / 100.0
                    candidates.append(sig)
            elif ob_type in ("bearish", "bearish_breaker"):
                if htf != "bearish" and not is_breaker:
                    continue
                if price > e200 * 1.002:
                    continue
                score = 15
                if has_bear_bos: score += 20
                if has_bear_mss: score += 25
                if has_bear_sweep: score += 20
                if has_bear_fvg: score += 10
                if 30 <= rsi_val <= 55: score += 10
                if is_breaker: score += 5
                if score >= 35:
                    sl = round(ob["top"] + atr_val * 1.2, 5)
                    tp = round(price - (sl - price) * 2.5, 5)
                    rr = round((price - tp) / (sl - price), 2) if price != sl else 1.0
                    sig = {"time": ts, "type": "sell", "price": round(price, 5),
                           "rsi": round(rsi_val, 1), "sl": sl, "tp": tp, "rr": rr,
                           "atr": round(atr_val, 5), "htf": htf,
                           "lot": lot_size(round(sl - price, 5), symbol),
                           "regime": regime, "session": session["session_name"],
                           "ob_quality": ob.get("quality", 0.5), "is_breaker": is_breaker,
                           "raw_score": score}
                    pd_zone = is_premium_discount(price, range_info, "sell")
                    sig["confidence"] = compute_confidence(
                        sig, "bearish", rsi_val, htf, regime,
                        has_bear_sweep, ob.get("quality", 0.5), best_bear_fvg, pd_zone)
                    sig["premium_discount"] = pd_zone
                    try:
                        from ml_model import predict_win_probability
                        sig["ml_prob"] = predict_win_probability(sig, df, i)
                    except Exception:
                        sig["ml_prob"] = sig["confidence"] / 100.0
                    candidates.append(sig)
        if candidates:
            best = max(candidates, key=lambda x: x["confidence"])
            if best["confidence"] >= CONFIDENCE_THRESHOLD:
                signals.append(best)
    deduped, last_ts = [], 0
    gap = 4 * 3600 if for_display else 2 * 3600
    for s in sorted(signals, key=lambda x: x["time"]):
        if s["time"] - last_ts > gap:
            deduped.append(s); last_ts = s["time"]
    return deduped[-10:] if for_display else deduped


# ══════════════════════════════════════════════════════════════
# MT5 INTEGRATION
# ══════════════════════════════════════════════════════════════

def mt5_connect():
    if not MT5_AVAILABLE:
        print("[MT5] MetaTrader5 package not available on this platform")
        return False
    if mt5.initialize():
        info = mt5.account_info()
        if info and info.login > 0:
            print(f"[MT5] Connected to existing session — {info.server} | Balance: ${info.balance:.2f}")
            return True
        mt5.shutdown()
    if mt5.initialize(login=VANTAGE_LOGIN, password=VANTAGE_PASSWORD, server=VANTAGE_SERVER):
        print(f"[MT5] Connected to {VANTAGE_SERVER}")
        return True
    print(f"[MT5] Login failed: {mt5.last_error()}")
    return False


def fetch_live_data_mt5(symbol="XAUUSD", timeframe=None, n=500):
    if not MT5_AVAILABLE:
        return []
    try:
        tf = timeframe or mt5.TIMEFRAME_H1
        if not mt5_connect():
            return []
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
        mt5.shutdown()
        if rates is None or len(rates) == 0:
            return []
        return [{"time": int(r["time"]), "open": float(r["open"]),
                 "high": float(r["high"]), "low": float(r["low"]),
                 "close": float(r["close"]),
                 "volume": float(r["tick_volume"]) if "tick_volume" in r.dtype.names else 0.0}
                for r in rates]
    except Exception as e:
        print(f"[MT5] fetch error: {e}")
        return []


def execute_trade_mt5(signal, symbol="XAUUSD", lot=None):
    if not MT5_AVAILABLE:
        print("[MT5] Not available on this platform")
        return None
    if not check_daily_limit():
        print("[MT5] Daily limit reached — trade blocked")
        return None
    try:
        if not mt5_connect():
            return None
        lot = lot or signal.get("lot", 0.01)
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            mt5.shutdown(); return None
        sym_info = mt5.symbol_info(symbol)
        if sym_info and not sym_info.visible:
            mt5.symbol_select(symbol, True)
        price      = tick.ask if signal["type"] == "buy" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if signal["type"] == "buy" else mt5.ORDER_TYPE_SELL
        digits     = sym_info.digits if sym_info else 5
        stop_level = getattr(sym_info, "trade_stops_level", 0) or 0
        point      = getattr(sym_info, "point", 0.00001) or 0.00001
        atr_val    = signal.get("atr", 0)
        min_dist   = max((stop_level + 10) * point, atr_val * 0.5)
        sl = round(signal["sl"], digits)
        tp = round(signal["tp"], digits)
        if signal["type"] == "buy":
            if price - sl < min_dist:
                sl = round(price - min_dist, digits)
            if tp - price < min_dist:
                tp = round(price + min_dist * 2, digits)
        else:
            if sl - price < min_dist:
                sl = round(price + min_dist, digits)
            if price - tp < min_dist:
                tp = round(price - min_dist * 2, digits)
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lot,
            "type":         order_type,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "deviation":    30,
            "magic":        123456,
            "comment":      f"SMCv2|{signal.get('confidence','—')}%|{signal.get('regime','—')}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        mt5.shutdown()
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[MT5] ✓ {signal['type'].upper()} {symbol} @ {price:.5f} | Lot {lot} | Conf {signal.get('confidence','—')}%")
        else:
            print(f"[MT5] ✗ {result.comment if result else mt5.last_error()}")
        return result
    except Exception as e:
        print(f"[MT5] Error: {e}")
        try: mt5.shutdown()
        except: pass
        return None


# ══════════════════════════════════════════════════════════════
# AI NARRATIVE
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
    atr_val    = float(compute_atr(df).iloc[-1])
    regime     = detect_market_regime(df)
    last_sig   = signals[-1]["type"] if signals else "none"
    return (f"Market Analysis:
"
            f"- Price: {round(last_price, 5)}
"
            f"- Trend: {trend}
"
            f"- Regime: {regime}
"
            f"- Momentum: RSI {round(rsi_val,1)}
"
            f"- Volatility (ATR): {round(atr_val, 5)}
"
            f"- Last Signal: {last_sig.upper()}
"
            f"Strategy: Trade WITH the regime. In {regime}, prefer \"
            f"{'trend continuation' if 'trend' in regime else 'range extremes' if regime == 'ranging' else 'caution'}. \"
            f"Avoid entries during volatile/news candles.")


def generate_summary(bias, htf_bias, last_rsi, last_close, ema20, ema50, ema200,
                     order_blocks, signals, symbol, timeframe):
    sym_label = {"EURUSD": "EUR/USD", "XAUUSD": "XAU/USD (Gold)",
                 "USDJPY": "USD/JPY"}.get(symbol, symbol)
    trend = "Above EMA200 — bullish." if last_close > ema200 else "Below EMA200 — bearish."
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
        sig_desc = (f"Latest: {s['type'].upper()} @ {s['price']} · "
                    f"SL {s['sl']} · TP {s['tp']} · 1:{s['rr']} · "
                    f"{s.get('confidence','—')}% conf · {s.get('regime','—')}")
    else:
        sig_desc = "No confirmed signals. Waiting for confluence."
    if bias == "bullish" and htf_bias == "bullish":
        rec = "BULLISH — HTF confirmed. Buy from bullish OBs in discount."
    elif bias == "bearish" and htf_bias == "bearish":
        rec = "BEARISH — HTF confirmed. Sell from bearish OBs in premium."
    elif htf_bias == "neutral":
        rec = "HTF NEUTRAL — wait for directional bias or range trade with tight risk."
    else:
        rec = f"LTF ({bias}) conflicts with HTF ({htf_bias}). Wait for alignment or CHoCH."
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
    if len(df) < 300:
        return {"error": "Need at least 300 candles for reliable analysis"}
    closes = df["close"]
    times  = df["time"].values
    atr    = compute_atr(df)
    base_threshold = 65 if symbol == "XAUUSD" else 70
    vol_factor = 5 if atr.iloc[-1] > atr.mean() * 1.3 else 0
    regime = detect_market_regime(df)
    if regime == "ranging":
        base_threshold += 5
    elif regime == "volatile":
        base_threshold += 10
    CONFIDENCE_THRESHOLD = max(55, min(85, base_threshold + vol_factor))
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
    htf_data       = get_htf_bias(df)
    bull_obs, bear_obs = detect_order_blocks(df)
    breakers_bull, breakers_bear = detect_breaker_blocks(df, bull_obs + bear_obs)
    display_obs    = bull_obs[-6:] + bear_obs[-6:] + breakers_bull[-3:] + breakers_bear[-3:]
    signals        = detect_entry_signals(df, atr, htf_data, symbol=symbol, for_display=True)
    last_close = float(closes.iloc[-1])
    last_e200  = float(ema200.iloc[-1])
    last_e50   = float(ema50.iloc[-1])
    last_rsi   = float(rsi.iloc[-1])
    bias = "bullish" if last_close > last_e200 and last_close > last_e50 else \
           "bearish" if last_close < last_e200 and last_close < last_e50 else "neutral"
    last_date = pd.Timestamp(int(times[-1]), unit="s").date()
    htf_bias  = htf_data.get("bias_map", {}).get(last_date, "neutral")
    summary     = generate_summary(bias, htf_bias, last_rsi, last_close,
                                   float(ema20.iloc[-1]), float(ema50.iloc[-1]), last_e200,
                                   display_obs, signals, symbol, timeframe)
    ai_analysis = generate_ai_analysis(df, signals)
    if signals:
        best = max(signals, key=lambda x: x.get("confidence", 0))
        print(f"[Signal] {best['type'].upper()} {symbol} @ {best['price']} | "
              f"Conf: {best.get('confidence',0)}% | Regime: {best.get('regime','—')} | "
              f"PD: {best.get('premium_discount','—')}")
    print(f"[Analysis] {symbol} {timeframe} | {len(signals)} signals | "
          f"Bias: {bias} | HTF: {htf_bias} | Regime: {regime} | Threshold: {CONFIDENCE_THRESHOLD}")
    return {
        "ema_lines":    ema_lines,
        "rsi":          rsi_line,
        "order_blocks": display_obs,
        "signals":      signals,
        "bias":         bias,
        "htf_bias":     htf_bias,
        "regime":       regime,
        "last_rsi":     round(last_rsi, 1),
        "summary":      summary,
        "ai_analysis":  ai_analysis,
    }


# ══════════════════════════════════════════════════════════════
# DEMO / STANDALONE RUN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SMC Trading Algorithm — Standalone Demo")
    print("=" * 60)

    # Try MT5 first
    candles = fetch_live_data_mt5(symbol="XAUUSD", n=500)

    if not candles:
        print("\n[MT5] No live data. Generating demo candles for testing...")
        import random
        candles = []
        price = 2400.00
        for t in range(500):
            open_p = price
            close_p = price + random.uniform(-3, 3)
            high_p = max(open_p, close_p) + random.uniform(0, 2)
            low_p = min(open_p, close_p) - random.uniform(0, 2)
            candles.append({
                "time": 1720000000 + t * 3600,
                "open": round(open_p, 5), "high": round(high_p, 5),
                "low": round(low_p, 5), "close": round(close_p, 5), "volume": 100
            })
            price = close_p

    result = run_analysis(candles, symbol="XAUUSD", timeframe="1h")

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    if "error" in result:
        print(f"ERROR: {result['error']}")
    else:
        print(f"\nRegime: {result['regime']}")
        print(f"Bias: {result['bias']} | HTF: {result['htf_bias']}")
        print(f"\n{result['ai_analysis']}")
        print(f"\n{result['summary']['rec']}")

        if result['signals']:
            print(f"\n--- Signals ({len(result['signals'])}) ---")
            for s in result['signals']:
                print(f"  {s['type'].upper()} @ {s['price']} | SL: {s['sl']} | TP: {s['tp']} | "
                      f"RR: 1:{s['rr']} | Conf: {s['confidence']}% | {s.get('premium_discount','')}")
        else:
            print("\nNo signals generated.")

        print(f"\n--- Order Blocks ({len(result['order_blocks'])}) ---")
        for ob in result['order_blocks'][-5:]:
            print(f"  {ob['type'].upper()} OB: {ob['bottom']}–{ob['top']} | Quality: {ob.get('quality','N/A')}")
