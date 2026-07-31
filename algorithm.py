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

# GLOBAL SETTINGS
ACCOUNT_BALANCE      = 1000
RISK_PER_TRADE       = 0.01
MAX_TRADES_PER_DAY   = 3
BASE_CONFIDENCE      = 60   # fixed base, never mutated by backtest

VANTAGE_LOGIN    = 25788296
VANTAGE_PASSWORD = "Black@123"
VANTAGE_SERVER   = "VantageMarkets-Demo"

_trade_history = []

# INDICATORS

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

# RISK MANAGEMENT & SESSION FILTER

def is_trading_session(ts):
    hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour
    return 7 <= hour <= 21


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
        if atr > 0:
            trade["sl"] = max(trade["sl"], current_price - 1.5 * atr)
    elif trade["type"] == "sell":
        if entry - current_price >= risk:
            trade["sl"] = min(sl, entry)
        if atr > 0:
            trade["sl"] = min(trade["sl"], current_price + 1.5 * atr)
    return trade


def get_adaptive_threshold():
    """Compute adaptive threshold from recent trade history WITHOUT mutating globals."""
    if len(_trade_history) < 10:
        return BASE_CONFIDENCE
    wins = sum(1 for t in _trade_history if t.get("result") == "win")
    rate = wins / len(_trade_history)
    if rate < 0.4:
        return min(85, BASE_CONFIDENCE + 5)
    elif rate > 0.6:
        return max(55, BASE_CONFIDENCE - 5)
    return BASE_CONFIDENCE


def record_trade_result(signal, result):
    _trade_history.append({**signal, "result": result})
    if len(_trade_history) > 50:
        _trade_history.pop(0)

# HTF BIAS (4H resampling)

def get_htf_bias(df):
    df2 = df.copy()
    df2["datetime"] = pd.to_datetime(df2["time"], unit="s", utc=True)
    df2 = df2.set_index("datetime")
    h4  = df2[["open","high","low","close"]].resample("4h").agg({
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

# MARKET STRUCTURE

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
        if lows[i]  == min(lows[i  - lookback:i + lookback + 1]):
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


def detect_structure_breaks(swing_highs, swing_lows):
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

# LIQUIDITY & FVG

def detect_liquidity(df, threshold=0.0005):
    highs     = df["high"].values
    lows      = df["low"].values
    avg_price = float(df["close"].mean())
    scaled    = threshold * max(1.0, avg_price / 2.0)
    liq       = []
    for i in range(1, len(df)):
        if abs(highs[i] - highs[i - 1]) < scaled:
            liq.append({"type": "equal_highs", "idx": i, "level": highs[i]})
        if abs(lows[i]  - lows[i - 1])  < scaled:
            liq.append({"type": "equal_lows",  "idx": i, "level": lows[i]})
    return liq


def detect_fvg(df):
    fvgs = []
    for i in range(2, len(df) - 1):
        if df["low"].iloc[i] > df["high"].iloc[i - 2]:
            fvgs.append({"idx": i, "time": int(df["time"].iloc[i - 1]),
                         "top": float(df["low"].iloc[i]),
                         "bottom": float(df["high"].iloc[i - 2]), "type": "bullish"})
        elif df["high"].iloc[i] < df["low"].iloc[i - 2]:
            fvgs.append({"idx": i, "time": int(df["time"].iloc[i - 1]),
                         "top": float(df["low"].iloc[i - 2]),
                         "bottom": float(df["high"].iloc[i]), "type": "bearish"})
    return fvgs

# ORDER BLOCKS

def detect_order_blocks(df, lookback=10):
    atr    = compute_atr(df)
    o      = df["open"].values
    c      = df["close"].values
    h      = df["high"].values
    l      = df["low"].values
    t      = df["time"].values
    bull, bear = [], []

    for i in range(lookback, len(df) - 1):
        body   = abs(c[i] - o[i])
        thresh = 1.1 * atr.iloc[i]
        if body <= thresh:
            continue

        direction_bull = c[i] > o[i]
        for j in range(i - 1, max(i - lookback, 0) - 1, -1):
            if (direction_bull and c[j] < o[j]) or (not direction_bull and c[j] > o[j]):
                ob_top = float(max(o[j], c[j]))
                ob_bot = float(min(o[j], c[j]))
                entry = {
                    "time": int(t[j]), "top": round(ob_top, 5),
                    "bottom": round(ob_bot, 5),
                    "type": "bullish" if direction_bull else "bearish",
                    "high": round(float(h[j]), 5), "low": round(float(l[j]), 5),
                    "atr": round(float(atr.iloc[j]), 5), "signal_idx": i,
                }
                (bull if direction_bull else bear).append(entry)
                break

    seen, unique_bull = set(), []
    for ob in bull:
        if ob["time"] not in seen:
            seen.add(ob["time"]); unique_bull.append(ob)
    seen, unique_bear = set(), []
    for ob in bear:
        if ob["time"] not in seen:
            seen.add(ob["time"]); unique_bear.append(ob)

    return unique_bull, unique_bear

# CONFIDENCE SCORE

def compute_confidence(signal, trend, rsi_val, htf_bias):
    score = 0
    if signal["type"] == "buy" and trend == "bullish" and htf_bias == "bullish":
        score += 40
    elif signal["type"] == "sell" and trend == "bearish" and htf_bias == "bearish":
        score += 40
    elif (signal["type"] == "buy" and trend == "bullish") or (signal["type"] == "sell" and trend == "bearish"):
        score += 20
    if signal["type"] == "buy":
        if 55 < rsi_val <= 70:
            score += 30
        elif 45 <= rsi_val <= 55:
            score += 15
    elif signal["type"] == "sell":
        if 30 <= rsi_val < 45:
            score += 30
        elif 45 <= rsi_val <= 55:
            score += 15
    if signal.get("rr", 0) >= 2:
        score += 30
    return min(score, 100)

# SIGNAL ENGINE

def detect_entry_signals(df, atr_series, htf_bias_map, confidence_threshold=60, for_display=True):
    closes = df["close"]
    ema200 = compute_ema(closes, 200)
    rsi    = compute_rsi(closes, 14)

    bull_obs, bear_obs = detect_order_blocks(df)
    all_obs            = bull_obs + bear_obs

    swing_h, swing_l = detect_swings(df)
    structure = detect_structure_breaks(swing_h, swing_l)
    liquidity = detect_liquidity(df)
    fvgs      = detect_fvg(df)
    times     = df["time"].values
    signals   = []

    # Precompute time -> row index once (O(n)) instead of scanning the whole
    # DataFrame for every (i, order_block) pair (was O(n^2 * m)).
    time_to_idx = {int(t): idx for idx, t in enumerate(times)}

    for i in range(200, len(df)):
        ts = int(times[i])
        if not is_trading_session(ts):
            continue

        price   = float(closes.iloc[i])
        e200    = float(ema200.iloc[i])
        rsi_val = float(rsi.iloc[i])
        atr_val = float(atr_series.iloc[i])

        candle_date  = pd.Timestamp(ts, unit="s").date()
        htf          = htf_bias_map.get(candle_date, "neutral")

        has_bull_bos = any(b["type"] == "bullish_bos" and i - 100 < b["idx"] < i for b in structure)
        has_bear_bos = any(b["type"] == "bearish_bos" and i - 100 < b["idx"] < i for b in structure)
        has_liq      = any(i - 40 < l["idx"] < i for l in liquidity)
        has_bull_fvg = any(i - 40 < f["idx"] < i for f in fvgs if f["type"] == "bullish")
        has_bear_fvg = any(i - 40 < f["idx"] < i for f in fvgs if f["type"] == "bearish")

        candidates = []

        for ob in all_obs:
            ob_idx = time_to_idx.get(ob["time"])
            if ob_idx is None or ob_idx >= i:
                continue
            in_zone = ob["bottom"] <= price <= ob["top"]
            near_zone = ob["bottom"] - atr_val <= price <= ob["top"] + atr_val

            if ob["type"] == "bullish" and (in_zone or near_zone) and price > e200:
                score = 20
                if has_bull_bos: score += 25
                if has_liq:      score += 20
                if has_bull_fvg: score += 15
                if 45 <= rsi_val <= 70: score += 15
                if score >= 30:
                    sl  = round(ob["bottom"] - atr_val * 1.5, 5)
                    tp  = round(price + (price - sl) * 2.0, 5)
                    sig = {"time": ts, "type": "buy", "price": round(price, 5),
                           "rsi": round(rsi_val, 1), "sl": sl, "tp": tp, "rr": 2.0,
                           "atr": round(atr_val, 5), "htf": htf,
                           "lot": lot_size(round(price - sl, 5))}
                    sig["confidence"] = compute_confidence(sig, "bullish", rsi_val, htf)
                    try:
                        from ml_model import predict_win_probability
                        sig["ml_prob"] = predict_win_probability(sig, df, i)
                    except Exception:
                        sig["ml_prob"] = sig["confidence"] / 100.0
                    if sig["confidence"] >= confidence_threshold:
                        candidates.append(sig)

            elif ob["type"] == "bearish" and (in_zone or near_zone) and price < e200:
                score = 20
                if has_bear_bos: score += 25
                if has_liq:      score += 20
                if has_bear_fvg: score += 15
                if 30 <= rsi_val <= 55: score += 15
                if score >= 30:
                    sl  = round(ob["top"] + atr_val * 1.5, 5)
                    tp  = round(price - (sl - price) * 2.0, 5)
                    sig = {"time": ts, "type": "sell", "price": round(price, 5),
                           "rsi": round(rsi_val, 1), "sl": sl, "tp": tp, "rr": 2.0,
                           "atr": round(atr_val, 5), "htf": htf,
                           "lot": lot_size(round(sl - price, 5))}
                    sig["confidence"] = compute_confidence(sig, "bearish", rsi_val, htf)
                    try:
                        from ml_model import predict_win_probability
                        sig["ml_prob"] = predict_win_probability(sig, df, i)
                    except Exception:
                        sig["ml_prob"] = sig["confidence"] / 100.0
                    if sig["confidence"] >= confidence_threshold:
                        candidates.append(sig)

        if candidates:
            best = max(candidates, key=lambda x: x["confidence"])
            signals.append(best)

    deduped, last_ts = [], 0
    gap = 5 * 3600 if for_display else 3 * 3600
    for s in sorted(signals, key=lambda x: x["time"]):
        if s["time"] - last_ts > gap:
            deduped.append(s)
            last_ts = s["time"]

    return deduped[-10:] if for_display else deduped

# MT5 INTEGRATION

def mt5_connect():
    if not MT5_AVAILABLE:
        print("[MT5] MetaTrader5 package not available on this platform")
        return False
    if mt5.initialize():
        info = mt5.account_info()
        if info and info.login > 0:
            print("[MT5] Connected to existing session — " + info.server + " | Balance: $" + str(round(info.balance, 2)))
            return True
        mt5.shutdown()
    if mt5.initialize(login=VANTAGE_LOGIN, password=VANTAGE_PASSWORD, server=VANTAGE_SERVER):
        print("[MT5] Connected to " + VANTAGE_SERVER)
        return True
    print("[MT5] Login failed: " + str(mt5.last_error()))
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
        print("[MT5] fetch error: " + str(e))
        return []


def execute_trade_mt5(signal, symbol="XAUUSD", lot=None):
    if not MT5_AVAILABLE:
        print("[MT5] Not available on this platform")
        return None
    try:
        if not mt5_connect():
            return None
        lot  = lot or signal.get("lot", 0.01)
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
            "comment":      "TradeView | " + str(signal.get("confidence", "—")) + "%",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        mt5.shutdown()
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print("[MT5] OK " + signal["type"].upper() + " " + symbol + " @ " + str(round(price, 5)) + " | Lot " + str(lot))
        else:
            print("[MT5] FAIL " + str(result.comment if result else mt5.last_error()))
        return result
    except Exception as e:
        print("[MT5] Error: " + str(e))
        try:
            mt5.shutdown()
        except:
            pass
        return None

# AI NARRATIVE

def generate_ai_analysis(df, signals):
    if len(df) < 50:
        return "Not enough data for analysis."
    close      = df["close"]
    ema50      = compute_ema(close, 50)
    ema200     = compute_ema(close, 200)
    last_price = float(close.iloc[-1])
    trend      = "bullish" if ema50.iloc[-1] > ema200.iloc[-1] else "bearish" if ema50.iloc[-1] < ema200.iloc[-1] else "sideways"
    rsi_val    = float(compute_rsi(close).iloc[-1])
    atr_val    = float(compute_atr(df).iloc[-1])
    last_sig   = signals[-1]["type"] if signals else "none"
    momentum   = "strong bullish" if rsi_val > 60 else "strong bearish" if rsi_val < 40 else "weak / ranging"
    return ("Market Analysis:" + chr(10) +
            "- Price: " + str(round(last_price, 5)) + chr(10) +
            "- Trend: " + trend + chr(10) +
            "- Momentum: " + momentum + " (RSI " + str(round(rsi_val, 1)) + ")" + chr(10) +
            "- Volatility (ATR): " + str(round(atr_val, 5)) + chr(10) +
            "Structure: " + trend + " with " + momentum + " momentum. Last signal: " + last_sig + "." + chr(10) +
            "Strategy: Trade with dominant trend. Avoid counter-trend entries " +
            "unless strong reversal confluence appears.")


def generate_summary(bias, htf_bias, last_rsi, last_close, ema20, ema50, ema200,
                     order_blocks, signals, symbol, timeframe):
    sym_label = {"EURUSD": "EUR/USD", "XAUUSD": "XAU/USD (Gold)",
                 "USDJPY": "USD/JPY"}.get(symbol, symbol)
    trend = "Above EMA200 — bullish." if last_close > ema200 else "Below EMA200 — bearish."
    if last_close > ema50 > ema200:   trend += " EMAs stacked bullishly."
    elif last_close < ema50 < ema200: trend += " EMAs stacked bearishly."
    else:                              trend += " EMAs mixed."
    trend += " 4H HTF: " + htf_bias.upper() + "."
    if last_rsi > 70:   rsi_desc = "RSI " + str(round(last_rsi, 1)) + " — overbought."
    elif last_rsi < 30: rsi_desc = "RSI " + str(round(last_rsi, 1)) + " — oversold."
    elif last_rsi > 55: rsi_desc = "RSI " + str(round(last_rsi, 1)) + " — bullish momentum."
    elif last_rsi < 45: rsi_desc = "RSI " + str(round(last_rsi, 1)) + " — bearish momentum."
    else:               rsi_desc = "RSI " + str(round(last_rsi, 1)) + " — neutral."
    bull_obs = [o for o in order_blocks if o["type"] == "bullish"]
    bear_obs = [o for o in order_blocks if o["type"] == "bearish"]
    ob_desc  = str(len(bull_obs)) + " bullish OB(s), " + str(len(bear_obs)) + " bearish OB(s)."
    if bull_obs: ob_desc += " Demand: " + str(bull_obs[-1]["bottom"]) + "–" + str(bull_obs[-1]["top"]) + "."
    if bear_obs: ob_desc += " Supply: " + str(bear_obs[-1]["bottom"]) + "–" + str(bear_obs[-1]["top"]) + "."
    if signals:
        s = signals[-1]
        sig_desc = (s["type"].upper() + " @ " + str(s["price"]) + " · "
                    "SL " + str(s["sl"]) + " · TP " + str(s["tp"]) + " · 1:" + str(s["rr"]) + " · "
                    + str(s.get("confidence","—")) + "% conf.")
    else:
        sig_desc = "No confirmed signals. Waiting for confluence."
    if bias == "bullish" and htf_bias == "bullish":
        rec = "BULLISH — HTF confirmed. Buy from bullish OBs."
    elif bias == "bearish" and htf_bias == "bearish":
        rec = "BEARISH — HTF confirmed. Sell from bearish OBs."
    elif htf_bias == "neutral":
        rec = "HTF NEUTRAL — wait for directional bias."
    else:
        rec = "LTF (" + bias + ") conflicts with HTF (" + htf_bias + "). Wait for alignment."
    return {"symbol": sym_label, "timeframe": timeframe, "trend": trend,
            "rsi_desc": rsi_desc, "ob_desc": ob_desc, "sig_desc": sig_desc,
            "rec": rec, "bias": bias, "htf_bias": htf_bias}

# MAIN RUNNER

def run_analysis(candles, symbol="EURUSD", timeframe="1h"):
    df = pd.DataFrame(candles)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = df["time"].astype(int)
    df = df.sort_values("time").reset_index(drop=True)
    closes = df["close"]
    times  = df["time"].values
    atr    = compute_atr(df)
    # LOCAL threshold — computed fresh every time, never mutated globally
    base_threshold   = 60 if symbol == "XAUUSD" else 65
    vol_factor       = 5 if atr.iloc[-1] > atr.mean() * 1.3 else 0
    adaptive_offset  = get_adaptive_threshold() - BASE_CONFIDENCE
    confidence_threshold = max(55, min(80, base_threshold + vol_factor + adaptive_offset))
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
    signals            = detect_entry_signals(df, atr, htf_bias_map, confidence_threshold=confidence_threshold, for_display=True)
    last_close = float(closes.iloc[-1])
    last_e200  = float(ema200.iloc[-1])
    last_e50   = float(ema50.iloc[-1])
    last_rsi   = float(rsi.iloc[-1])
    bias = "bullish" if last_close > last_e200 and last_close > last_e50 else "bearish" if last_close < last_e200 and last_close < last_e50 else "neutral"
    last_date = pd.Timestamp(int(times[-1]), unit="s").date()
    htf_bias  = htf_bias_map.get(last_date, "neutral")
    summary     = generate_summary(bias, htf_bias, last_rsi, last_close,
                                   float(ema20.iloc[-1]), float(ema50.iloc[-1]), last_e200,
                                   display_obs, signals, symbol, timeframe)
    ai_analysis = generate_ai_analysis(df, signals)
    if signals:
        best = max(signals, key=lambda x: x.get("confidence", 0))
        print("[Signal] " + best["type"].upper() + " " + symbol + " @ " + str(best["price"]) + " | "
              "Conf: " + str(best.get("confidence",0)) + "% | ML: " + str(best.get("ml_prob","—")))
    print("[Analysis] " + symbol + " " + timeframe + " | " + str(len(signals)) + " signals | "
          "Bias: " + bias + " | HTF: " + htf_bias + " | Threshold: " + str(confidence_threshold))
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

# DEMO / STANDALONE RUN

if __name__ == "__main__":
    print("=" * 60)
    print("  SMC Trading Algorithm — Standalone Demo")
    print("=" * 60)
    candles = fetch_live_data_mt5(symbol="XAUUSD", n=500)
    if not candles:
        print("[MT5] No live data. Generating demo candles...")
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
    print(chr(10) + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    if "error" in result:
        print("ERROR: " + result["error"])
    else:
        print(chr(10) + "Bias: " + result["bias"] + " | HTF: " + result["htf_bias"])
        print(chr(10) + result["ai_analysis"])
        print(chr(10) + result["summary"]["rec"])
        if result["signals"]:
            print(chr(10) + "--- Signals (" + str(len(result["signals"])) + ") ---")
            for s in result["signals"]:
                print("  " + s["type"].upper() + " @ " + str(s["price"]) + " | SL: " + str(s["sl"]) + " | TP: " + str(s["tp"]) + " | " + "RR: 1:" + str(s["rr"]) + " | Conf: " + str(s["confidence"]) + "%")
        else:
            print(chr(10) + "No signals generated.")
        print(chr(10) + "--- Order Blocks (" + str(len(result["order_blocks"])) + ") ---")
        for ob in result["order_blocks"][-5:]:
            print("  " + ob["type"].upper() + " OB: " + str(ob["bottom"]) + "–" + str(ob["top"]))