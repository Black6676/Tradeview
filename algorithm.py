import pandas as pd
import numpy as np
from datetime import datetime, timezone
import MetaTrader5 as mt5

try:
    from scipy.signal import argrelextrema
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ========================== GLOBAL SETTINGS ==========================
ACCOUNT_BALANCE = 1000
RISK_PER_TRADE = 0.01
CONFIDENCE_THRESHOLD = 60

# Vantage MT5 Credentials (CHANGE THESE)
VANTAGE_LOGIN = 24786681          # e.g. 12345678
VANTAGE_PASSWORD = "Black@123"
VANTAGE_SERVER = "VantageInternational-Demo"   # Change to your exact server name

_trade_history = []

# ========================== INDICATORS ==========================
def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def compute_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

# ========================== RISK & SESSION ==========================
def is_trading_session(ts):
    hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour
    return 7 <= hour <= 20

def lot_size(sl_distance):
    risk = ACCOUNT_BALANCE * RISK_PER_TRADE
    if sl_distance <= 0:
        return 0.01
    lot = risk / (sl_distance * 100)
    return round(max(lot, 0.01), 2)

def apply_trade_management(trade, current_price, atr):
    entry = trade["entry_price"]
    sl = trade["sl"]
    tp = trade["tp"]
    risk = abs(entry - sl)

    if trade["type"] == "buy":
        if current_price - entry >= risk:
            trade["sl"] = max(sl, entry)
        trade["sl"] = max(trade["sl"], current_price - 1.5 * atr)
    else:
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
        CONFIDENCE_THRESHOLD = min(85, CONFIDENCE_THRESHOLD + 5)
    elif rate > 0.6:
        CONFIDENCE_THRESHOLD = max(55, CONFIDENCE_THRESHOLD - 5)

def record_trade_result(signal, result):
    _trade_history.append({**signal, "result": result})
    if len(_trade_history) > 50:
        _trade_history.pop(0)
    adapt_strategy()

# ========================== HTF BIAS ==========================
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
        e20 = h4_ema20.loc[dt]
        e50 = h4_ema50.loc[dt]
        bias = "bullish" if close > e20 > e50 else "bearish" if close < e20 < e50 else "neutral"
        for h in range(4):
            key = (dt + pd.Timedelta(hours=h)).date()
            bias_map[key] = bias
    return bias_map

# ========================== STRUCTURE & DETECTION ==========================
def detect_swings(df, lookback=5):
    highs = df["high"].values
    lows = df["low"].values
    if SCIPY_AVAILABLE:
        try:
            order = max(2, lookback)
            sh_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
            sl_idx = argrelextrema(lows, np.less_equal, order=order)[0]
            swing_highs = [(i, highs[i]) for i in sh_idx if lookback <= i < len(df) - lookback]
            swing_lows = [(i, lows[i]) for i in sl_idx if lookback <= i < len(df) - lookback]
            return swing_highs, swing_lows
        except:
            pass
    swing_highs, swing_lows = [], []
    for i in range(lookback, len(df) - lookback):
        if highs[i] == max(highs[i-lookback:i+lookback+1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i-lookback:i+lookback+1]):
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows

def detect_structure_breaks(swing_highs, swing_lows):
    breaks = []
    for i in range(1, len(swing_highs)):
        if swing_highs[i][1] > swing_highs[i-1][1]:
            breaks.append({"type": "bullish_bos", "idx": swing_highs[i][0], "level": swing_highs[i][1]})
    for i in range(1, len(swing_lows)):
        if swing_lows[i][1] < swing_lows[i-1][1]:
            breaks.append({"type": "bearish_bos", "idx": swing_lows[i][0], "level": swing_lows[i][1]})
    return breaks

def detect_liquidity(df, threshold=0.0005):
    highs = df["high"].values
    lows = df["low"].values
    avg_price = float(df["close"].mean())
    scaled = threshold * max(1.0, avg_price / 2.0)
    liq = []
    for i in range(1, len(df)):
        if abs(highs[i] - highs[i-1]) < scaled:
            liq.append({"type": "equal_highs", "idx": i, "level": highs[i]})
        if abs(lows[i] - lows[i-1]) < scaled:
            liq.append({"type": "equal_lows", "idx": i, "level": lows[i]})
    return liq

def detect_fvg(df):
    fvgs = []
    for i in range(2, len(df)-1):
        if df["low"].iloc[i] > df["high"].iloc[i-2]:
            fvgs.append({"idx": i, "type": "bullish"})
        elif df["high"].iloc[i] < df["low"].iloc[i-2]:
            fvgs.append({"idx": i, "type": "bearish"})
    return fvgs

def detect_order_blocks(df, lookback=10):
    atr = compute_atr(df)
    o, c, h, l, t = df["open"].values, df["close"].values, df["high"].values, df["low"].values, df["time"].values
    bull, bear = [], []
    for i in range(lookback, len(df)-1):
        body = abs(c[i] - o[i])
        thresh = 1.1 * atr.iloc[i]
        if body > thresh:
            direction_bull = c[i] > o[i]
            for j in range(i-1, max(i-lookback,0)-1, -1):
                if (direction_bull and c[j] < o[j]) or (not direction_bull and c[j] > o[j]):
                    ob_top = max(o[j], c[j])
                    ob_bot = min(o[j], c[j])
                    mitigated = any(ob_bot <= p <= ob_top for p in c[j+1:i])
                    if mitigated or (i - j <= 8):
                        obs_list = bull if direction_bull else bear
                        obs_list.append({
                            "time": int(t[j]), "top": round(ob_top,5), "bottom": round(ob_bot,5),
                            "type": "bullish" if direction_bull else "bearish",
                            "atr": round(float(atr.iloc[j]),5), "signal_idx": i
                        })
                    break
    seen = set()
    unique_bull = [ob for ob in bull if not (ob["time"] in seen or seen.add(ob["time"]))]
    unique_bear = [ob for ob in bear if not (ob["time"] in seen or seen.add(ob["time"]))]
    return unique_bull, unique_bear

# ========================== SIGNAL ENGINE ==========================
def detect_entry_signals(df, atr_series, htf_bias_map):
    closes = df["close"]
    ema200 = compute_ema(closes, 200)
    rsi = compute_rsi(closes, 14)
    bull_obs, bear_obs = detect_order_blocks(df)
    all_obs = bull_obs + bear_obs
    swing_h, swing_l = detect_swings(df)
    structure = detect_structure_breaks(swing_h, swing_l)
    liquidity = detect_liquidity(df)
    fvgs = detect_fvg(df)
    times = df["time"].values
    signals = []

    for i in range(200, len(df)):
        if not is_trading_session(int(times[i])):
            continue
        price = float(closes.iloc[i])
        ts = int(times[i])
        e200 = float(ema200.iloc[i])
        rsi_val = float(rsi.iloc[i])
        atr_val = float(atr_series.iloc[i])

        candle_date = pd.Timestamp(ts, unit="s").date()
        htf = htf_bias_map.get(candle_date, "neutral")

        has_bull_bos = any(b["type"] == "bullish_bos" and i - 100 < b["idx"] < i for b in structure)
        has_bear_bos = any(b["type"] == "bearish_bos" and i - 100 < b["idx"] < i for b in structure)
        has_liq = any(i - 40 < l["idx"] < i for l in liquidity)
        has_bull_fvg = any(i - 40 < f["idx"] < i for f in fvgs if f["type"] == "bullish")
        has_bear_fvg = any(i - 40 < f["idx"] < i for f in fvgs if f["type"] == "bearish")

        for ob in all_obs:
            if df[df["time"] == ob["time"]].empty or df[df["time"] == ob["time"]].index[0] >= i:
                continue
            in_zone = ob["bottom"] <= price <= ob["top"]

            if ob["type"] == "bullish" and htf == "bullish" and in_zone and price > e200:
                score = 20 + (25 if has_bull_bos else 0) + (20 if has_liq else 0) + (15 if has_bull_fvg else 0) + (15 if 45 <= rsi_val <= 70 else 0)
                if score >= 30:
                    sl = round(ob["bottom"] - atr_val * 1.5, 5)
                    tp = round(price + (price - sl) * 2.0, 5)
                    sig = {"time": ts, "type": "buy", "price": round(price,5),
                           "sl": sl, "tp": tp, "rr": 2.0, "atr": round(atr_val,5),
                           "htf": htf, "lot": lot_size(price - sl)}
                    sig["confidence"] = 40 if htf == "bullish" else 20
                    if sig["confidence"] >= CONFIDENCE_THRESHOLD:
                        signals.append(sig)
                        break

            elif ob["type"] == "bearish" and htf == "bearish" and in_zone and price < e200:
                score = 20 + (25 if has_bear_bos else 0) + (20 if has_liq else 0) + (15 if has_bear_fvg else 0) + (15 if 30 <= rsi_val <= 55 else 0)
                if score >= 30:
                    sl = round(ob["top"] + atr_val * 1.5, 5)
                    tp = round(price - (sl - price) * 2.0, 5)
                    sig = {"time": ts, "type": "sell", "price": round(price,5),
                           "sl": sl, "tp": tp, "rr": 2.0, "atr": round(atr_val,5),
                           "htf": htf, "lot": lot_size(sl - price)}
                    sig["confidence"] = 40 if htf == "bearish" else 20
                    if sig["confidence"] >= CONFIDENCE_THRESHOLD:
                        signals.append(sig)
                        break
    return signals

# ========================== MT5 INTEGRATION (Vantage) ==========================
def mt5_connect():
    if VANTAGE_LOGIN and VANTAGE_PASSWORD and VANTAGE_SERVER:
        if not mt5.initialize(login=VANTAGE_LOGIN, password=VANTAGE_PASSWORD, server=VANTAGE_SERVER):
            print(f"[MT5] Login failed: {mt5.last_error()}")
            return False
    else:
        if not mt5.initialize():
            print(f"[MT5] Init failed: {mt5.last_error()}")
            return False
    print(f"[MT5] Connected to {VANTAGE_SERVER or 'terminal'}")
    return True

def fetch_live_data_mt5(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H1, n=500):
    if not mt5_connect():
        return []
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        print("[MT5] No data received")
        return []
    return [{"time": int(r["time"]), "open": float(r["open"]), "high": float(r["high"]),
             "low": float(r["low"]), "close": float(r["close"]), "volume": float(r["tick_volume"])} for r in rates]

def execute_trade_mt5(signal, symbol="XAUUSD", lot=None):
    if not mt5_connect():
        return None
    if lot is None:
        lot = signal.get("lot", 0.01)
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        mt5.shutdown()
        return None
    price = tick.ask if signal["type"] == "buy" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal["type"] == "buy" else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": signal["sl"],
        "tp": signal["tp"],
        "deviation": 30,
        "magic": 123456,
        "comment": f"TradeView | Conf {signal.get('confidence',0)}%",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    mt5.shutdown()
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[MT5] ✓ SUCCESS: {signal['type'].upper()} {symbol} @ {price:.5f} | Lot {lot}")
    else:
        print(f"[MT5] ✗ FAILED: {result.comment if result else 'No result'}")
    return result

# ========================== LIVE RUNNER ==========================
def run_analysis(candles, symbol="XAUUSD", timeframe="1h"):
    global CONFIDENCE_THRESHOLD
    df = pd.DataFrame(candles)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = df["time"].astype(int)
    df = df.sort_values("time").reset_index(drop=True)

    atr = compute_atr(df)
    htf_bias_map = get_htf_bias(df)

    # Dynamic threshold
    base = 60 if symbol == "XAUUSD" else 65
    vol_factor = 5 if atr.iloc[-1] > atr.mean() * 1.3 else 0
    CONFIDENCE_THRESHOLD = max(55, min(80, base + vol_factor))

    signals = detect_entry_signals(df, atr, htf_bias_map)

    if signals:
        best = max(signals, key=lambda x: x.get("confidence", 0))
        if best.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
            print(f"[Live] Strong signal detected — executing on Vantage")
            execute_trade_mt5(best, symbol=symbol)

    print(f"[Live] {len(signals)} signals generated | Current Conf Threshold: {CONFIDENCE_THRESHOLD}")
    return {"signals": signals}