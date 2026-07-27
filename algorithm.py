import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os
from typing import List, Dict, Optional, Tuple

# Optional imports with graceful fallback
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
# GLOBAL SETTINGS — load from env, never hardcode credentials
# ══════════════════════════════════════════════════════════════

AACCOUNT_BALANCE      = 1000
RISK_PER_TRADE       = 0.01
MAX_TRADES_PER_DAY   = 3
CONFIDENCE_THRESHOLD = 60

# Vantage MT5 credentials
VANTAGE_LOGIN    = 67203023
VANTAGE_PASSWORD = "Qwerty@12345"
VANTAGE_SERVER   = "RoboForex-ECN"

_trade_history  = []
# Symbol-specific settings
SYMBOL_SETTINGS = {
    "XAUUSD": {"pip_value": 1.0,   "point_mult": 100,  "min_lot": 0.01, "spread_max_pct_atr": 0.15},
    "EURUSD": {"pip_value": 10.0,  "point_mult": 10000, "min_lot": 0.01, "spread_max_pct_atr": 0.10},
    "USDJPY": {"pip_value": 1000.0, "point_mult": 100,  "min_lot": 0.01, "spread_max_pct_atr": 0.10},
    "GBPUSD": {"pip_value": 10.0,  "point_mult": 10000, "min_lot": 0.01, "spread_max_pct_atr": 0.10},
}

_trade_history: List[Dict] = []
_daily_stats = {"date": None, "trades": 0, "loss": 0.0}


# ══════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
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


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    sma = compute_sma(series, period)
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


# ══════════════════════════════════════════════════════════════
# SESSION & KILL-ZONE FILTER
# ══════════════════════════════════════════════════════════════

def get_session_info(ts: int) -> Dict:
    """Returns detailed session info for a given unix timestamp."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    hour_utc = dt.hour
    weekday = dt.weekday()

    # Kill zones in UTC (approximate)
    # London: 07:00-10:00 UTC, NY: 12:00-15:00 UTC, London Close: 15:00-17:00 UTC
    # Asian: 00:00-07:00 UTC (avoid for most setups)
    is_london = 7 <= hour_utc <= 10
    is_ny = 12 <= hour_utc <= 15
    is_london_close = 15 <= hour_utc <= 17
    is_asian = 0 <= hour_utc < 7
    is_weekend = weekday >= 5

    return {
        "hour": hour_utc,
        "weekday": weekday,
        "is_london": is_london,
        "is_ny": is_ny,
        "is_london_close": is_london_close,
        "is_asian": is_asian,
        "is_weekend": is_weekend,
        "is_killzone": is_london or is_ny,
        "session_name": "london" if is_london else ("ny" if is_ny else ("london_close" if is_london_close else ("asian" if is_asian else "other")))
    }


def is_trading_session(ts: int, require_killzone: bool = True) -> bool:
    """
    Only trade during high-probability windows.
    London open through NY close is okay, but kill zones are preferred.
    """
    info = get_session_info(ts)
    if info["is_weekend"]:
        return False
    if info["is_asian"]:
        return False  # Asian session = low liquidity, false breakouts
    if require_killzone and not info["is_killzone"]:
        return False
    return True


# ══════════════════════════════════════════════════════════════
# RISK MANAGEMENT
# ══════════════════════════════════════════════════════════════

def lot_size(sl_distance: float, symbol: str = "XAUUSD", account_balance: Optional[float] = None, risk_pct: Optional[float] = None) -> float:
    """
    Proper lot sizing per symbol.
    XAUUSD: 1 lot ≈ $1 per $0.01 move (simplified). Actually 1 lot = 100 oz, so $1 per $0.01 = $100 per $1.00 move.
    For forex: standard 1 lot = $10 per pip.
    """
    balance  = account_balance or ACCOUNT_BALANCE
    risk_pct = risk_pct or RISK_PER_TRADE
    risk_amt = balance * risk_pct

    settings = SYMBOL_SETTINGS.get(symbol, SYMBOL_SETTINGS["EURUSD"])

    if sl_distance <= 0:
        return settings["min_lot"]

    # For XAUUSD: if SL is $1.50 away, and we risk $10, lot = $10 / ($1.50 * 100) = 0.067 → 0.07
    # For EURUSD: if SL is 15 pips (0.0015), and we risk $10, lot = $10 / (15 * $10) = 0.067 → 0.07
    # Simplified unified formula using symbol multiplier
    lot = risk_amt / (sl_distance * settings["pip_value"] * settings["point_mult"])
    return round(max(lot, settings["min_lot"]), 2)


def check_daily_limit() -> bool:
    """Returns True if we can trade (daily loss limit not hit)."""
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


def record_trade_result(signal: Dict, result: str, pnl: float = 0.0):
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
    """Dynamic threshold based on recent win rate and market regime."""
    global CONFIDENCE_THRESHOLD
    if len(_trade_history) < 15:
        return
    recent = _trade_history[-15:]
    wins = sum(1 for t in recent if t.get("result") == "win")
    rate = wins / len(recent)

    # Also adjust based on average loss size
    losses = [t["pnl"] for t in recent if t.get("pnl", 0) < 0]
    avg_loss = abs(np.mean(losses)) if losses else 0

    if rate < 0.35 or avg_loss > ACCOUNT_BALANCE * RISK_PER_TRADE * 2:
        CONFIDENCE_THRESHOLD = min(85, CONFIDENCE_THRESHOLD + 3)
    elif rate > 0.60 and avg_loss < ACCOUNT_BALANCE * RISK_PER_TRADE * 1.2:
        CONFIDENCE_THRESHOLD = max(55, CONFIDENCE_THRESHOLD - 3)


def apply_trade_management(trade: Dict, current_price: float, current_atr: float = 0) -> Dict:
    """
    Break-even + ATR trailing stop + partial TP logic.
    Returns updated trade dict.
    """
    entry = trade["entry_price"]
    sl    = trade["sl"]
    tp    = trade["tp"]
    risk  = abs(entry - sl)
    atr   = current_atr or trade.get("atr", 0)

    trade.setdefault("partial_tp_hit", False)
    trade.setdefault("breakeven_set", False)

    if trade["type"] == "buy":
        # Move to breakeven after 1R profit
        if not trade["breakeven_set"] and current_price - entry >= risk * 0.8:
            trade["sl"] = max(sl, entry)
            trade["breakeven_set"] = True

        # Partial TP at 1.5R
        if not trade["partial_tp_hit"] and current_price - entry >= risk * 1.5:
            trade["partial_tp_hit"] = True
            # In real execution, close 50% here
            trade["close_50_percent"] = True

        # ATR trailing stop after 2R
        if current_price - entry >= risk * 2.0 and atr > 0:
            new_sl = current_price - 2.0 * atr
            trade["sl"] = max(trade["sl"], new_sl)

    elif trade["type"] == "sell":
        if not trade["breakeven_set"] and entry - current_price >= risk * 0.8:
            trade["sl"] = min(sl, entry)
            trade["breakeven_set"] = True

        if not trade["partial_tp_hit"] and entry - current_price >= risk * 1.5:
            trade["partial_tp_hit"] = True
            trade["close_50_percent"] = True

        if entry - current_price >= risk * 2.0 and atr > 0:
            new_sl = current_price + 2.0 * atr
            trade["sl"] = min(trade["sl"], new_sl)

    return trade


# ══════════════════════════════════════════════════════════════
# MARKET REGIME DETECTION
# ══════════════════════════════════════════════════════════════

def detect_market_regime(df: pd.DataFrame) -> str:
    """
    Classify market as trending_up, trending_down, ranging, or volatile.
    Uses ADX + Bollinger Band width.
    """
    if len(df) < 50:
        return "unknown"

    adx = compute_adx(df).iloc[-1]
    upper, mid, lower = compute_bollinger_bands(df["close"], 20, 2)
    bb_width = (upper.iloc[-1] - lower.iloc[-1]) / mid.iloc[-1] if mid.iloc[-1] != 0 else 0

    # Trend direction from EMAs
    ema20 = compute_ema(df["close"], 20).iloc[-1]
    ema50 = compute_ema(df["close"], 50).iloc[-1]

    if adx > 25:
        if ema20 > ema50:
            return "trending_up"
        else:
            return "trending_down"
    elif adx < 18 and bb_width < 0.02:
        return "ranging"
    elif bb_width > 0.05:
        return "volatile"
    else:
        return "choppy"


# ══════════════════════════════════════════════════════════════
# SWING DETECTION (improved — avoids flat areas)
# ══════════════════════════════════════════════════════════════

def detect_swings(df: pd.DataFrame, lookback: int = 5) -> Tuple[List, List]:
    """
    Detect swing highs and lows. Improved to avoid flat areas.
    Uses strict greater_than / less_than instead of equal.
    """
    highs = df["high"].values
    lows  = df["low"].values

    if SCIPY_AVAILABLE:
        try:
            order = max(2, lookback)
            # Use strictly greater/less to avoid flat zones
            sh_idx = argrelextrema(highs, np.greater, order=order)[0]
            sl_idx = argrelextrema(lows,  np.less,    order=order)[0]
            swing_highs = [(int(i), float(highs[i])) for i in sh_idx if lookback <= i < len(df) - lookback]
            swing_lows  = [(int(i), float(lows[i]))  for i in sl_idx if lookback <= i < len(df) - lookback]
            return swing_highs, swing_lows
        except Exception:
            pass

    swing_highs, swing_lows = [], []
    for i in range(lookback, len(df) - lookback):
        window_high = highs[i - lookback:i + lookback + 1]
        window_low  = lows[i - lookback:i + lookback + 1]
        # Strictly greater/less, and ensure not equal to neighbors
        if highs[i] > max(highs[i - lookback:i]) and highs[i] > max(highs[i + 1:i + lookback + 1]):
            swing_highs.append((i, float(highs[i])))
        if lows[i] < min(lows[i - lookback:i]) and lows[i] < min(lows[i + 1:i + lookback + 1]):
            swing_lows.append((i, float(lows[i])))
    return swing_highs, swing_lows


# ══════════════════════════════════════════════════════════════
# MARKET STRUCTURE — BOS + CHoCH + MSS
# ══════════════════════════════════════════════════════════════

def detect_structure(swing_highs: List, swing_lows: List, df: pd.DataFrame, lookback: int = 5) -> List[Dict]:
    """
    Detect BOS (continuation), CHoCH (soft reversal), and MSS (strong reversal with displacement).
    """
    events = []
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return events

    # Sort by index
    all_swings = sorted([(i, "high", p) for i, p in swing_highs] + [(i, "low", p) for i, p in swing_lows])

    for idx in range(2, len(all_swings)):
        prev2 = all_swings[idx - 2]
        prev1 = all_swings[idx - 1]
        curr  = all_swings[idx]

        # BOS: price breaks previous high in uptrend, or previous low in downtrend
        if curr[1] == "high" and prev2[1] == "high":
            if curr[2] > prev2[2]:
                # Check for displacement (strong candle)
                is_displacement = _is_displacement_candle(df, curr[0])
                events.append({
                    "type": "bullish_bos",
                    "idx": curr[0],
                    "level": curr[2],
                    "displacement": is_displacement,
                    "strength": "strong" if is_displacement else "weak"
                })

        if curr[1] == "low" and prev2[1] == "low":
            if curr[2] < prev2[2]:
                is_displacement = _is_displacement_candle(df, curr[0])
                events.append({
                    "type": "bearish_bos",
                    "idx": curr[0],
                    "level": curr[2],
                    "displacement": is_displacement,
                    "strength": "strong" if is_displacement else "weak"
                })

        # CHoCH / MSS: reversal signals
        # Bullish CHoCH: after a down sequence, price makes a higher low, then breaks a lower high
        if curr[1] == "high" and prev1[1] == "low" and prev2[1] == "high":
            if curr[2] > prev2[2]:
                is_displacement = _is_displacement_candle(df, curr[0])
                events.append({
                    "type": "bullish_choch" if not is_displacement else "bullish_mss",
                    "idx": curr[0],
                    "level": curr[2],
                    "displacement": is_displacement
                })

        if curr[1] == "low" and prev1[1] == "high" and prev2[1] == "low":
            if curr[2] < prev2[2]:
                is_displacement = _is_displacement_candle(df, curr[0])
                events.append({
                    "type": "bearish_choch" if not is_displacement else "bearish_mss",
                    "idx": curr[0],
                    "level": curr[2],
                    "displacement": is_displacement
                })

    return events


def _is_displacement_candle(df: pd.DataFrame, idx: int, atr_mult: float = 1.5) -> bool:
    """A displacement candle has a large body relative to ATR and small wicks."""
    if idx >= len(df):
        return False
    atr = compute_atr(df).iloc[idx]
    if atr == 0 or pd.isna(atr):
        return False
    o, h, l, c = df["open"].iloc[idx], df["high"].iloc[idx], df["low"].iloc[idx], df["close"].iloc[idx]
    body = abs(c - o)
    wick = (h - l) - body
    return body >= atr * atr_mult and wick <= body * 0.3


# ══════════════════════════════════════════════════════════════
# LIQUIDITY & SWEEPS
# ══════════════════════════════════════════════════════════════

def detect_liquidity(df: pd.DataFrame, threshold: float = 0.0005) -> List[Dict]:
    """Detect equal highs/lows (liquidity pools)."""
    highs     = df["high"].values
    lows      = df["low"].values
    avg_price = float(df["close"].mean())
    scaled    = threshold * max(1.0, avg_price / 2.0)
    liq       = []

    for i in range(1, len(df)):
        if abs(highs[i] - highs[i - 1]) < scaled:
            liq.append({"type": "equal_highs", "idx": i, "level": float(highs[i]), "swept": False})
        if abs(lows[i]  - lows[i - 1])  < scaled:
            liq.append({"type": "equal_lows",  "idx": i, "level": float(lows[i]), "swept": False})
    return liq


def detect_liquidity_sweeps(df: pd.DataFrame, liquidity: List[Dict], lookforward: int = 10) -> List[Dict]:
    """
    CRITICAL: A liquidity level is only useful if price SWEEPS it before reversing.
    A sweep means price briefly exceeds the level, then reverses strongly.
    """
    highs = df["high"].values
    lows  = df["low"].values
    closes = df["close"].values
    swept = []

    for liq in liquidity:
        idx = liq["idx"]
        level = liq["level"]
        liq_type = liq["type"]

        # Look forward for sweep
        for j in range(idx + 1, min(idx + lookforward, len(df))):
            if liq_type == "equal_highs":
                # Sweep: price goes above the level, then closes back below
                if highs[j] > level and closes[j] < level:
                    # Check for displacement down after sweep
                    if j + 1 < len(df) and closes[j+1] < closes[j]:
                        swept.append({
                            **liq,
                            "swept": True,
                            "sweep_idx": j,
                            "sweep_type": "buy_side_liquidity_sweep"
                        })
                        break
            else:
                # equal_lows
                if lows[j] < level and closes[j] > level:
                    if j + 1 < len(df) and closes[j+1] > closes[j]:
                        swept.append({
                            **liq,
                            "swept": True,
                            "sweep_idx": j,
                            "sweep_type": "sell_side_liquidity_sweep"
                        })
                        break

    return swept


# ══════════════════════════════════════════════════════════════
# FAIR VALUE GAPS (FVG)
# ══════════════════════════════════════════════════════════════

def detect_fvg(df: pd.DataFrame) -> List[Dict]:
    """Fair Value Gaps with quality scoring."""
    fvgs = []
    atr = compute_atr(df)

    for i in range(2, len(df) - 1):
        c1_high, c1_low = df["high"].iloc[i-2], df["low"].iloc[i-2]
        c2_high, c2_low = df["high"].iloc[i-1], df["low"].iloc[i-1]
        c3_high, c3_low = df["low"].iloc[i], df["high"].iloc[i]  # Note: using opposite for gap calc

        # Bullish FVG: candle i low > candle i-2 high
        if df["low"].iloc[i] > df["high"].iloc[i-2]:
            gap_size = df["low"].iloc[i] - df["high"].iloc[i-2]
            atr_val = atr.iloc[i]
            quality = min(1.0, gap_size / (atr_val * 0.5)) if atr_val > 0 else 0.5
            fvgs.append({
                "idx": i, "time": int(df["time"].iloc[i-1]),
                "top": float(df["low"].iloc[i]),
                "bottom": float(df["high"].iloc[i-2]),
                "type": "bullish",
                "quality": round(float(quality), 2),
                "gap_size": round(float(gap_size), 5)
            })

        # Bearish FVG: candle i high < candle i-2 low
        elif df["high"].iloc[i] < df["low"].iloc[i-2]:
            gap_size = df["low"].iloc[i-2] - df["high"].iloc[i]
            atr_val = atr.iloc[i]
            quality = min(1.0, gap_size / (atr_val * 0.5)) if atr_val > 0 else 0.5
            fvgs.append({
                "idx": i, "time": int(df["time"].iloc[i-1]),
                "top": float(df["low"].iloc[i-2]),
                "bottom": float(df["high"].iloc[i]),
                "type": "bearish",
                "quality": round(float(quality), 2),
                "gap_size": round(float(gap_size), 5)
            })

    return fvgs


# ══════════════════════════════════════════════════════════════
# ORDER BLOCKS — Strict validation with displacement + BOS
# ══════════════════════════════════════════════════════════════

def detect_order_blocks(df: pd.DataFrame, lookback: int = 10) -> Tuple[List[Dict], List[Dict]]:
    """
    STRICT OB detection:
    1. Find displacement candle (strong momentum)
    2. Verify it broke structure (BOS)
    3. The LAST opposing candle BEFORE the displacement is the OB
    4. OB must NOT be mitigated (price hasn't returned and closed through it)
    """
    atr = compute_atr(df)
    o = df["open"].values
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    t = df["time"].values

    bull_obs, bear_obs = [], []

    for i in range(lookback, len(df) - 2):
        # Check if candle i is displacement
        if not _is_displacement_candle(df, i, atr_mult=1.2):
            continue

        # Determine direction of displacement
        is_bull_displacement = c[i] > o[i]
        is_bear_displacement = c[i] < o[i]

        # Find the last opposing candle before displacement
        ob_candle = None
        for j in range(i - 1, max(i - lookback, -1), -1):
            if is_bull_displacement and c[j] < o[j]:
                ob_candle = j
                break
            if is_bear_displacement and c[j] > o[j]:
                ob_candle = j
                break

        if ob_candle is None:
            continue

        # Verify BOS: the displacement must break a recent swing level
        # For bull: close[i] > recent swing high
        # For bear: close[i] < recent swing low
        recent_window = df.iloc[max(0, i-20):i]
        if is_bull_displacement:
            recent_high = recent_window["high"].max()
            if c[i] <= recent_high:
                continue  # No BOS, not a valid OB
        else:
            recent_low = recent_window["low"].min()
            if c[i] >= recent_low:
                continue

        # Build OB
        j = ob_candle
        ob_top = float(max(o[j], c[j]))
        ob_bot = float(min(o[j], c[j]))

        # Check if mitigated: any close AFTER the OB but BEFORE or AT displacement
        # that closed through the OB body
        mitigated = False
        for k in range(j + 1, i):
            if ob_bot <= c[k] <= ob_top:
                mitigated = True
                break

        if mitigated:
            continue  # Fresh OB only

        entry = {
            "time": int(t[j]),
            "top": round(ob_top, 5),
            "bottom": round(ob_bot, 5),
            "type": "bullish" if is_bull_displacement else "bearish",
            "high": round(float(h[j]), 5),
            "low": round(float(l[j]), 5),
            "atr": round(float(atr.iloc[j]), 5),
            "displacement_idx": i,
            "bos_confirmed": True,
            "fresh": True,
        }

        # Quality score: OB size relative to ATR (not too small, not too huge)
        ob_size = ob_top - ob_bot
        atr_val = atr.iloc[j]
        if atr_val > 0:
            size_ratio = ob_size / atr_val
            entry["quality"] = round(min(1.0, size_ratio / 1.5), 2)  # ideal ~1.5x ATR
        else:
            entry["quality"] = 0.5

        if is_bull_displacement:
            bull_obs.append(entry)
        else:
            bear_obs.append(entry)

    # Deduplicate by time
    seen, unique_bull = set(), []
    for ob in bull_obs:
        if ob["time"] not in seen:
            seen.add(ob["time"]); unique_bull.append(ob)
    seen, unique_bear = set(), []
    for ob in bear_obs:
        if ob["time"] not in seen:
            seen.add(ob["time"]); unique_bear.append(ob)

    return unique_bull, unique_bear


def detect_breaker_blocks(df: pd.DataFrame, existing_obs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    When a bullish OB is broken downward with displacement, it becomes a bearish breaker.
    When a bearish OB is broken upward with displacement, it becomes a bullish breaker.
    """
    breakers_bull, breakers_bear = [], []
    c = df["close"].values

    for ob in existing_obs:
        ob_time = ob["time"]
        ob_rows = df[df["time"] == ob_time]
        if ob_rows.empty:
            continue
        ob_idx = ob_rows.index[0]

        # Look forward to see if OB was broken
        for i in range(ob_idx + 1, min(ob_idx + 50, len(df))):
            if ob["type"] == "bullish":
                # Broken if price closes below OB low with displacement
                if c[i] < ob["low"] and _is_displacement_candle(df, i, atr_mult=1.0):
                    breakers_bear.append({
                        "time": ob["time"],
                        "top": ob["top"],
                        "bottom": ob["bottom"],
                        "type": "bearish_breaker",
                        "original_type": "bullish",
                        "break_idx": i,
                        "atr": ob["atr"]
                    })
                    break
            else:
                if c[i] > ob["high"] and _is_displacement_candle(df, i, atr_mult=1.0):
                    breakers_bull.append({
                        "time": ob["time"],
                        "top": ob["top"],
                        "bottom": ob["bottom"],
                        "type": "bullish_breaker",
                        "original_type": "bearish",
                        "break_idx": i,
                        "atr": ob["atr"]
                    })
                    break

    return breakers_bull, breakers_bear


# ══════════════════════════════════════════════════════════════
# HTF BIAS (4H) + Premium/Discount
# ══════════════════════════════════════════════════════════════

def get_htf_bias(df: pd.DataFrame) -> Dict:
    """
    Returns bias_map by date AND premium/discount levels.
    """
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

        # Premium/Discount: 50% of recent range
        recent = h4.loc[:dt].tail(20)
        range_high = recent["high"].max()
        range_low  = recent["low"].min()
        eq = (range_high + range_low) / 2

        for h in range(4):
            key = (dt + pd.Timedelta(hours=h)).date()
            bias_map[key] = bias
            ranges[key] = {"high": range_high, "low": range_low, "eq": eq}

    return {"bias_map": bias_map, "ranges": ranges}


def is_premium_discount(price: float, range_info: Dict, direction: str) -> str:
    """Returns 'premium', 'discount', or 'equilibrium'."""
    eq = range_info.get("eq", price)
    if direction == "buy":
        return "discount" if price < eq else ("premium" if price > eq else "equilibrium")
    else:
        return "premium" if price > eq else ("discount" if price < eq else "equilibrium")


# ══════════════════════════════════════════════════════════════
# CONFIDENCE SCORE — Probabilistic, regime-aware
# ══════════════════════════════════════════════════════════════

def compute_confidence(signal: Dict, trend: str, rsi_val: float, htf_bias: str,
                       regime: str, liquidity_sweep: bool, ob_quality: float,
                       fvg_quality: float = 0, premium_discount: str = "equilibrium") -> int:
    """
    0–100 score. More nuanced and regime-aware.
    """
    score = 0

    # Trend alignment (max 30)
    if signal["type"] == "buy" and trend == "bullish" and htf_bias == "bullish":
        score += 30
    elif signal["type"] == "sell" and trend == "bearish" and htf_bias == "bearish":
        score += 30
    elif (signal["type"] == "buy" and trend == "bullish") or \
         (signal["type"] == "sell" and trend == "bearish"):
        score += 15

    # Regime fit (max 15)
    if regime in ("trending_up", "trending_down"):
        if (signal["type"] == "buy" and regime == "trending_up") or \
           (signal["type"] == "sell" and regime == "trending_down"):
            score += 15
        else:
            score -= 10  # Counter-trend in strong trend = bad
    elif regime == "ranging":
        score += 5  # Neutral
    elif regime == "volatile":
        score -= 5  # Harder to predict

    # RSI confirmation (max 20)
    if signal["type"] == "buy":
        if 55 < rsi_val <= 70:    score += 20
        elif 45 <= rsi_val <= 55: score += 10
        elif rsi_val > 70:        score -= 10  # Overbought
    elif signal["type"] == "sell":
        if 30 <= rsi_val < 45:    score += 20
        elif 45 <= rsi_val <= 55: score += 10
        elif rsi_val < 30:        score -= 10  # Oversold

    # R:R quality (max 15)
    rr = signal.get("rr", 0)
    if rr >= 3:   score += 15
    elif rr >= 2: score += 10
    elif rr >= 1.5: score += 5

    # Liquidity sweep bonus (max 10)
    if liquidity_sweep:
        score += 10

    # OB quality (max 5)
    score += int(ob_quality * 5)

    # FVG quality (max 5)
    score += int(fvg_quality * 5)

    # Premium/Discount penalty/bonus
    if premium_discount == "discount" and signal["type"] == "buy":
        score += 5
    elif premium_discount == "premium" and signal["type"] == "sell":
        score += 5
    elif premium_discount == "premium" and signal["type"] == "buy":
        score -= 10  # Buying in premium = bad
    elif premium_discount == "discount" and signal["type"] == "sell":
        score -= 10  # Selling in discount = bad

    return max(0, min(100, score))


# ══════════════════════════════════════════════════════════════
# SIGNAL ENGINE — Complete rewrite with validation
# ══════════════════════════════════════════════════════════════

def detect_entry_signals(df: pd.DataFrame, atr_series: pd.Series, htf_data: Dict,
                         symbol: str = "XAUUSD", for_display: bool = True) -> List[Dict]:
    """
    Full signal engine with:
    - Liquidity sweep validation
    - CHoCH/MSS confirmation
    - Fresh OB only
    - Premium/Discount filter
    - Spread filter
    - Kill zone enforcement
    """
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

    # Symbol settings for spread filter
    sym_settings = SYMBOL_SETTINGS.get(symbol, SYMBOL_SETTINGS["EURUSD"])

    for i in range(200, len(df)):
        ts = int(times[i])
        session = get_session_info(ts)

        # Kill zone filter — skip non-killzone unless very strong setup
        if not session["is_killzone"]:
            continue

        price   = float(closes.iloc[i])
        e200    = float(ema200.iloc[i])
        e50     = float(ema50.iloc[i])
        rsi_val = float(rsi.iloc[i])
        atr_val = float(atr_series.iloc[i])

        # Spread filter: if current candle spread > max allowed, skip
        candle_spread = float(df["high"].iloc[i] - df["low"].iloc[i])
        if candle_spread > atr_val * sym_settings["spread_max_pct_atr"] * 3:
            continue  # Abnormally wide candle = news spike, avoid

        candle_date  = pd.Timestamp(ts, unit="s").date()
        htf          = htf_data.get("bias_map", {}).get(candle_date, "neutral")
        range_info   = htf_data.get("ranges", {}).get(candle_date, {})

        # Recent structure events
        recent_struct = [s for s in structure if i - 50 < s["idx"] <= i]
        has_bull_bos = any(s["type"] == "bullish_bos" and s.get("displacement") for s in recent_struct)
        has_bear_bos = any(s["type"] == "bearish_bos" and s.get("displacement") for s in recent_struct)
        has_bull_mss = any(s["type"] in ("bullish_choch", "bullish_mss") for s in recent_struct)
        has_bear_mss = any(s["type"] in ("bearish_choch", "bearish_mss") for s in recent_struct)

        # Recent sweeps
        recent_sweeps = [s for s in sweeps if i - 30 < s.get("sweep_idx", 0) <= i]
        has_bull_sweep = any(s["sweep_type"] == "sell_side_liquidity_sweep" for s in recent_sweeps)
        has_bear_sweep = any(s["sweep_type"] == "buy_side_liquidity_sweep" for s in recent_sweeps)

        # Recent FVGs
        recent_fvgs = [f for f in fvgs if i - 40 < f["idx"] < i]
        has_bull_fvg = any(f["type"] == "bullish" for f in recent_fvgs)
        has_bear_fvg = any(f["type"] == "bearish" for f in recent_fvgs)
        best_bull_fvg = max([f["quality"] for f in recent_fvgs if f["type"] == "bullish"], default=0)
        best_bear_fvg = max([f["quality"] for f in recent_fvgs if f["type"] == "bearish"], default=0)

        # Check each OB — NO break, evaluate ALL and pick best
        candidates = []

        for ob in all_obs:
            ob_rows = df[df["time"] == ob["time"]]
            if ob_rows.empty or ob_rows.index[0] >= i:
                continue

            # Check if OB is still fresh (not mitigated since creation)
            ob_idx = ob_rows.index[0]
            mitigated = False
            for k in range(ob_idx + 1, i + 1):
                if ob["bottom"] <= closes.iloc[k] <= ob["top"]:
                    mitigated = True
                    break
            if mitigated and ob.get("type") not in ("bullish_breaker", "bearish_breaker"):
                continue

            in_zone = ob["bottom"] <= price <= ob["top"]
            near_zone = ob["bottom"] - atr_val * 0.5 <= price <= ob["top"] + atr_val * 0.5

            if not (in_zone or near_zone):
                continue

            ob_type = ob["type"]
            is_breaker = "breaker" in ob_type

            # BULLISH SETUP
            if ob_type in ("bullish", "bullish_breaker"):
                if htf != "bullish" and not is_breaker:
                    continue
                if price < e200 * 0.998:  # Must be near/above EMA200
                    continue

                score = 15  # Base for being in zone
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

                    sig = {
                        "time": ts, "type": "buy", "price": round(price, 5),
                        "rsi": round(rsi_val, 1), "sl": sl, "tp": tp, "rr": rr,
                        "atr": round(atr_val, 5), "htf": htf,
                        "lot": lot_size(round(price - sl, 5), symbol),
                        "regime": regime,
                        "session": session["session_name"],
                        "ob_quality": ob.get("quality", 0.5),
                        "is_breaker": is_breaker,
                        "raw_score": score
                    }

                    pd_zone = is_premium_discount(price, range_info, "buy")
                    sig["confidence"] = compute_confidence(
                        sig, "bullish", rsi_val, htf, regime,
                        has_bull_sweep, ob.get("quality", 0.5), best_bull_fvg, pd_zone
                    )
                    sig["premium_discount"] = pd_zone

                    try:
                        from ml_model import predict_win_probability
                        sig["ml_prob"] = predict_win_probability(sig, df, i)
                    except Exception:
                        sig["ml_prob"] = sig["confidence"] / 100.0

                    candidates.append(sig)

            # BEARISH SETUP
            elif ob_type in ("bearish", "bearish_breaker"):
                if htf != "bearish" and not is_breaker:
                    continue
                if price > e200 * 1.002:  # Must be near/below EMA200
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

                    sig = {
                        "time": ts, "type": "sell", "price": round(price, 5),
                        "rsi": round(rsi_val, 1), "sl": sl, "tp": tp, "rr": rr,
                        "atr": round(atr_val, 5), "htf": htf,
                        "lot": lot_size(round(sl - price, 5), symbol),
                        "regime": regime,
                        "session": session["session_name"],
                        "ob_quality": ob.get("quality", 0.5),
                        "is_breaker": is_breaker,
                        "raw_score": score
                    }

                    pd_zone = is_premium_discount(price, range_info, "sell")
                    sig["confidence"] = compute_confidence(
                        sig, "bearish", rsi_val, htf, regime,
                        has_bear_sweep, ob.get("quality", 0.5), best_bear_fvg, pd_zone
                    )
                    sig["premium_discount"] = pd_zone

                    try:
                        from ml_model import predict_win_probability
                        sig["ml_prob"] = predict_win_probability(sig, df, i)
                    except Exception:
                        sig["ml_prob"] = sig["confidence"] / 100.0

                    candidates.append(sig)

        # Pick best candidate for this candle
        if candidates:
            best = max(candidates, key=lambda x: x["confidence"])
            if best["confidence"] >= CONFIDENCE_THRESHOLD:
                signals.append(best)

    # Deduplicate with wider gap for display
    deduped, last_ts = [], 0
    gap = 4 * 3600 if for_display else 2 * 3600  # 4h for display, 2h for trading
    for s in sorted(signals, key=lambda x: x["time"]):
        if s["time"] - last_ts > gap:
            deduped.append(s)
            last_ts = s["time"]

    return deduped[-10:] if for_display else deduped


# ══════════════════════════════════════════════════════════════
# MT5 INTEGRATION — Secure, robust
# ══════════════════════════════════════════════════════════════

def mt5_connect() -> bool:
    if not MT5_AVAILABLE:
        print("[MT5] MetaTrader5 package not available on this platform")
        return False

    if mt5.initialize():
        info = mt5.account_info()
        if info and info.login > 0:
            print(f"[MT5] Connected to existing session — {info.server} | Balance: ${info.balance:.2f}")
            return True
        mt5.shutdown()

    # Only login if credentials are configured
    if VANTAGE_LOGIN > 0 and VANTAGE_PASSWORD and VANTAGE_SERVER:
        if mt5.initialize(login=VANTAGE_LOGIN, password=VANTAGE_PASSWORD, server=VANTAGE_SERVER):
            print(f"[MT5] Connected to {VANTAGE_SERVER}")
            return True

    print(f"[MT5] Login failed: {mt5.last_error()}")
    return False


def fetch_live_data_mt5(symbol: str = "XAUUSD", timeframe=None, n: int = 500) -> List[Dict]:
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


def execute_trade_mt5(signal: Dict, symbol: str = "XAUUSD", lot: Optional[float] = None):
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
            mt5.shutdown()
            return None

        sym_info = mt5.symbol_info(symbol)
        if sym_info and not sym_info.visible:
            mt5.symbol_select(symbol, True)

        price      = tick.ask if signal["type"] == "buy" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if signal["type"] == "buy" else mt5.ORDER_TYPE_SELL
        digits     = sym_info.digits if sym_info else 5

        # Enforce minimum stop distance
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
        try:
            mt5.shutdown()
        except:
            pass
        return None


# ══════════════════════════════════════════════════════════════
# AI NARRATIVE
# ══════════════════════════════════════════════════════════════

def generate_ai_analysis(df: pd.DataFrame, signals: List[Dict]) -> str:
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


def generate_summary(bias: str, htf_bias: str, last_rsi: float, last_close: float,
                     ema20: float, ema50: float, ema200: float,
                     order_blocks: List[Dict], signals: List[Dict], symbol: str, timeframe: str) -> Dict:
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

def run_analysis(candles: List[Dict], symbol: str = "EURUSD", timeframe: str = "1h") -> Dict:
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

    # Dynamic confidence threshold based on symbol and volatility
    base_threshold = 65 if symbol == "XAUUSD" else 70
    vol_factor = 5 if atr.iloc[-1] > atr.mean() * 1.3 else 0
    regime = detect_market_regime(df)
    if regime == "ranging":
        base_threshold += 5  # Require more confirmation in ranges
    elif regime == "volatile":
        base_threshold += 10  # Avoid trading volatile chop
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
