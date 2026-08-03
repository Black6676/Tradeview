"""
TradeView MT5 Executor — runs separately from the Flask app.
Polls your local TradeView app for signals and executes on MT5.
Must run on the MAIN THREAD — do not import into Flask.

Usage (open a second Command Prompt):
    python mt5_executor.py

Keep this running alongside python app.py
"""
import time
import json
import os
import requests
import MetaTrader5 as mt5
from datetime import datetime, timezone

# ── Config ─────────────────────────────────────────────────
APP_URL            = "http://localhost:8080"
SCAN_INTERVAL      = 15 * 60
MIN_CONFIDENCE     = 70        # execute signals with 70%+ confidence
PIP_SL_TP          = 10        # fixed ±10 pip SL/TP
MAX_DAILY_LOSS_PCT = 0.04      # 4% daily loss limit
MAX_TOTAL_LOSS_PCT = 0.10      # 10% total drawdown limit
RISK_PER_TRADE     = 0.01      # 1% risk per trade for lot sizing

_start_balance = None
_daily_pnl     = 0.0
_daily_date    = None

SCAN_TARGETS = [
    {"symbol": "EURUSD", "tf": "1h"},
    {"symbol": "EURUSD", "tf": "4h"},
    {"symbol": "XAUUSD", "tf": "1h"},
    {"symbol": "USDJPY", "tf": "1h"},
]

EXECUTED_FILE = "mt5_executed.json"


# ── Persistence ─────────────────────────────────────────────
def load_executed():
    if os.path.exists(EXECUTED_FILE):
        with open(EXECUTED_FILE) as f:
            return set(json.load(f))
    return set()


def save_executed(executed):
    with open(EXECUTED_FILE, "w") as f:
        json.dump(list(executed)[-500:], f)


# ── MT5 ─────────────────────────────────────────────────────
def connect_mt5():
    """Connect to already-running MT5 terminal (main thread safe)."""
    if mt5.initialize():
        info = mt5.account_info()
        if info and info.login > 0:
            print(f"[MT5] ✓ Session: {info.login} | {info.server} | ${info.balance:.2f}")
            return True
        mt5.shutdown()
    print(f"[MT5] ✗ Could not connect: {mt5.last_error()}")
    print("      Make sure MT5 is open and logged in with Algo Trading enabled.")
    return False


def pip_dist(symbol):
    pip = 0.10 if symbol == "XAUUSD" else 0.001 if symbol == "USDJPY" else 0.0001
    return PIP_SL_TP * pip


def get_account_balance():
    info = mt5.account_info()
    return info.balance if info else 0.0


def check_risk_limits():
    global _start_balance, _daily_pnl, _daily_date
    from datetime import date
    today = date.today()
    balance = get_account_balance()
    if _start_balance is None:
        _start_balance = balance
    if _daily_date != today:
        _daily_pnl  = 0.0
        _daily_date = today
    peak = max(_start_balance, balance)
    if abs(_daily_pnl) >= peak * MAX_DAILY_LOSS_PCT and _daily_pnl < 0:
        return False, f"Daily loss limit: ${abs(_daily_pnl):.2f} / ${peak * MAX_DAILY_LOSS_PCT:.2f}"
    if (peak - balance) >= peak * MAX_TOTAL_LOSS_PCT:
        return False, f"Max drawdown: ${peak - balance:.2f} / ${peak * MAX_TOTAL_LOSS_PCT:.2f}"
    return True, "OK"


def dynamic_lot(sl_distance):
    """1% risk-based lot sizing using live balance."""
    balance = get_account_balance()
    risk    = balance * RISK_PER_TRADE
    if sl_distance <= 0:
        return 0.01
    return round(max(risk / (sl_distance * 100), 0.01), 2)


def place_order(signal, symbol):
    # Risk limit check
    can_trade, reason = check_risk_limits()
    if not can_trade:
        print(f"[Risk] Trade blocked — {reason}")
        return False

    tick     = mt5.symbol_info_tick(symbol)
    sym_info = mt5.symbol_info(symbol)
    if not tick or not sym_info:
        print(f"[MT5] Cannot get symbol info for {symbol}")
        return False

    if not sym_info.visible:
        mt5.symbol_select(symbol, True)

    digits = sym_info.digits
    dist   = pip_dist(symbol)

    # Enforce broker minimum stop distance
    stop_level = getattr(sym_info, "trade_stops_level", 0) or 0
    point      = getattr(sym_info, "point", 0.00001) or 0.00001
    min_dist   = max((stop_level + 10) * point, dist)

    if signal["type"] == "buy":
        price      = tick.ask
        order_type = mt5.ORDER_TYPE_BUY
        sl         = round(price - min_dist, digits)
        tp         = round(price + min_dist, digits)
    else:
        price      = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
        sl         = round(price + min_dist, digits)
        tp         = round(price - min_dist, digits)

    # Calculate lot size based on actual SL distance
    sl_distance = abs(signal["price"] - signal.get("sl", signal["price"]))
    lot = dynamic_lot(sl_distance)
    print(f"[Risk] Lot: {lot} | SL dist: {sl_distance:.5f} | Risk: 1% of balance")

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
        "comment":      f"TradeView | Conf:{signal.get('confidence',0)}%",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[MT5] ✓ {signal['type'].upper()} {symbol} @ {price:.5f} "
              f"| SL:{sl} TP:{tp} | Ticket:{result.order}")
        return True
    else:
        code = result.retcode if result else "?"
        msg  = result.comment if result else str(mt5.last_error())
        print(f"[MT5] ✗ Failed: {code} — {msg}")
        return False


# ── Signal fetching ─────────────────────────────────────────
def fetch_signals(symbol, tf):
    try:
        url  = f"{APP_URL}/api/ohlcv?symbol={symbol}&timeframe={tf}"
        resp = requests.get(url, timeout=30)
        data = resp.json()
        return data.get("signals", [])
    except Exception as e:
        print(f"[Fetch] Error {symbol} {tf}: {e}")
        return []


# ── Main scan ───────────────────────────────────────────────
def run_scan(executed):
    now = datetime.now(tz=timezone.utc).strftime("%H:%M:%S UTC")
    print(f"\n[Scan] Starting at {now}")

    if not connect_mt5():
        print("[Scan] MT5 unavailable — skipping execution this cycle")
        mt5.shutdown()
        return

    executed_this_scan = 0

    for target in SCAN_TARGETS:
        symbol = target["symbol"]
        tf     = target["tf"]
        signals = fetch_signals(symbol, tf)

        if not signals:
            print(f"  {symbol} {tf}: no signals")
            continue

        best = max(signals, key=lambda x: x.get("confidence", 0))
        conf = best.get("confidence", 0)
        key  = f"{symbol}_{tf}_{best['time']}_{best['type']}"

        print(f"  {symbol} {tf}: {best['type'].upper()} @ {best['price']} | Conf:{conf}%")

        if conf < MIN_CONFIDENCE:
            print(f"    → Skip (conf {conf}% < {MIN_CONFIDENCE}%)")
            continue

        if key in executed:
            print(f"    → Already executed")
            continue

        # Check risk limits before each trade
        can_trade, reason = check_risk_limits()
        if not can_trade:
            print(f"    → BLOCKED: {reason}")
            continue

        bal_before = get_account_balance()
        success = place_order(best, symbol)
        if success:
            executed.add(key)
            save_executed(executed)
            executed_this_scan += 1
            # Update daily P&L
            global _daily_pnl
            bal_after = get_account_balance()
            _daily_pnl += (bal_after - bal_before)
            print(f"    → Daily P&L so far: ${_daily_pnl:.2f}")

        time.sleep(1)

    mt5.shutdown()
    print(f"[Scan] Complete — {executed_this_scan} trade(s) placed")
    return executed


# ── Entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("TradeView MT5 Executor")
    print(f"App URL      : {APP_URL}")
    print(f"Min Confidence: {MIN_CONFIDENCE}%")
    print(f"Lot Size     : {LOT_SIZE}")
    print(f"SL/TP        : ±{PIP_SL_TP} pips")
    print(f"Scan Interval: every {SCAN_INTERVAL // 60} minutes")
    print("=" * 50)
    print("\nMake sure:")
    print("  1. python app.py is running in another terminal")
    print("  2. MT5 is open and logged in")
    print("  3. Algo Trading button is GREEN in MT5\n")

    executed = load_executed()

    while True:
        try:
            run_scan(executed)
        except Exception as e:
            print(f"[Error] {e}")
        print(f"[Sleep] Next scan in {SCAN_INTERVAL // 60} minutes...\n")
        time.sleep(SCAN_INTERVAL)
