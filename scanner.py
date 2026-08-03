"""
Background scanner — checks all instruments every 15 minutes.
Runs as a thread inside Flask. Executes MT5 trades on Windows (local only).
"""
import os
import threading
import time
import requests as req
import pandas as pd
from datetime import datetime
from algorithm import run_analysis
from alerts import process_signals

API_KEY  = "0a603f27b4664a98bfb3d6bac2af9f9b"
BASE_URL = "https://api.twelvedata.com"

SCAN_TARGETS = [
    {"symbol": "EURUSD", "td_symbol": "EUR/USD", "timeframe": "1h",  "interval": "1h",   "outputsize": 500},
    {"symbol": "EURUSD", "td_symbol": "EUR/USD", "timeframe": "4h",  "interval": "4h",   "outputsize": 300},
    {"symbol": "XAUUSD", "td_symbol": "XAU/USD", "timeframe": "1h",  "interval": "1h",   "outputsize": 500},
    {"symbol": "USDJPY", "td_symbol": "USD/JPY", "timeframe": "1h",  "interval": "1h",   "outputsize": 500},
]

SCAN_INTERVAL  = 15 * 60
IS_RENDER      = os.environ.get("RENDER", False)

# Shared state
latest_alerts  = []
scanner_status = {"last_scan": None, "next_scan": None, "running": False}
email_config   = {"enabled": False}
_lock          = threading.Lock()
_executed_keys = set()   # tracks signals already executed
_EXECUTED_FILE = "scanner_executed.json"

def _load_executed():
    global _executed_keys
    import json, os
    if os.path.exists(_EXECUTED_FILE):
        try:
            with open(_EXECUTED_FILE) as f:
                _executed_keys = set(json.load(f))
        except Exception:
            _executed_keys = set()

def _save_executed():
    import json
    with open(_EXECUTED_FILE, "w") as f:
        json.dump(list(_executed_keys)[-500:], f)

_load_executed()  # load on import


def fetch_candles(td_symbol, interval, outputsize):
    params = {
        "symbol":     td_symbol,
        "interval":   interval,
        "outputsize": outputsize,
        "apikey":     API_KEY,
        "format":     "JSON",
    }
    resp = req.get(f"{BASE_URL}/time_series", params=params, timeout=15)
    data = resp.json()
    if "values" not in data:
        return []
    candles = []
    for bar in data["values"]:
        candles.append({
            "time":   int(pd.Timestamp(bar["datetime"]).timestamp()),
            "open":   round(float(bar["open"]),  5),
            "high":   round(float(bar["high"]),  5),
            "low":    round(float(bar["low"]),   5),
            "close":  round(float(bar["close"]), 5),
            "volume": round(float(bar.get("volume", 0)), 2),
        })
    candles.sort(key=lambda x: x["time"])
    return candles


def try_execute_mt5(signal, symbol):
    """Execute trade via MT5 — only on Windows local, never on Render."""
    if IS_RENDER:
        return
    pip  = 0.10 if symbol == "XAUUSD" else 0.001 if symbol == "USDJPY" else 0.0001
    dist = 10 * pip
    entry = signal["price"]
    if signal["type"] == "buy":
        signal["sl"] = round(entry - dist, 5)
        signal["tp"] = round(entry + dist, 5)
    else:
        signal["sl"] = round(entry + dist, 5)
        signal["tp"] = round(entry - dist, 5)
    print(f"[Scanner-MT5] {signal['type'].upper()} {symbol} @ {entry} "
          f"| SL:{signal['sl']} TP:{signal['tp']} | Conf:100%")
    try:
        from algorithm import execute_trade_mt5
        execute_trade_mt5(signal, symbol=symbol, lot=signal.get("lot", 0.01))
    except Exception as e:
        print(f"[Scanner-MT5] Error: {e}")


def run_scan():
    global latest_alerts
    print(f"[Scanner] Running scan at {datetime.utcnow().strftime('%H:%M:%S UTC')}")
    new_alerts = []

    for target in SCAN_TARGETS:
        try:
            candles = fetch_candles(target["td_symbol"], target["interval"], target["outputsize"])
            if not candles:
                continue

            analysis = run_analysis(candles, symbol=target["symbol"], timeframe=target["timeframe"])
            signals  = analysis.get("signals", [])

            if not signals:
                continue

            best    = max(signals, key=lambda x: x.get("confidence", 0))
            conf    = best.get("confidence", 0)
            sig_key = (best["time"], best["type"], target["symbol"], target["timeframe"])

            # ── MT5 execution (local Windows only) ────────────
            # Risk rule: execute if confidence >= 70% (not hardcoded 100)
            # Risk limits (4% daily, 10% total) checked inside execute_trade_mt5
            if conf >= 70 and sig_key not in _executed_keys and not IS_RENDER:
                _executed_keys.add(sig_key)
                if len(_executed_keys) > 200:
                    _executed_keys.pop()
                _save_executed()
                try_execute_mt5(dict(best), target["symbol"])

            # ── Alert system ───────────────────────────────────
            cfg   = email_config if email_config.get("enabled") else None
            fresh = process_signals([best], target["symbol"], target["timeframe"], cfg)
            new_alerts.extend(fresh)

            time.sleep(2)

        except Exception as e:
            print(f"[Scanner] Error scanning {target['symbol']} {target['timeframe']}: {e}")

    with _lock:
        existing_keys = {
            (a.get("symbol"), a.get("timeframe"), a.get("time"), a.get("type"))
            for a in latest_alerts
        }
        truly_new = [
            a for a in new_alerts
            if (a.get("symbol"), a.get("timeframe"), a.get("time"), a.get("type"))
            not in existing_keys
        ]
        latest_alerts = truly_new + latest_alerts
        latest_alerts = latest_alerts[:50]

    now = datetime.utcnow()
    scanner_status["last_scan"] = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    scanner_status["next_scan"] = f"in {SCAN_INTERVAL // 60} minutes"
    print(f"[Scanner] Scan complete. {len(truly_new)} new signal(s) found.")


def scanner_loop():
    scanner_status["running"] = True
    while True:
        try:
            run_scan()
        except Exception as e:
            print(f"[Scanner] Unhandled error: {e}")
        time.sleep(SCAN_INTERVAL)


def start_scanner():
    t = threading.Thread(target=scanner_loop, daemon=True)
    t.start()
    print("[Scanner] Background scanner started — checking every 15 minutes.")
