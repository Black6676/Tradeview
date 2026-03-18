"""
Background scanner — checks all instruments and timeframes every 15 minutes.
Runs as a separate thread inside the Flask app.
"""
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

SCAN_INTERVAL = 15 * 60   # 15 minutes in seconds

# Shared state — Flask reads this to serve /api/alerts
latest_alerts  = []
scanner_status = {"last_scan": None, "next_scan": None, "running": False}
email_config   = {"enabled": False}
_lock          = threading.Lock()


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
            if signals:
                cfg = email_config if email_config.get("enabled") else None
                fresh = process_signals(signals, target["symbol"], target["timeframe"], cfg)
                new_alerts.extend(fresh)
            # Small delay between API calls to respect rate limits
            time.sleep(2)
        except Exception as e:
            print(f"[Scanner] Error scanning {target['symbol']} {target['timeframe']}: {e}")

    with _lock:
        latest_alerts = new_alerts + latest_alerts
        latest_alerts = latest_alerts[:50]   # keep last 50

    now = datetime.utcnow()
    scanner_status["last_scan"] = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    scanner_status["next_scan"] = f"in {SCAN_INTERVAL // 60} minutes"
    print(f"[Scanner] Scan complete. {len(new_alerts)} new signal(s) found.")


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
