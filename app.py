from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from algorithm import run_analysis
from backtest import run_backtest
from scanner import start_scanner, latest_alerts, scanner_status, email_config
import requests
import pandas as pd

app = Flask(__name__)
CORS(app)

API_KEY  = "0a603f27b4664a98bfb3d6bac2af9f9b"
BASE_URL = "https://api.twelvedata.com"

INSTRUMENTS = {
    "XAUUSD": {"symbol": "XAU/USD", "label": "XAU/USD (Gold)"},
    "USDJPY": {"symbol": "USD/JPY", "label": "USD/JPY"},
    "EURUSD": {"symbol": "EUR/USD", "label": "EUR/USD"},
}

TIMEFRAMES = {
    "15m": {"interval": "15min", "outputsize": 200},
    "1h":  {"interval": "1h",    "outputsize": 500},
    "4h":  {"interval": "4h",    "outputsize": 300},
    "1d":  {"interval": "1day",  "outputsize": 365},
}


# ── Pages ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/backtest")
def backtest_page():
    return render_template("backtest.html")

@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html")


# ── API: Live chart data ───────────────────────────────────────

@app.route("/api/ohlcv")
def get_ohlcv():
    symbol    = request.args.get("symbol", "EURUSD")
    timeframe = request.args.get("timeframe", "1h")
    if symbol not in INSTRUMENTS:
        return jsonify({"error": "Unknown symbol"}), 400
    if timeframe not in TIMEFRAMES:
        return jsonify({"error": "Unknown timeframe"}), 400
    instrument = INSTRUMENTS[symbol]
    tf         = TIMEFRAMES[timeframe]
    try:
        params = {"symbol": instrument["symbol"], "interval": tf["interval"],
                  "outputsize": tf["outputsize"], "apikey": API_KEY, "format": "JSON"}
        resp = requests.get(f"{BASE_URL}/time_series", params=params, timeout=10)
        data = resp.json()
        if "code" in data:
            return jsonify({"error": data.get("message", "Twelve Data error")}), 502
        if "values" not in data:
            return jsonify({"error": "No data returned from Twelve Data"}), 502
        candles = []
        for bar in data["values"]:
            candles.append({"time": int(pd.Timestamp(bar["datetime"]).timestamp()),
                "open": round(float(bar["open"]), 5), "high": round(float(bar["high"]), 5),
                "low": round(float(bar["low"]), 5), "close": round(float(bar["close"]), 5),
                "volume": round(float(bar.get("volume", 0)), 2)})
        candles.sort(key=lambda x: x["time"])
        analysis   = run_analysis(candles, symbol=symbol, timeframe=timeframe)
        last_price = candles[-1]["close"]
        change_pct = round((candles[-1]["close"] - candles[-2]["close"]) / candles[-2]["close"] * 100, 3) if len(candles) >= 2 else 0
        return jsonify({"meta": {"symbol": symbol, "label": instrument["label"], "timeframe": timeframe,
            "bars": len(candles), "last_price": last_price, "change_pct": change_pct,
            "bias": analysis["bias"], "last_rsi": analysis["last_rsi"]},
            "candles": candles, "ema_lines": analysis["ema_lines"], "rsi": analysis["rsi"],
            "order_blocks": analysis["order_blocks"], "signals": analysis["signals"],
            "summary": analysis["summary"], "htf_bias": analysis.get("htf_bias", "neutral"), "ai_analysis": analysis.get("ai_analysis", "")})
    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timed out"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Instruments ───────────────────────────────────────────

@app.route("/api/instruments")
def get_instruments():
    return jsonify(INSTRUMENTS)


# ── API: Backtest ──────────────────────────────────────────────

@app.route("/api/backtest")
def get_backtest():
    symbol    = request.args.get("symbol", "EURUSD")
    timeframe = request.args.get("timeframe", "1d")
    if symbol not in INSTRUMENTS:
        return jsonify({"error": "Unknown symbol"}), 400
    instrument = INSTRUMENTS[symbol]
    tf_map = {"1h": {"interval": "1h", "outputsize": 500},
              "4h": {"interval": "4h", "outputsize": 300},
              "1d": {"interval": "1day", "outputsize": 365}}
    tf = tf_map.get(timeframe, tf_map["1d"])
    try:
        params = {"symbol": instrument["symbol"], "interval": tf["interval"],
                  "outputsize": tf["outputsize"], "apikey": API_KEY, "format": "JSON"}
        resp = requests.get(f"{BASE_URL}/time_series", params=params, timeout=15)
        data = resp.json()
        if "code" in data:
            return jsonify({"error": data.get("message", "Twelve Data error")}), 502
        if "values" not in data:
            return jsonify({"error": "No data returned"}), 502
        candles = []
        for bar in data["values"]:
            candles.append({"time": int(pd.Timestamp(bar["datetime"]).timestamp()),
                "open": round(float(bar["open"]), 5), "high": round(float(bar["high"]), 5),
                "low": round(float(bar["low"]), 5), "close": round(float(bar["close"]), 5),
                "volume": round(float(bar.get("volume", 0)), 2)})
        candles.sort(key=lambda x: x["time"])
        result = run_backtest(candles, symbol=symbol)
        return jsonify(result)
    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timed out"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Alerts ────────────────────────────────────────────────

@app.route("/api/alerts")
def get_alerts():
    return jsonify({
        "alerts":  latest_alerts,
        "status":  scanner_status,
        "count":   len(latest_alerts),
    })

@app.route("/api/alerts/configure", methods=["POST"])
def configure_alerts():
    data = request.get_json()
    email_config["enabled"]    = data.get("enabled", False)
    email_config["sender"]     = data.get("sender", "")
    email_config["app_password"] = data.get("app_password", "")
    email_config["recipient"]  = data.get("recipient", "")
    return jsonify({"ok": True, "email_enabled": email_config["enabled"]})

@app.route("/api/alerts/clear", methods=["POST"])
def clear_alerts():
    latest_alerts.clear()
    return jsonify({"ok": True})

@app.route("/api/alerts/scan", methods=["POST"])
def trigger_scan():
    """Manual scan trigger from the UI."""
    import threading
    from scanner import run_scan
    threading.Thread(target=run_scan, daemon=True).start()
    return jsonify({"ok": True, "message": "Scan started"})


# ── Run ────────────────────────────────────────────────────────

# Only start scanner locally — Render free tier has 512MB RAM limit
import os
IS_RENDER = os.environ.get("RENDER", False)
if not IS_RENDER and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
    start_scanner()

if __name__ == "__main__":
    app.run(debug=False, port=8080, use_reloader=False)
