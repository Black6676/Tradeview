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

@app.route("/walkforward")
def walkforward_page():
    return render_template("walkforward.html")


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


# ── API: Walk-Forward Backtest ────────────────────────────────

@app.route("/api/walkforward")
def get_walkforward():
    symbol    = request.args.get("symbol", "EURUSD")
    timeframe = request.args.get("timeframe", "1h")
    if symbol not in INSTRUMENTS:
        return jsonify({"error": "Unknown symbol"}), 400
    instrument = INSTRUMENTS[symbol]
    tf_map = {"1h": {"interval": "1h", "outputsize": 500},
              "4h": {"interval": "4h", "outputsize": 300}}
    tf = tf_map.get(timeframe, tf_map["1h"])
    try:
        params = {"symbol": instrument["symbol"], "interval": tf["interval"],
                  "outputsize": tf["outputsize"], "apikey": API_KEY, "format": "JSON"}
        resp = requests.get(f"{BASE_URL}/time_series", params=params, timeout=15)
        data = resp.json()
        if "code" in data or "values" not in data:
            return jsonify({"error": data.get("message", "No data")}), 502
        candles = []
        for bar in data["values"]:
            candles.append({"time": int(pd.Timestamp(bar["datetime"]).timestamp()),
                "open": round(float(bar["open"]), 5), "high": round(float(bar["high"]), 5),
                "low": round(float(bar["low"]), 5), "close": round(float(bar["close"]), 5),
                "volume": round(float(bar.get("volume", 0)), 2)})
        candles.sort(key=lambda x: x["time"])
        from walkforward import run_walkforward
        result = run_walkforward(candles, symbol=symbol, timeframe=timeframe)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Optimization ────────────────────────────────────────

@app.route("/api/optimize")
def get_optimize():
    symbol    = request.args.get("symbol", "XAUUSD")
    timeframe = request.args.get("timeframe", "1h")
    n_trials  = int(request.args.get("trials", 50))
    if symbol not in INSTRUMENTS:
        return jsonify({"error": "Unknown symbol"}), 400
    try:
        from optimize import run_optimization
        result = run_optimization(symbol=symbol, timeframe=timeframe, n_trials=n_trials)
        if result is None:
            return jsonify({"error": "Optuna not installed on server"}), 500
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: MT5 Trade Execution ──────────────────────────────────

@app.route("/api/trade", methods=["POST"])
def execute_mt5_trade():
    """
    Trigger a real MT5 trade from the web UI.
    Only works when running locally on Windows with MT5 open.
    Body: { symbol, type, price, sl, tp, rr, confidence, lot }
    """
    try:
        data   = request.get_json()
        signal = {
            "type":       data.get("type"),
            "price":      data.get("price"),
            "sl":         data.get("sl"),
            "tp":         data.get("tp"),
            "rr":         data.get("rr", 2.0),
            "confidence": data.get("confidence", 0),
        }
        symbol   = data.get("symbol", "XAUUSD")
        lot      = float(data.get("lot", 0.01))
        use_mt5  = data.get("use_mt5", False)

        from algorithm import execute_trade
        execute_trade(signal, symbol=symbol, lot=lot, use_mt5=use_mt5)

        return jsonify({"ok": True, "message": f"Trade submitted: {signal['type'].upper()} {symbol}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Signal Diagnostics ───────────────────────────────────

@app.route("/api/debug")
def debug_signals():
    """Returns raw signal engine stats to diagnose why trades aren't firing."""
    symbol    = request.args.get("symbol", "EURUSD")
    timeframe = request.args.get("timeframe", "1h")
    instrument = INSTRUMENTS.get(symbol, INSTRUMENTS["EURUSD"])
    tf_map = {"1h": {"interval": "1h", "outputsize": 500},
              "4h": {"interval": "4h", "outputsize": 300}}
    tf = tf_map.get(timeframe, tf_map["1h"])
    try:
        params = {"symbol": instrument["symbol"], "interval": tf["interval"],
                  "outputsize": tf["outputsize"], "apikey": API_KEY, "format": "JSON"}
        resp = requests.get(f"{BASE_URL}/time_series", params=params, timeout=10)
        data = resp.json()
        if "values" not in data:
            return jsonify({"error": "No data"}), 502
        candles = []
        for bar in data["values"]:
            candles.append({"time": int(pd.Timestamp(bar["datetime"]).timestamp()),
                "open": round(float(bar["open"]), 5), "high": round(float(bar["high"]), 5),
                "low": round(float(bar["low"]), 5), "close": round(float(bar["close"]), 5),
                "volume": round(float(bar.get("volume", 0)), 2)})
        candles.sort(key=lambda x: x["time"])
        import algorithm as alg
        import pandas as pd2
        df = pd2.DataFrame(candles)
        for col in ["open","high","low","close"]: df[col] = df[col].astype(float)
        df["time"] = df["time"].astype(int)
        df = df.sort_values("time").reset_index(drop=True)
        atr = alg.compute_atr(df)
        htf_bias_map = alg.get_htf_bias(df)
        bull_obs, bear_obs = alg.detect_order_blocks(df)
        swing_highs, swing_lows = alg.detect_swings(df)
        structure = alg.detect_structure_breaks(swing_highs, swing_lows)
        liquidity = alg.detect_liquidity(df)
        fvgs = alg.detect_fvg(df)
        closes = df["close"]
        ema200 = alg.compute_ema(closes, 200)
        last_date = pd2.Timestamp(int(df["time"].values[-1]), unit="s").date()
        htf_bias = htf_bias_map.get(last_date, "neutral")
        session_candles = sum(1 for ts in df["time"].values if alg.is_trading_session(int(ts)))
        return jsonify({
            "symbol": symbol, "timeframe": timeframe,
            "total_candles": len(df),
            "session_candles": session_candles,
            "bull_obs": len(bull_obs), "bear_obs": len(bear_obs),
            "bos_events": len(structure),
            "liquidity_events": len(liquidity),
            "fvg_events": len(fvgs),
            "htf_bias": htf_bias,
            "confidence_threshold": alg.CONFIDENCE_THRESHOLD,
            "last_close": round(float(closes.iloc[-1]), 5),
            "ema200": round(float(ema200.iloc[-1]), 5),
            "above_ema200": float(closes.iloc[-1]) > float(ema200.iloc[-1]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: MT5 Live Trading ─────────────────────────────────────

@app.route("/api/mt5/account")
def mt5_account():
    """Get live Vantage account info."""
    try:
        from mt5_connection import connect, get_account_info, disconnect
        if not connect():
            return jsonify({"error": "MT5 not connected — open MT5 and check credentials"}), 503
        info = get_account_info()
        disconnect()
        return jsonify(info)
    except ImportError:
        return jsonify({"error": "MT5 only available on Windows with MetaTrader5 installed"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/mt5/positions")
def mt5_positions():
    """Get all open positions on Vantage demo."""
    try:
        from mt5_connection import connect, get_open_positions, disconnect
        if not connect():
            return jsonify({"error": "MT5 not connected"}), 503
        positions = get_open_positions()
        disconnect()
        return jsonify({"positions": positions, "count": len(positions)})
    except ImportError:
        return jsonify({"error": "MT5 only available on Windows"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/mt5/trade", methods=["POST"])
def mt5_trade():
    """Place a live trade on Vantage demo via MT5."""
    try:
        data   = request.get_json()
        signal = {
            "type":       data.get("type"),
            "price":      data.get("price"),
            "sl":         data.get("sl"),
            "tp":         data.get("tp"),
            "rr":         data.get("rr", 2.0),
            "confidence": data.get("confidence", 0),
        }
        symbol = data.get("symbol", "XAUUSD")
        lot    = float(data.get("lot", 0.01))

        from mt5_connection import connect, place_order, disconnect
        if not connect():
            return jsonify({"error": "MT5 not connected — open MT5 first"}), 503
        result = place_order(signal, symbol=symbol, lot=lot)
        disconnect()

        if result and result.retcode == 10009:  # TRADE_RETCODE_DONE
            return jsonify({"ok": True, "ticket": result.order,
                           "message": f"{signal['type'].upper()} {symbol} placed"})
        else:
            code = result.retcode if result else "unknown"
            comment = result.comment if result else "No result"
            return jsonify({"ok": False, "retcode": code, "message": comment}), 400

    except ImportError:
        return jsonify({"error": "MT5 only available on Windows"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/mt5/close", methods=["POST"])
def mt5_close():
    """Close a specific open position by ticket."""
    try:
        data   = request.get_json()
        ticket = int(data.get("ticket"))
        from mt5_connection import connect, close_position, disconnect
        if not connect():
            return jsonify({"error": "MT5 not connected"}), 503
        result = close_position(ticket)
        disconnect()
        if result and result.retcode == 10009:
            return jsonify({"ok": True, "message": f"Position {ticket} closed"})
        return jsonify({"ok": False, "message": "Close failed"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ────────────────────────────────────────────────────────

# Only start scanner locally — disabled on Render (RAM limit)
import os
IS_RENDER = os.environ.get("RENDER", False)
if not IS_RENDER and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
    try:
        start_scanner()
    except Exception as e:
        print(f"[Scanner] Failed to start: {e}")

if __name__ == "__main__":
    app.run(debug=False, port=8080, use_reloader=False)
