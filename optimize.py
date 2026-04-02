"""
Hyperparameter Optimization using Optuna (Bayesian search).
Tunes the TradeView algorithm parameters to maximize Sharpe ratio.

Install: pip install optuna
Usage:
    python optimize.py --symbol XAUUSD --timeframe 1h --trials 100
"""
import argparse
import pandas as pd
import numpy as np
import requests

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[Optimize] Optuna not installed. Run: pip install optuna")

API_KEY  = "0a603f27b4664a98bfb3d6bac2af9f9b"
BASE_URL = "https://api.twelvedata.com"


# ── Fetch historical data ──────────────────────────────────────

def fetch_candles(symbol="EURUSD", timeframe="1h", outputsize=500):
    symbol_map = {
        "EURUSD": "EUR/USD", "XAUUSD": "XAU/USD", "USDJPY": "USD/JPY"
    }
    interval_map = {"1h": "1h", "4h": "4h", "1d": "1day"}
    params = {
        "symbol":     symbol_map.get(symbol, symbol),
        "interval":   interval_map.get(timeframe, "1h"),
        "outputsize": outputsize,
        "apikey":     API_KEY,
        "format":     "JSON",
    }
    resp = requests.get(f"{BASE_URL}/time_series", params=params, timeout=15)
    data = resp.json()
    if "values" not in data:
        raise ValueError(f"No data: {data.get('message', 'Unknown error')}")
    candles = [{"time": int(pd.Timestamp(b["datetime"]).timestamp()),
                "open": round(float(b["open"]), 5), "high": round(float(b["high"]), 5),
                "low": round(float(b["low"]), 5), "close": round(float(b["close"]), 5),
                "volume": round(float(b.get("volume", 0)), 2)} for b in data["values"]]
    candles.sort(key=lambda x: x["time"])
    return candles


# ── Parameterised signal engine ────────────────────────────────

def run_analysis_with_params(candles, params, symbol="EURUSD", timeframe="1h"):
    """
    Runs the full analysis pipeline with tunable parameters injected.
    Keeps algorithm.py clean — no globals modified.
    """
    import algorithm as alg
    import pandas as pd_

    df = pd_.DataFrame(candles)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = df["time"].astype(int)
    df = df.sort_values("time").reset_index(drop=True)

    closes = df["close"]
    times  = df["time"].values
    atr    = alg.compute_atr(df)

    ema20  = alg.compute_ema(closes, 20)
    ema50  = alg.compute_ema(closes, 50)
    ema200 = alg.compute_ema(closes, 200)
    rsi    = alg.compute_rsi(closes, 14)

    htf_bias_map       = alg.get_htf_bias(df)
    bull_obs, bear_obs = alg.detect_order_blocks(df)
    all_obs            = bull_obs + bear_obs

    swing_highs, swing_lows = alg.detect_swings(df)
    structure = alg.detect_structure_breaks(swing_highs, swing_lows)
    liquidity = alg.detect_liquidity(df)

    # FVG with tunable lookback
    fvg_lookback = params.get("fvg_lookback", 30)
    fvgs         = alg.detect_fvg(df)

    conf_threshold = params.get("conf_base", 70)
    atr_mult       = params.get("atr_mult", 1.5)
    bos_window     = params.get("bos_window", 50)
    risk_pct       = params.get("risk_pct", 0.01)

    signals = []

    for i in range(200, len(df)):
        price   = float(closes.iloc[i])
        ts      = int(times[i])
        e200    = float(ema200.iloc[i])
        rsi_val = float(rsi.iloc[i])
        atr_val = float(atr.iloc[i])

        if not alg.is_trading_session(ts):
            continue

        candle_date  = pd_.Timestamp(ts, unit="s").date()
        htf          = htf_bias_map.get(candle_date, "neutral")

        has_bull_bos = any(b["type"] == "bullish_bos" and i - bos_window < b["idx"] < i for b in structure)
        has_bear_bos = any(b["type"] == "bearish_bos" and i - bos_window < b["idx"] < i for b in structure)
        has_liq      = any(i - 20 < l["idx"] < i for l in liquidity)
        has_bull_fvg = any(i - fvg_lookback < f["idx"] < i for f in fvgs if f["type"] == "bullish")
        has_bear_fvg = any(i - fvg_lookback < f["idx"] < i for f in fvgs if f["type"] == "bearish")

        for ob in all_obs:
            ob_rows = df[df["time"] == ob["time"]]
            if ob_rows.empty or ob_rows.index[0] >= i:
                continue
            in_zone = ob["bottom"] <= price <= ob["top"]

            if ob["type"] == "bullish" and htf == "bullish" and in_zone and price > e200:
                score = 20
                if has_bull_bos: score += 25
                if has_liq:      score += 20
                if has_bull_fvg: score += 15
                if 45 <= rsi_val <= 70: score += 15
                if score >= 50:
                    sl  = round(ob["bottom"] - atr_val * atr_mult, 5)
                    tp  = round(price + (price - sl) * 2.0, 5)
                    sig = {"time": ts, "type": "buy", "price": round(price, 5),
                           "rsi": round(rsi_val, 1), "sl": sl, "tp": tp, "rr": 2.0,
                           "atr": round(atr_val, 5), "htf": htf,
                           "lot": alg.lot_size(round(price - sl, 5),
                                               risk_pct=risk_pct)}
                    sig["confidence"] = alg.compute_confidence(sig, "bullish", rsi_val, htf)
                    if sig["confidence"] >= conf_threshold:
                        signals.append(sig)

            elif ob["type"] == "bearish" and htf == "bearish" and in_zone and price < e200:
                score = 20
                if has_bear_bos: score += 25
                if has_liq:      score += 20
                if has_bear_fvg: score += 15
                if 30 <= rsi_val <= 55: score += 15
                if score >= 50:
                    sl  = round(ob["top"] + atr_val * atr_mult, 5)
                    tp  = round(price - (sl - price) * 2.0, 5)
                    sig = {"time": ts, "type": "sell", "price": round(price, 5),
                           "rsi": round(rsi_val, 1), "sl": sl, "tp": tp, "rr": 2.0,
                           "atr": round(atr_val, 5), "htf": htf,
                           "lot": alg.lot_size(round(sl - price, 5),
                                               risk_pct=risk_pct)}
                    sig["confidence"] = alg.compute_confidence(sig, "bearish", rsi_val, htf)
                    if sig["confidence"] >= conf_threshold:
                        signals.append(sig)

    # Deduplicate
    deduped, last_ts = [], 0
    for s in sorted(signals, key=lambda x: x["time"]):
        if s["time"] - last_ts > 3 * 3600:
            deduped.append(s); last_ts = s["time"]

    return deduped


# ── Forward simulation ─────────────────────────────────────────

def simulate_outcome(sig, future_candles):
    for c in future_candles:
        if sig["type"] == "buy":
            if float(c["low"])  <= sig["sl"]: return "loss", -1.0
            if float(c["high"]) >= sig["tp"]: return "win",  +2.0
        else:
            if float(c["high"]) >= sig["sl"]: return "loss", -1.0
            if float(c["low"])  <= sig["tp"]: return "win",  +2.0
    return "open", 0.0


# ── Walk-forward with injected params ─────────────────────────

def walkforward_with_params(candles, params, symbol="EURUSD",
                             timeframe="1h", window=400, step=80,
                             account_balance=1000):
    df = pd.DataFrame(candles)
    for col in ["open","high","low","close"]:
        df[col] = df[col].astype(float)
    df["time"] = df["time"].astype(int)
    df = df.sort_values("time").reset_index(drop=True)

    equity       = account_balance
    equity_curve = [equity]
    trades       = []
    seen         = set()

    for i in range(window, len(df), step):
        window_candles = df.iloc[i - window:i].to_dict("records")
        signals = run_analysis_with_params(window_candles, params, symbol, timeframe)
        if not signals:
            equity_curve.append(equity); continue

        sig     = signals[-1]
        sig_key = (sig["time"], sig["type"])
        if sig_key in seen:
            equity_curve.append(equity); continue
        seen.add(sig_key)

        future  = df.iloc[i:min(i + 50, len(df))].to_dict("records")
        outcome, pnl_r = simulate_outcome(sig, future)

        risk_dollars = account_balance * params.get("risk_pct", 0.01)
        pnl_dollars  = pnl_r * risk_dollars if outcome != "open" else 0
        equity      += pnl_dollars
        trades.append({**sig, "result": outcome, "pnl_r": pnl_r})
        equity_curve.append(equity)

    closed = [t for t in trades if t["result"] != "open"]
    if len(closed) < 3:
        return {"sharpe": -99, "trades": len(closed), "win_rate": 0}

    wins   = [t for t in closed if t["result"] == "win"]
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe  = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    win_rate = len(wins) / len(closed)
    pf_num   = sum(t["pnl_r"] for t in wins)
    pf_den   = abs(sum(t["pnl_r"] for t in closed if t["result"] == "loss"))
    pf       = pf_num / pf_den if pf_den > 0 else pf_num

    # Composite score: balance Sharpe + win rate + profit factor
    composite = sharpe * 0.5 + win_rate * 0.3 + min(pf, 5) * 0.2

    return {
        "sharpe":        round(sharpe, 3),
        "win_rate":      round(win_rate, 3),
        "profit_factor": round(pf, 3),
        "composite":     round(composite, 3),
        "trades":        len(closed),
    }


# ── Optuna objective ───────────────────────────────────────────

def make_objective(candles, symbol, timeframe):
    def objective(trial):
        params = {
            "atr_mult":    trial.suggest_float("atr_mult",    1.0, 2.5),
            "conf_base":   trial.suggest_int(  "conf_base",   55,  80),
            "fvg_lookback":trial.suggest_int(  "fvg_lookback",15,  50),
            "bos_window":  trial.suggest_int(  "bos_window",  20,  80),
            "risk_pct":    trial.suggest_float("risk_pct",    0.005, 0.02),
        }
        result = walkforward_with_params(candles, params, symbol, timeframe)
        if result["trades"] < 3:
            raise optuna.exceptions.TrialPruned()
        return result["composite"]
    return objective


# ── Main ───────────────────────────────────────────────────────

def run_optimization(symbol="XAUUSD", timeframe="1h", n_trials=100, outputsize=500):
    if not OPTUNA_AVAILABLE:
        print("Install optuna: pip install optuna")
        return None

    print(f"\n[Optimize] Fetching {outputsize} candles for {symbol} {timeframe}…")
    candles = fetch_candles(symbol, timeframe, outputsize)
    print(f"[Optimize] Got {len(candles)} candles. Starting {n_trials} trials…\n")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(make_objective(candles, symbol, timeframe), n_trials=n_trials,
                   show_progress_bar=True)

    best = study.best_params
    best_val = study.best_value

    print(f"\n{'='*50}")
    print(f"[Optimize] Best params for {symbol} {timeframe}:")
    for k, v in best.items():
        print(f"  {k}: {v}")
    print(f"  Composite score: {best_val:.3f}")
    print(f"{'='*50}\n")

    # Validate best params on full dataset
    print("[Optimize] Validating best params on full dataset…")
    validation = walkforward_with_params(candles, best, symbol, timeframe)
    print(f"  Sharpe: {validation['sharpe']} | Win Rate: {validation['win_rate']:.1%} | "
          f"PF: {validation['profit_factor']} | Trades: {validation['trades']}")

    return {
        "best_params":  best,
        "best_score":   best_val,
        "validation":   validation,
        "trials_df":    study.trials_dataframe().to_dict("records"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    default="XAUUSD")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--trials",    type=int, default=100)
    args = parser.parse_args()

    result = run_optimization(args.symbol, args.timeframe, args.trials)
    if result:
        print("\nBest params summary:", result["best_params"])
