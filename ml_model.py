"""
Phase 5 — XGBoost ML Signal Classifier
Predicts win probability for each trade signal using engineered features.
Replaces the manual confidence scoring with a data-driven model.

Install: pip install xgboost scikit-learn joblib
"""
import numpy as np
import pandas as pd
import os
import json

try:
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

MODEL_PATH   = "ml_model.joblib"
SCALER_PATH  = "ml_scaler.joblib"
DATA_PATH    = "ml_training_data.json"
MIN_SAMPLES  = 20   # minimum trades needed before training


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def extract_features(signal, df, idx):
    """
    Convert a signal + its market context into ML features.
    All features are normalized/relative so the model generalises
    across different instruments and price levels.
    """
    closes = df["close"].astype(float)
    highs  = df["high"].astype(float)
    lows   = df["low"].astype(float)

    from algorithm import (compute_ema, compute_rsi, compute_atr,
                            detect_swings, detect_structure_breaks,
                            detect_liquidity, detect_fvg)

    ema20  = compute_ema(closes, 20)
    ema50  = compute_ema(closes, 50)
    ema200 = compute_ema(closes, 200)
    rsi    = compute_rsi(closes, 14)
    atr    = compute_atr(df)

    price  = float(closes.iloc[idx])
    e20    = float(ema20.iloc[idx])
    e50    = float(ema50.iloc[idx])
    e200   = float(ema200.iloc[idx])
    rsi_v  = float(rsi.iloc[idx])
    atr_v  = float(atr.iloc[idx])

    # Price relative to EMAs (normalised by ATR)
    dist_e20  = (price - e20)  / atr_v if atr_v > 0 else 0
    dist_e50  = (price - e50)  / atr_v if atr_v > 0 else 0
    dist_e200 = (price - e200) / atr_v if atr_v > 0 else 0

    # EMA alignment (trend structure)
    ema_bullish = 1 if e20 > e50 > e200 else -1 if e20 < e50 < e200 else 0

    # RSI features
    rsi_norm      = (rsi_v - 50) / 50   # -1 to +1
    rsi_overbought = 1 if rsi_v > 70 else 0
    rsi_oversold   = 1 if rsi_v < 30 else 0

    # ATR relative to recent average (volatility regime)
    atr_mean = float(atr.iloc[max(0, idx-20):idx].mean()) if idx > 20 else atr_v
    atr_ratio = atr_v / atr_mean if atr_mean > 0 else 1.0

    # Signal direction encoded
    direction = 1 if signal["type"] == "buy" else -1

    # OB zone width relative to ATR
    ob_width = abs(signal.get("tp", price) - signal.get("sl", price))
    ob_width_norm = ob_width / atr_v if atr_v > 0 else 1.0

    # SL distance relative to ATR
    sl_dist = abs(price - signal["sl"]) / atr_v if atr_v > 0 else 1.0

    # HTF bias encoded
    htf = signal.get("htf", "neutral")
    htf_enc = 1 if htf == "bullish" else -1 if htf == "bearish" else 0

    # Alignment: signal direction matches HTF
    htf_aligned = 1 if (direction == 1 and htf == "bullish") or \
                       (direction == -1 and htf == "bearish") else 0

    # Recent BOS (structure breaks in last 50 candles)
    swing_h, swing_l = detect_swings(df.iloc[:idx+1])
    structure = detect_structure_breaks(swing_h, swing_l)
    bull_bos = sum(1 for b in structure if b["type"] == "bullish_bos" and idx - 50 < b["idx"] <= idx)
    bear_bos = sum(1 for b in structure if b["type"] == "bearish_bos" and idx - 50 < b["idx"] <= idx)
    recent_bos = bull_bos if direction == 1 else bear_bos

    # Liquidity events in last 40 candles
    liq = detect_liquidity(df.iloc[:idx+1])
    recent_liq = sum(1 for l in liq if idx - 40 < l["idx"] <= idx)

    # FVG presence in last 40 candles
    fvgs = detect_fvg(df.iloc[:idx+1])
    bull_fvg = sum(1 for f in fvgs if f["type"] == "bullish" and idx - 40 < f["idx"] <= idx)
    bear_fvg = sum(1 for f in fvgs if f["type"] == "bearish" and idx - 40 < f["idx"] <= idx)
    relevant_fvg = bull_fvg if direction == 1 else bear_fvg

    # Price momentum (last 5 candles)
    if idx >= 5:
        momentum = (float(closes.iloc[idx]) - float(closes.iloc[idx-5])) / atr_v if atr_v > 0 else 0
    else:
        momentum = 0

    # Candle body strength (current candle)
    open_p = float(df["open"].iloc[idx])
    body   = abs(float(closes.iloc[idx]) - open_p) / atr_v if atr_v > 0 else 0

    return {
        "direction":     direction,
        "dist_e20":      dist_e20,
        "dist_e50":      dist_e50,
        "dist_e200":     dist_e200,
        "ema_alignment": ema_bullish,
        "rsi_norm":      rsi_norm,
        "rsi_overbought":rsi_overbought,
        "rsi_oversold":  rsi_oversold,
        "atr_ratio":     atr_ratio,
        "ob_width_norm": ob_width_norm,
        "sl_dist":       sl_dist,
        "htf_encoded":   htf_enc,
        "htf_aligned":   htf_aligned,
        "recent_bos":    recent_bos,
        "recent_liq":    min(recent_liq, 20),   # cap to avoid outliers
        "relevant_fvg":  relevant_fvg,
        "momentum":      momentum,
        "body_strength": body,
    }


FEATURE_COLS = [
    "direction", "dist_e20", "dist_e50", "dist_e200", "ema_alignment",
    "rsi_norm", "rsi_overbought", "rsi_oversold", "atr_ratio",
    "ob_width_norm", "sl_dist", "htf_encoded", "htf_aligned",
    "recent_bos", "recent_liq", "relevant_fvg", "momentum", "body_strength",
]


# ══════════════════════════════════════════════════════════════
# TRAINING DATA MANAGEMENT
# ══════════════════════════════════════════════════════════════

def load_training_data():
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH) as f:
        return json.load(f)


def save_training_data(data):
    with open(DATA_PATH, "w") as f:
        json.dump(data[-500:], f)   # keep last 500 samples


def add_training_sample(features, result):
    """Add a completed trade to the training dataset."""
    data = load_training_data()
    data.append({**features, "result": 1 if result == "win" else 0})
    save_training_data(data)
    print(f"[ML] Training sample added — total: {len(data)}")


def build_training_data_from_backtest(backtest_trades, candles):
    """
    Seed the ML model using existing backtest results.
    Converts historical trades into training samples.
    """
    df = pd.DataFrame(candles)
    for col in ["open","high","low","close"]:
        df[col] = df[col].astype(float)
    df["time"] = df["time"].astype(int)
    df = df.sort_values("time").reset_index(drop=True)

    time_to_idx = {int(t): i for i, t in enumerate(df["time"].values)}
    added = 0

    for trade in backtest_trades:
        if trade.get("result") not in ("win", "loss"):
            continue
        ts  = trade.get("entry_time")
        idx = time_to_idx.get(ts)
        if idx is None or idx < 200:
            continue
        features = extract_features(trade, df, idx)
        add_training_sample(features, trade["result"])
        added += 1

    print(f"[ML] Seeded {added} training samples from backtest")
    return added


# ══════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════

def train_model():
    """Train XGBoost on accumulated trade data. Returns True if successful."""
    if not ML_AVAILABLE:
        print("[ML] XGBoost not installed — run: pip install xgboost scikit-learn joblib")
        return False

    data = load_training_data()
    if len(data) < MIN_SAMPLES:
        print(f"[ML] Need {MIN_SAMPLES} samples to train, have {len(data)}")
        return False

    df   = pd.DataFrame(data)
    X    = df[FEATURE_COLS].fillna(0).values
    y    = df["result"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_sc, y)

    # Cross-validation score
    scores = cross_val_score(model, X_sc, y, cv=min(5, len(data)//4), scoring="accuracy")
    print(f"[ML] Model trained — CV accuracy: {scores.mean():.1%} ± {scores.std():.1%} | Samples: {len(data)}")

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return True


def load_model():
    """Load trained model and scaler. Returns (model, scaler) or (None, None)."""
    if not ML_AVAILABLE:
        return None, None
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    try:
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    except Exception as e:
        print(f"[ML] Model load error: {e}")
        return None, None


# ══════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════

def predict_win_probability(signal, df, idx):
    """
    Returns win probability 0.0–1.0 using the trained ML model.
    Falls back to manual confidence score if model not available.
    """
    model, scaler = load_model()
    if model is None:
        # Fallback: convert manual confidence to probability
        return signal.get("confidence", 50) / 100.0

    try:
        features = extract_features(signal, df, idx)
        X = np.array([[features[c] for c in FEATURE_COLS]])
        X_sc = scaler.transform(X)
        prob = float(model.predict_proba(X_sc)[0][1])
        return round(prob, 3)
    except Exception as e:
        print(f"[ML] Prediction error: {e}")
        return signal.get("confidence", 50) / 100.0


def get_model_stats():
    """Return info about current model state."""
    data = load_training_data()
    model, _ = load_model()
    wins = sum(1 for d in data if d.get("result") == 1)
    return {
        "samples":       len(data),
        "wins":          wins,
        "losses":        len(data) - wins,
        "win_rate":      round(wins / len(data) * 100, 1) if data else 0,
        "model_trained": model is not None,
        "min_samples":   MIN_SAMPLES,
        "ready":         len(data) >= MIN_SAMPLES and model is not None,
    }
