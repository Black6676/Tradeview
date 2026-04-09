"""
MetaTrader 5 — Vantage Demo Connection
Account: 24786681
Server:  VantageInternational-Demo

SETUP:
1. Open MetaTrader 5 on your laptop
2. Make sure you are logged into your Vantage demo account
3. Run: pip install MetaTrader5
4. Fill in your password below where it says YOUR_PASSWORD_HERE
5. Run: python mt5_connection.py  (to test the connection)
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# ── Credentials ────────────────────────────────────────────────
MT5_ACCOUNT  = 24786681
MT5_PASSWORD = "Black@123"   # <-- fill this in
MT5_SERVER   = "VantageInternational-Demo"


# ── Connect ────────────────────────────────────────────────────

def connect():
    """Initialize and log into MT5. Returns True if successful."""
    if not mt5.initialize():
        print(f"[MT5] initialize() failed — error: {mt5.last_error()}")
        return False

    authorized = mt5.login(
        login=MT5_ACCOUNT,
        password=MT5_PASSWORD,
        server=MT5_SERVER,
    )

    if not authorized:
        print(f"[MT5] Login failed — error: {mt5.last_error()}")
        mt5.shutdown()
        return False

    info = mt5.account_info()
    print(f"[MT5] ✓ Connected to {MT5_SERVER}")
    print(f"      Account : {info.login}")
    print(f"      Name    : {info.name}")
    print(f"      Balance : ${info.balance:.2f}")
    print(f"      Equity  : ${info.equity:.2f}")
    print(f"      Leverage: 1:{info.leverage}")
    return True


def disconnect():
    mt5.shutdown()
    print("[MT5] Disconnected.")


# ── Fetch live OHLCV data ──────────────────────────────────────

def fetch_candles(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H1, n=500):
    """
    Fetch OHLCV candles directly from Vantage via MT5.
    Timeframes: mt5.TIMEFRAME_M5, M15, H1, H4, D1
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) == 0:
        print(f"[MT5] No data for {symbol} — error: {mt5.last_error()}")
        return []

    candles = []
    for r in rates:
        candles.append({
            "time":   int(r["time"]),
            "open":   float(r["open"]),
            "high":   float(r["high"]),
            "low":    float(r["low"]),
            "close":  float(r["close"]),
            "volume": float(r["tick_volume"]) if "tick_volume" in r.dtype.names else 0.0,
        })

    print(f"[MT5] Fetched {len(candles)} candles for {symbol}")
    return candles


# ── Account info ───────────────────────────────────────────────

def get_account_info():
    info = mt5.account_info()
    if info is None:
        return {}
    return {
        "login":    info.login,
        "name":     info.name,
        "balance":  info.balance,
        "equity":   info.equity,
        "margin":   info.margin,
        "free_margin": info.margin_free,
        "leverage": info.leverage,
        "currency": info.currency,
        "server":   info.server,
    }


# ── Get open positions ─────────────────────────────────────────

def get_open_positions():
    positions = mt5.positions_get()
    if positions is None:
        return []
    result = []
    for p in positions:
        result.append({
            "ticket":  p.ticket,
            "symbol":  p.symbol,
            "type":    "buy" if p.type == 0 else "sell",
            "volume":  p.volume,
            "open_price": p.price_open,
            "sl":      p.sl,
            "tp":      p.tp,
            "profit":  p.profit,
            "comment": p.comment,
        })
    return result


# ── Place order ────────────────────────────────────────────────

def place_order(signal, symbol="XAUUSD", lot=0.01):
    """
    Send a market order to Vantage via MT5.
    signal dict must have: type, sl, tp, confidence
    CAUTION: This places real orders on your demo account.
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[MT5] Cannot get tick for {symbol}")
        return None

    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        print(f"[MT5] Cannot get symbol info for {symbol}")
        return None

    if not sym_info.visible:
        mt5.symbol_select(symbol, True)

    price      = tick.ask if signal["type"] == "buy" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal["type"] == "buy" else mt5.ORDER_TYPE_SELL

    # Round SL/TP to symbol's decimal places
    digits = sym_info.digits
    sl     = round(signal["sl"], digits)
    tp     = round(signal["tp"], digits)

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot,
        "type":         order_type,
        "price":        price,
        "sl":           sl,
        "tp":           tp,
        "deviation":    20,
        "magic":        123456,
        "comment":      f"TradeView | {signal.get('confidence', '—')}% | AI",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None:
        print(f"[MT5] order_send returned None — error: {mt5.last_error()}")
        return None

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[MT5] ✓ Order placed: {signal['type'].upper()} {symbol}")
        print(f"      Ticket : {result.order}")
        print(f"      Price  : {price}")
        print(f"      Lot    : {lot}")
        print(f"      SL     : {sl} | TP: {tp}")
        print(f"      Confidence: {signal.get('confidence', '—')}%")
    else:
        print(f"[MT5] ✗ Order failed")
        print(f"      Retcode : {result.retcode}")
        print(f"      Comment : {result.comment}")

    return result


# ── Close position ─────────────────────────────────────────────

def close_position(ticket):
    """Close a specific open position by ticket number."""
    position = mt5.positions_get(ticket=ticket)
    if not position:
        print(f"[MT5] Position {ticket} not found")
        return None

    pos  = position[0]
    tick = mt5.symbol_info_tick(pos.symbol)
    price = tick.bid if pos.type == 0 else tick.ask
    close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       pos.symbol,
        "volume":       pos.volume,
        "type":         close_type,
        "position":     ticket,
        "price":        price,
        "deviation":    20,
        "magic":        123456,
        "comment":      "TradeView | Close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[MT5] ✓ Position {ticket} closed.")
    else:
        print(f"[MT5] ✗ Close failed: {result.comment if result else mt5.last_error()}")
    return result


# ── Test connection ────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing MT5 connection to Vantage...\n")

    if connect():
        print("\n── Account Info ──────────────────────────")
        info = get_account_info()
        for k, v in info.items():
            print(f"  {k}: {v}")

        print("\n── Fetching XAUUSD H1 candles ────────────")
        candles = fetch_candles("XAUUSD", mt5.TIMEFRAME_H1, 10)
        if candles:
            df = pd.DataFrame(candles)
            df["datetime"] = pd.to_datetime(df["time"], unit="s")
            print(df[["datetime","open","high","low","close","volume"]].tail(5).to_string(index=False))

        print("\n── Open Positions ────────────────────────")
        positions = get_open_positions()
        if positions:
            for p in positions:
                print(f"  {p['symbol']} {p['type'].upper()} {p['volume']} lots | P&L: ${p['profit']:.2f}")
        else:
            print("  No open positions.")

        disconnect()
    else:
        print("\n✗ Connection failed. Check:")
        print("  1. MT5 is open and logged into your Vantage demo account")
        print("  2. Your password is correct in mt5_connection.py")
        print("  3. Server name is VantageInternational-Demo")