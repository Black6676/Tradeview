import smtplib
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


ALERT_LOG_FILE = "alert_log.json"


def load_alert_log():
    if os.path.exists(ALERT_LOG_FILE):
        with open(ALERT_LOG_FILE, "r") as f:
            return json.load(f)
    return []


def save_alert_log(log):
    with open(ALERT_LOG_FILE, "w") as f:
        json.dump(log[-100:], f)  # keep last 100 alerts


def is_duplicate(signal, log):
    """Prevent re-alerting the same signal within 4 hours."""
    for entry in log:
        if (entry["symbol"]    == signal["symbol"] and
            entry["type"]      == signal["type"] and
            entry["timeframe"] == signal["timeframe"] and
            abs(entry["time"]  - signal["time"]) < 4 * 3600):
            return True
    return False


def send_email_alert(signal, email_cfg):
    """Send email via Gmail SMTP."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🔔 TradeView Signal: {signal['type'].upper()} {signal['symbol']} @ {signal['price']}"
        msg["From"]    = email_cfg["sender"]
        msg["To"]      = email_cfg["recipient"]

        direction = "BUY  ▲" if signal["type"] == "buy" else "SELL ▼"
        color     = "#26a69a" if signal["type"] == "buy" else "#ef5350"

        html = f"""
        <html><body style="background:#0a0b0e;color:#e2e6f0;font-family:monospace;padding:24px;">
          <div style="max-width:480px;margin:0 auto;background:#111318;border:1px solid #1f2535;border-radius:8px;padding:24px;">
            <div style="font-size:11px;color:#8892a8;letter-spacing:0.1em;margin-bottom:8px;">TRADEVIEW SIGNAL ALERT</div>
            <div style="font-size:28px;font-weight:700;color:{color};margin-bottom:16px;">{direction}</div>
            <table style="width:100%;border-collapse:collapse;">
              <tr><td style="color:#8892a8;padding:6px 0;font-size:12px;">Instrument</td><td style="font-weight:700;font-size:14px;">{signal['symbol']}</td></tr>
              <tr><td style="color:#8892a8;padding:6px 0;font-size:12px;">Timeframe</td><td style="font-weight:700;">{signal['timeframe'].upper()}</td></tr>
              <tr><td style="color:#8892a8;padding:6px 0;font-size:12px;">Entry Price</td><td style="font-weight:700;">{signal['price']}</td></tr>
              <tr><td style="color:#8892a8;padding:6px 0;font-size:12px;">Stop Loss</td><td style="color:#ef5350;font-weight:700;">{signal['sl']}</td></tr>
              <tr><td style="color:#8892a8;padding:6px 0;font-size:12px;">Take Profit</td><td style="color:#26a69a;font-weight:700;">{signal['tp']}</td></tr>
              <tr><td style="color:#8892a8;padding:6px 0;font-size:12px;">Risk/Reward</td><td style="font-weight:700;">1:{signal['rr']}</td></tr>
              <tr><td style="color:#8892a8;padding:6px 0;font-size:12px;">RSI</td><td style="font-weight:700;">{signal['rsi']}</td></tr>
              <tr><td style="color:#8892a8;padding:6px 0;font-size:12px;">HTF Bias</td><td style="font-weight:700;">{signal.get('htf','—').upper()}</td></tr>
            </table>
            <div style="margin-top:20px;padding:12px;background:#0a0b0e;border-radius:4px;font-size:11px;color:#4a5470;">
              Signal detected at {datetime.utcfromtimestamp(signal['time']).strftime('%Y-%m-%d %H:%M UTC')}
            </div>
          </div>
        </body></html>
        """

        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(email_cfg["sender"], email_cfg["app_password"])
            server.sendmail(email_cfg["sender"], email_cfg["recipient"], msg.as_string())

        return True
    except Exception as e:
        print(f"[Email error] {e}")
        return False


def process_signals(signals, symbol, timeframe, email_cfg=None):
    """
    Compare new signals against alert log.
    Returns list of NEW signals that haven't been alerted yet.
    Sends email for each new signal if email_cfg is provided.
    """
    log = load_alert_log()
    new_alerts = []

    for sig in signals:
        enriched = {**sig, "symbol": symbol, "timeframe": timeframe}
        if not is_duplicate(enriched, log):
            new_alerts.append(enriched)
            log.append({
                "symbol":    symbol,
                "type":      sig["type"],
                "timeframe": timeframe,
                "time":      sig["time"],
                "price":     sig["price"],
                "alerted_at": int(datetime.utcnow().timestamp()),
            })
            if email_cfg and email_cfg.get("enabled"):
                send_email_alert(enriched, email_cfg)

    save_alert_log(log)
    return new_alerts
