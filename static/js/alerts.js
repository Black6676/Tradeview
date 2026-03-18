let pollInterval = null;
let notifPermission = Notification.permission;

document.addEventListener('DOMContentLoaded', () => {
  bindControls();
  loadConfig();
  startPolling();
  updatePermBtn();
});

function bindControls() {
  document.getElementById('scanNowBtn').addEventListener('click', async () => {
    setStatus('Scanning all instruments…');
    document.getElementById('scanNowBtn').textContent = 'Scanning…';
    document.getElementById('scanNowBtn').disabled = true;
    await fetch('/api/alerts/scan', { method: 'POST' });
    setTimeout(() => {
      fetchAlerts();
      document.getElementById('scanNowBtn').textContent = 'Scan Now';
      document.getElementById('scanNowBtn').disabled = false;
    }, 8000);
  });

  document.getElementById('clearBtn').addEventListener('click', async () => {
    await fetch('/api/alerts/clear', { method: 'POST' });
    document.getElementById('alertList').innerHTML = emptyState();
    setStatus('Alerts cleared');
  });

  document.getElementById('saveConfigBtn').addEventListener('click', saveConfig);

  document.getElementById('permBtn').addEventListener('click', async () => {
    const perm = await Notification.requestPermission();
    notifPermission = perm;
    updatePermBtn();
  });
}

function startPolling() {
  fetchAlerts();
  pollInterval = setInterval(fetchAlerts, 30 * 1000); // poll every 30s
}

async function fetchAlerts() {
  try {
    const res  = await fetch('/api/alerts');
    const data = await res.json();
    renderAlerts(data.alerts);
    updateScannerBadge(data.status);
    setStatus(`Last scan: ${data.status.last_scan || '—'} · Next: ${data.status.next_scan || '—'}`);
  } catch (e) {
    setStatus('Error fetching alerts');
  }
}

function renderAlerts(alerts) {
  const list = document.getElementById('alertList');
  if (!alerts || alerts.length === 0) {
    list.innerHTML = emptyState();
    return;
  }

  list.innerHTML = alerts.map(a => {
    const dir      = a.type === 'buy' ? '▲ BUY' : '▼ SELL';
    const dirClass = a.type;
    const time     = a.time ? new Date(a.time * 1000).toLocaleString() : '—';
    const sym      = { EURUSD: 'EUR/USD', XAUUSD: 'XAU/USD', USDJPY: 'USD/JPY' }[a.symbol] || a.symbol;

    return `
    <div class="alert-card ${dirClass}">
      <div class="alert-dir ${dirClass}">${dir}</div>
      <div class="alert-body">
        <div class="alert-pair">${sym}</div>
        <div class="alert-tf">${(a.timeframe || '').toUpperCase()} · RSI ${a.rsi}</div>
        <div class="alert-meta">
          <div class="alert-meta-item">
            <span class="alert-meta-label">Entry</span>
            <span class="alert-meta-value">${a.price}</span>
          </div>
          <div class="alert-meta-item">
            <span class="alert-meta-label">Stop Loss</span>
            <span class="alert-meta-value sl">${a.sl}</span>
          </div>
          <div class="alert-meta-item">
            <span class="alert-meta-label">Take Profit</span>
            <span class="alert-meta-value tp">${a.tp}</span>
          </div>
          <div class="alert-meta-item">
            <span class="alert-meta-label">R:R</span>
            <span class="alert-meta-value">1:${a.rr}</span>
          </div>
        </div>
      </div>
      <div class="alert-time">${time}</div>
    </div>`;
  }).join('');

  // Fire browser notification for newest alert
  if (alerts.length > 0 && notifPermission === 'granted') {
    const a   = alerts[0];
    const sym = { EURUSD: 'EUR/USD', XAUUSD: 'XAU/USD', USDJPY: 'USD/JPY' }[a.symbol] || a.symbol;
    new Notification(`TradeView: ${a.type.toUpperCase()} ${sym}`, {
      body: `Entry: ${a.price} · SL: ${a.sl} · TP: ${a.tp}`,
      icon: '/static/favicon.ico',
    });
  }
}

function emptyState() {
  return `<div class="empty-alerts" id="emptyState">
    <div class="empty-icon">◎</div>
    <div class="empty-text">No signals yet — scanner checks every 15 minutes</div>
    <div class="empty-sub">Click "Scan Now" to run an immediate check</div>
  </div>`;
}

function updateScannerBadge(status) {
  const el = document.getElementById('scannerStatus');
  if (el) el.textContent = status.last_scan ? `Last scan: ${status.last_scan}` : 'Scanner running…';
}

async function saveConfig() {
  const cfg = {
    enabled:      document.getElementById('emailToggle').checked,
    sender:       document.getElementById('senderEmail').value.trim(),
    app_password: document.getElementById('appPassword').value.trim(),
    recipient:    document.getElementById('recipientEmail').value.trim(),
  };
  const res  = await fetch('/api/alerts/configure', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg),
  });
  const data = await res.json();
  const el   = document.getElementById('saveStatus');
  el.textContent = data.ok ? '✓ Saved' : 'Error saving';
  setTimeout(() => { el.textContent = ''; }, 3000);
  localStorage.setItem('alertConfig', JSON.stringify(cfg));
}

function loadConfig() {
  const raw = localStorage.getItem('alertConfig');
  if (!raw) return;
  const cfg = JSON.parse(raw);
  document.getElementById('emailToggle').checked    = cfg.enabled || false;
  document.getElementById('senderEmail').value      = cfg.sender || '';
  document.getElementById('recipientEmail').value   = cfg.recipient || '';
}

function updatePermBtn() {
  const btn = document.getElementById('permBtn');
  const msg = document.getElementById('permStatus');
  if (notifPermission === 'granted') {
    btn.textContent  = '✓ Notifications Enabled';
    btn.style.color  = 'var(--green)';
    btn.style.borderColor = 'var(--green)';
    msg.textContent  = 'Browser will show a popup when a signal fires.';
  } else if (notifPermission === 'denied') {
    btn.textContent  = 'Notifications Blocked';
    btn.style.color  = 'var(--red)';
    msg.textContent  = 'Unblock in browser settings to enable.';
  }
}

function setStatus(msg) {
  const el = document.getElementById('alertStatus');
  if (el) el.textContent = msg;
}
