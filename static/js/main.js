/* ── State ── */
let currentSymbol    = 'EURUSD';
let currentTimeframe = '1h';
let candleChart      = null;
let rsiChart         = null;
let obBoxes          = [];

/* ── Init ── */
window.addEventListener('load', () => {
  bindControls();
  loadChart();
});

function bindControls() {
  document.getElementById('instrumentTabs').addEventListener('click', e => {
    const btn = e.target.closest('.inst-btn');
    if (!btn) return;
    document.querySelectorAll('.inst-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentSymbol = btn.dataset.symbol;
    loadChart();
  });

  document.getElementById('tfTabs').addEventListener('click', e => {
    const btn = e.target.closest('.tf-btn');
    if (!btn) return;
    document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentTimeframe = btn.dataset.tf;
    loadChart();
  });

  document.getElementById('refreshBtn').addEventListener('click', () => loadChart());
}

async function loadChart() {
  showLoading(true);
  hideError();
  setStatus('Fetching ' + currentSymbol + ' ' + currentTimeframe + '…');
  document.getElementById('refreshBtn').classList.add('spinning');

  try {
    const res  = await fetch(`/api/ohlcv?symbol=${currentSymbol}&timeframe=${currentTimeframe}`);
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); }
    catch(e) { throw new Error('Server returned invalid response — please retry'); }

    if (!res.ok || data.error) throw new Error(data.error || 'Server error');
    if (!data.candles || !data.meta) throw new Error('Incomplete data');

    renderChart(data);
    updatePriceStrip(data.meta);
    updateAnalysisPanel(data.meta, data.order_blocks || [], data.signals || []);
    updateSummaryPanel(data.summary || {}, data.signals || []);
  const aiEl = document.getElementById('aiNarrative');
  if (aiEl && data.ai_analysis) aiEl.textContent = data.ai_analysis;
    updateAiNarrative(data.ai_analysis || '');
    setStatus(`Loaded ${data.meta.bars} bars · ${data.meta.label} · ${currentTimeframe.toUpperCase()} · Bias: ${(data.meta.bias||'—').toUpperCase()}`);

  } catch(err) {
    showError(err.message);
    setStatus('Error — ' + err.message);
  } finally {
    showLoading(false);
    document.getElementById('refreshBtn').classList.remove('spinning');
  }
}

function renderChart(data) {
  const candles = data.candles;
  const labels  = candles.map(c => {
    const d = new Date(c.time * 1000);
    return d.toLocaleDateString() + ' ' + d.getHours() + ':00';
  });

  // ── Candlestick data for Chart.js financial plugin ──
  const ohlc = candles.map(c => ({
    x: c.time * 1000,
    o: c.open, h: c.high, l: c.low, c: c.close,
  }));

  const closes = candles.map(c => ({ x: c.time * 1000, y: c.close }));

  // EMAs
  const ema20  = (data.ema_lines?.ema20  || []).map(e => ({ x: e.time * 1000, y: e.value }));
  const ema50  = (data.ema_lines?.ema50  || []).map(e => ({ x: e.time * 1000, y: e.value }));
  const ema200 = (data.ema_lines?.ema200 || []).map(e => ({ x: e.time * 1000, y: e.value }));

  // Volume
  const volume = candles.map(c => ({
    x: c.time * 1000, y: c.volume,
    color: c.close >= c.open ? 'rgba(38,166,154,0.4)' : 'rgba(239,83,80,0.4)',
  }));

  // Signal markers
  const buySignals  = (data.signals || []).filter(s => s.type === 'buy') .map(s => ({ x: s.time * 1000, y: s.price }));
  const sellSignals = (data.signals || []).filter(s => s.type === 'sell').map(s => ({ x: s.time * 1000, y: s.price }));

  // Destroy old charts
  if (candleChart) { candleChart.destroy(); candleChart = null; }
  if (rsiChart)    { rsiChart.destroy();    rsiChart    = null; }

  // ── Main price chart ──────────────────────────────────
  const ctx = document.getElementById('candleCanvas').getContext('2d');
  candleChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        { label: 'Close', data: closes, borderColor: '#e2e6f0', borderWidth: 1.5,
          pointRadius: 0, fill: false, yAxisID: 'price', tension: 0 },
        { label: 'EMA20',  data: ema20,  borderColor: '#f0b429', borderWidth: 1,
          pointRadius: 0, fill: false, yAxisID: 'price', tension: 0 },
        { label: 'EMA50',  data: ema50,  borderColor: '#4fc3f7', borderWidth: 1,
          pointRadius: 0, fill: false, yAxisID: 'price', tension: 0 },
        { label: 'EMA200', data: ema200, borderColor: '#ef5350', borderWidth: 1,
          pointRadius: 0, fill: false, yAxisID: 'price', tension: 0 },
        { label: 'Volume', data: volume, type: 'bar',
          backgroundColor: volume.map(v => v.color), yAxisID: 'volume', barPercentage: 0.8 },
        { label: 'Buy',  data: buySignals,  borderColor: '#26a69a', backgroundColor: '#26a69a',
          pointStyle: 'triangle', pointRadius: 8, showLine: false, yAxisID: 'price' },
        { label: 'Sell', data: sellSignals, borderColor: '#ef5350', backgroundColor: '#ef5350',
          pointStyle: 'triangle', pointRadius: 8, rotation: 180, showLine: false, yAxisID: 'price' },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: true, labels: { color: '#8892a8', font: { size: 11 }, boxWidth: 12 } },
        tooltip: {
          backgroundColor: '#111318', borderColor: '#1f2535', borderWidth: 1,
          titleColor: '#e2e6f0', bodyColor: '#8892a8',
          callbacks: {
            label: ctx => {
              if (ctx.dataset.label === 'Volume') return '';
              return ctx.dataset.label + ': ' + fmt(ctx.parsed.y);
            }
          }
        },
      },
      scales: {
        x: { type: 'time', time: { unit: 'hour', displayFormats: { hour: 'MMM d HH:mm' } },
             ticks: { color: '#8892a8', maxTicksLimit: 8, font: { size: 10 } },
             grid:  { color: '#1a1f2e' } },
        price:  { position: 'right', ticks: { color: '#8892a8', font: { size: 10 },
                  callback: v => fmt(v) }, grid: { color: '#1a1f2e' } },
        volume: { position: 'left', display: false, max: v => v.max * 8 },
      },
      onHover: (e, els, chart) => {
        if (els.length) {
          const idx = els[0].index;
          const c   = candles[idx];
          if (c) {
            document.getElementById('metaO').textContent = fmt(c.open);
            document.getElementById('metaH').textContent = fmt(c.high);
            document.getElementById('metaL').textContent = fmt(c.low);
            document.getElementById('metaC').textContent = fmt(c.close);
          }
        }
      },
    }
  });

  // Draw order block lines as annotations (manual)
  drawOrderBlockAnnotations(data.order_blocks || [], candles);

  // ── RSI chart ─────────────────────────────────────────
  const rsiData = (data.rsi || []).map(r => ({ x: r.time * 1000, y: r.value }));
  const rsiCtx  = document.getElementById('rsiCanvas').getContext('2d');
  const lastTime = candles.length ? candles[candles.length-1].time * 1000 : Date.now();
  const firstTime = candles.length ? candles[0].time * 1000 : Date.now() - 86400000;

  rsiChart = new Chart(rsiCtx, {
    type: 'line',
    data: {
      datasets: [
        { label: 'RSI', data: rsiData, borderColor: '#b39ddb', borderWidth: 1.5,
          pointRadius: 0, fill: false, tension: 0 },
        { label: 'OB',  data: [{ x: firstTime, y: 70 }, { x: lastTime, y: 70 }],
          borderColor: '#ef535055', borderWidth: 1, borderDash: [4,4], pointRadius: 0, fill: false },
        { label: 'OS',  data: [{ x: firstTime, y: 30 }, { x: lastTime, y: 30 }],
          borderColor: '#26a69a55', borderWidth: 1, borderDash: [4,4], pointRadius: 0, fill: false },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false },
        tooltip: { backgroundColor: '#111318', borderColor: '#1f2535', borderWidth: 1,
          titleColor: '#e2e6f0', bodyColor: '#8892a8' } },
      scales: {
        x: { type: 'time', ticks: { color: '#8892a8', maxTicksLimit: 6, font: { size: 9 } },
             grid: { color: '#1a1f2e' } },
        y: { min: 0, max: 100, position: 'right',
             ticks: { color: '#8892a8', font: { size: 9 } }, grid: { color: '#1a1f2e' } },
      }
    }
  });
}

function drawOrderBlockAnnotations(obs, candles) {
  if (!obs.length || !candleChart) return;
  // We'll add OB labels to the chart title area as text — full annotation plugin would require extra lib
  // For now render as a subtle background band using afterDraw plugin
  const lastTime = candles.length ? candles[candles.length-1].time * 1000 : 0;
  obs.forEach(ob => {
    const color = ob.type === 'bullish' ? 'rgba(38,166,154,0.12)' : 'rgba(239,83,80,0.12)';
    const border = ob.type === 'bullish' ? 'rgba(38,166,154,0.6)' : 'rgba(239,83,80,0.6)';
    candleChart.data.datasets.push({
      label: ob.type === 'bullish' ? '● Demand' : '● Supply',
      data: [
        { x: ob.time * 1000, y: ob.top },
        { x: lastTime,       y: ob.top },
      ],
      borderColor: border, borderWidth: 1.5, borderDash: [3, 3],
      pointRadius: 0, fill: false, yAxisID: 'price', tension: 0,
    });
  });
  candleChart.update('none');
}

function updatePriceStrip(meta) {
  if (!meta) return;
  const priceEl  = document.getElementById('priceMain');
  const changeEl = document.getElementById('priceChange');
  if (document.getElementById('pairLabel')) document.getElementById('pairLabel').textContent = meta.label || '—';
  if (priceEl)  priceEl.textContent  = fmt(meta.last_price);
  const up = (meta.change_pct || 0) >= 0;
  if (changeEl) { changeEl.textContent = (up ? '+' : '') + (meta.change_pct || 0).toFixed(3) + '%'; changeEl.className = 'price-change ' + (up ? 'up' : 'down'); }
  if (priceEl)  priceEl.className = 'price-main ' + (up ? 'up' : 'down');
  const metaC = document.getElementById('metaC');
  if (metaC) metaC.textContent = fmt(meta.last_price);
}

function updateAnalysisPanel(meta, orderBlocks, signals) {
  if (!meta) return;
  const set = (id, val, cls) => { const el = document.getElementById(id); if (el) { el.textContent = val; if (cls) el.className = cls; } };
  set('biasBadge', (meta.bias||'neutral').toUpperCase(), 'badge ' + (meta.bias||'neutral'));
  set('rsiValue',  meta.last_rsi || '—');
  set('obCount',   orderBlocks.length + ' zones');
  set('sigCount',  signals.length + ' signals');
  const htfEl = document.getElementById('htfBadge');
  if (htfEl) { htfEl.textContent = (meta.htf_bias||'neutral').toUpperCase(); htfEl.className = 'badge ' + (meta.htf_bias||'neutral'); }
  const lastSig = document.getElementById('lastSignal');
  if (lastSig) {
    if (signals.length > 0) {
      const s = signals[signals.length-1];
      lastSig.textContent = s.type.toUpperCase() + ' @ ' + fmt(s.price) + (s.confidence ? ' · ' + s.confidence + '%' : '');
      lastSig.style.color = s.type === 'buy' ? '#26a69a' : '#ef5350';
    } else { lastSig.textContent = 'None'; lastSig.style.color = '#4a5470'; }
  }
  const slTpBlock = document.getElementById('slTpBlock');
  if (slTpBlock) {
    if (signals.length > 0) {
      const last = signals[signals.length-1];
      slTpBlock.style.display = 'block';
      set('slValue', fmt(last.sl)); set('tpValue', fmt(last.tp));
      set('rrValue', '1:' + (last.rr||2).toFixed(1));
    } else slTpBlock.style.display = 'none';
  }
}

function updateSummaryPanel(summary, signals) {
  if (!summary) return;
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val || '—'; };
  set('sumTrend', summary.trend); set('sumRsi', summary.rsi_desc);
  set('sumOb',    summary.ob_desc); set('sumSig', summary.sig_desc);
  set('sumRec',   summary.rec);
}

function fmt(val) {
  if (val === null || val === undefined) return '—';
  const n = parseFloat(val);
  if (isNaN(n)) return '—';
  if (currentSymbol === 'XAUUSD') return n.toFixed(2);
  if (currentSymbol === 'USDJPY') return n.toFixed(3);
  return n.toFixed(5);
}

function showLoading(show) { const el = document.getElementById('loadingOverlay'); if (el) el.classList.toggle('hidden', !show); }
function showError(msg)    { const el = document.getElementById('errorOverlay'); const m = document.getElementById('errorMsg'); if (m) m.textContent = msg; if (el) el.classList.remove('hidden'); }
function hideError()       { const el = document.getElementById('errorOverlay'); if (el) el.classList.add('hidden'); }
function setStatus(msg)    { const el = document.getElementById('statusText'); if (el) el.textContent = msg; }

function updateAiNarrative(text) {
  const el = document.getElementById('aiNarrative');
  if (!el) return;
  el.textContent = text || '—';
  el.style.whiteSpace = 'pre-line';
}
