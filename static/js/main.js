/* ── State ── */
let currentSymbol    = 'EURUSD';
let currentTimeframe = '1h';
let chart            = null;
let candleSeries     = null;
let volumeSeries     = null;
let ema20Series      = null;
let ema50Series      = null;
let ema200Series     = null;
let rsiChart         = null;
let rsiSeries        = null;
let obSeries         = [];

/* ── Init ── */
window.addEventListener('load', () => {
  try {
    initCharts();
    bindControls();
    loadChart();
  } catch(e) {
    console.error('Init error:', e);
    setStatus('Init error: ' + e.message);
  }
});

function initCharts() {
  const container = document.getElementById('chart');
  if (!container) throw new Error('Chart container not found');

  const w = container.offsetWidth  || window.innerWidth  || 800;
  const h = container.offsetHeight || window.innerHeight - 250 || 500;

  chart = LightweightCharts.createChart(container, {
    width:  w,
    height: h,
    layout: {
      background:  { color: '#0a0b0e' },
      textColor:   '#8892a8',
      fontFamily:  "'Space Mono', monospace",
      fontSize:    11,
    },
    grid: {
      vertLines: { color: '#1a1f2e', style: 1 },
      horzLines: { color: '#1a1f2e', style: 1 },
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
      vertLine: { color: '#f0b429', width: 1, style: 2, labelBackgroundColor: '#f0b429' },
      horzLine: { color: '#f0b429', width: 1, style: 2, labelBackgroundColor: '#f0b429' },
    },
    rightPriceScale: { borderColor: '#1f2535', scaleMargins: { top: 0.08, bottom: 0.28 } },
    timeScale: { borderColor: '#1f2535', timeVisible: true, secondsVisible: false, rightOffset: 8 },
  });

  candleSeries = chart.addCandlestickSeries({
    upColor: '#26a69a', downColor: '#ef5350',
    borderUpColor: '#26a69a', borderDownColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
  });

  volumeSeries = chart.addHistogramSeries({
    priceFormat: { type: 'volume' }, priceScaleId: 'volume',
  });
  chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });

  ema20Series  = chart.addLineSeries({ color: '#f0b429', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
  ema50Series  = chart.addLineSeries({ color: '#4fc3f7', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
  ema200Series = chart.addLineSeries({ color: '#ef5350', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });

  /* RSI chart */
  const rsiContainer = document.getElementById('rsiChart');
  if (rsiContainer) {
    const rw = rsiContainer.offsetWidth || w;
    rsiChart = LightweightCharts.createChart(rsiContainer, {
      width:  rw,
      height: 120,
      layout: { background: { color: '#0a0b0e' }, textColor: '#8892a8', fontFamily: "'Space Mono', monospace", fontSize: 10 },
      grid:   { vertLines: { color: '#1a1f2e', style: 1 }, horzLines: { color: '#1a1f2e', style: 1 } },
      rightPriceScale: { borderColor: '#1f2535', scaleMargins: { top: 0.1, bottom: 0.1 } },
      timeScale: { borderColor: '#1f2535', timeVisible: true, secondsVisible: false, visible: false },
      crosshair: {
        vertLine: { color: '#f0b429', width: 1, style: 2, labelBackgroundColor: '#f0b429' },
        horzLine: { color: '#f0b429', width: 1, style: 2, labelBackgroundColor: '#f0b429' },
      },
    });
    rsiSeries = rsiChart.addLineSeries({ color: '#b39ddb', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: true });
  }

  /* Crosshair OHLC */
  chart.subscribeCrosshairMove(param => {
    if (!param || !param.seriesData) return;
    const bar = param.seriesData.get(candleSeries);
    if (bar) {
      document.getElementById('metaO').textContent = fmt(bar.open);
      document.getElementById('metaH').textContent = fmt(bar.high);
      document.getElementById('metaL').textContent = fmt(bar.low);
      document.getElementById('metaC').textContent = fmt(bar.close);
    }
  });

  /* Sync time scales */
  if (rsiChart) {
    chart.timeScale().subscribeVisibleTimeRangeChange(range => {
      if (range) rsiChart.timeScale().setVisibleRange(range);
    });
  }

  /* Resize */
  const ro = new ResizeObserver(() => {
    if (chart && container) {
      chart.applyOptions({ width: container.offsetWidth, height: container.offsetHeight });
    }
    if (rsiChart && rsiContainer) {
      rsiChart.applyOptions({ width: rsiContainer.offsetWidth });
    }
  });
  ro.observe(container);
  if (rsiContainer) ro.observe(rsiContainer);
}

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
    try {
      data = JSON.parse(text);
    } catch(e) {
      throw new Error('Server returned invalid response. Please refresh.');
    }

    if (!res.ok || data.error) throw new Error(data.error || 'Server error');
    if (!data.candles || !data.meta) throw new Error('Incomplete data from server');

    renderCandles(data.candles);
    renderEMAs(data.ema_lines || { ema20: [], ema50: [], ema200: [] });
    renderRSI(data.rsi || []);
    renderOrderBlocks(data.order_blocks || [], data.candles);
    renderSignals(data.signals || []);
    updatePriceStrip(data.meta);
    updateAnalysisPanel(data.meta, data.order_blocks || [], data.signals || []);
    updateSummaryPanel(data.summary || {}, data.signals || []);

    chart.timeScale().fitContent();
    if (rsiChart) rsiChart.timeScale().fitContent();

    setStatus(`Loaded ${data.meta.bars} bars · ${data.meta.label} · ${currentTimeframe.toUpperCase()} · Bias: ${(data.meta.bias || '—').toUpperCase()}`);
  } catch (err) {
    showError(err.message || 'Failed to fetch data');
    setStatus('Error — ' + err.message);
  } finally {
    showLoading(false);
    document.getElementById('refreshBtn').classList.remove('spinning');
  }
}

function renderCandles(candles) {
  if (!candleSeries || !candles.length) return;
  candleSeries.setData(candles.map(c => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close })));
  if (volumeSeries) {
    volumeSeries.setData(candles.map(c => ({
      time: c.time, value: c.volume,
      color: c.close >= c.open ? 'rgba(38,166,154,0.35)' : 'rgba(239,83,80,0.35)',
    })));
  }
}

function renderEMAs(emaLines) {
  if (!emaLines) return;
  if (ema20Series  && emaLines.ema20)  ema20Series.setData(emaLines.ema20);
  if (ema50Series  && emaLines.ema50)  ema50Series.setData(emaLines.ema50);
  if (ema200Series && emaLines.ema200) ema200Series.setData(emaLines.ema200);
}

function renderRSI(rsiData) {
  if (!rsiSeries || !rsiData || !rsiData.length) return;
  rsiSeries.setData(rsiData);
  if (rsiChart && rsiData.length > 1) {
    const first = rsiData[0].time;
    const last  = rsiData[rsiData.length - 1].time;
    try {
      const ob = rsiChart.addLineSeries({ color: '#ef535055', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
      const os = rsiChart.addLineSeries({ color: '#26a69a55', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
      ob.setData([{ time: first, value: 70 }, { time: last, value: 70 }]);
      os.setData([{ time: first, value: 30 }, { time: last, value: 30 }]);
    } catch(e) {}
  }
}

function renderOrderBlocks(orderBlocks, candles) {
  obSeries.forEach(s => { try { chart.removeSeries(s); } catch(e){} });
  obSeries = [];
  if (!orderBlocks.length || !candles.length) return;

  const lastTime = candles[candles.length - 1].time;

  orderBlocks.forEach(ob => {
    try {
      const color  = ob.type === 'bullish' ? 'rgba(38,166,154,0.18)' : 'rgba(239,83,80,0.18)';
      const border = ob.type === 'bullish' ? 'rgba(38,166,154,0.7)'  : 'rgba(239,83,80,0.7)';

      const s = chart.addAreaSeries({
        topColor: color, bottomColor: color, lineColor: border, lineWidth: 1,
        priceLineVisible: false, lastValueVisible: false, crossHairMarkerVisible: false,
      });
      s.setData([{ time: ob.time, value: ob.top }, { time: lastTime, value: ob.top }]);

      const b = chart.addLineSeries({
        color: border, lineWidth: 1, lineStyle: 1,
        priceLineVisible: false, lastValueVisible: false, crossHairMarkerVisible: false,
      });
      b.setData([{ time: ob.time, value: ob.bottom }, { time: lastTime, value: ob.bottom }]);

      obSeries.push(s, b);
    } catch(e) {}
  });
}

function renderSignals(signals) {
  if (!candleSeries || !signals) return;
  const markers = signals.map(s => ({
    time:     s.time,
    position: s.type === 'buy' ? 'belowBar' : 'aboveBar',
    color:    s.type === 'buy' ? '#26a69a'  : '#ef5350',
    shape:    s.type === 'buy' ? 'arrowUp'  : 'arrowDown',
    text:     s.type.toUpperCase() + ' RSI:' + s.rsi,
  }));
  try { candleSeries.setMarkers(markers); } catch(e) {}
}

function updatePriceStrip(meta) {
  if (!meta) return;
  const priceEl  = document.getElementById('priceMain');
  const changeEl = document.getElementById('priceChange');
  document.getElementById('pairLabel').textContent = meta.label || '—';
  priceEl.textContent  = fmt(meta.last_price);
  const up = (meta.change_pct || 0) >= 0;
  changeEl.textContent = (up ? '+' : '') + (meta.change_pct || 0).toFixed(3) + '%';
  changeEl.className   = 'price-change ' + (up ? 'up' : 'down');
  priceEl.className    = 'price-main '   + (up ? 'up' : 'down');
  document.getElementById('metaC').textContent = fmt(meta.last_price);
  document.getElementById('metaO').textContent = '—';
  document.getElementById('metaH').textContent = '—';
  document.getElementById('metaL').textContent = '—';
}

function updateAnalysisPanel(meta, orderBlocks, signals) {
  if (!meta) return;
  const biasEl = document.getElementById('biasBadge');
  const rsiEl  = document.getElementById('rsiValue');
  const obEl   = document.getElementById('obCount');
  const sigEl  = document.getElementById('sigCount');
  const lastSig = document.getElementById('lastSignal');
  const htfEl  = document.getElementById('htfBadge');

  if (biasEl) { biasEl.textContent = (meta.bias || 'neutral').toUpperCase(); biasEl.className = 'badge ' + (meta.bias || 'neutral'); }
  if (htfEl)  { htfEl.textContent  = '—'; }
  if (rsiEl)  { rsiEl.textContent  = meta.last_rsi || '—'; rsiEl.style.color = meta.last_rsi > 70 ? '#ef5350' : meta.last_rsi < 30 ? '#26a69a' : '#e2e6f0'; }
  if (obEl)   { obEl.textContent   = (orderBlocks.length) + ' zones'; }
  if (sigEl)  { sigEl.textContent  = (signals.length) + ' signals'; }

  const htfData = window._lastHtf || 'neutral';
  if (htfEl)  { htfEl.textContent = htfData.toUpperCase(); htfEl.className = 'badge ' + htfData; }

  if (lastSig) {
    if (signals.length > 0) {
      const s = signals[signals.length - 1];
      lastSig.textContent = s.type.toUpperCase() + ' @ ' + fmt(s.price);
      lastSig.style.color = s.type === 'buy' ? '#26a69a' : '#ef5350';
    } else {
      lastSig.textContent = 'None';
      lastSig.style.color = '#4a5470';
    }
  }

  const slTpBlock = document.getElementById('slTpBlock');
  if (slTpBlock) {
    if (signals.length > 0) {
      const last = signals[signals.length - 1];
      slTpBlock.style.display = 'block';
      const slEl = document.getElementById('slValue');
      const tpEl = document.getElementById('tpValue');
      const rrEl = document.getElementById('rrValue');
      if (slEl) slEl.textContent = fmt(last.sl);
      if (tpEl) tpEl.textContent = fmt(last.tp);
      if (rrEl) rrEl.textContent = '1:' + (last.rr || 2).toFixed(1);
    } else {
      slTpBlock.style.display = 'none';
    }
  }
}

function updateSummaryPanel(summary, signals) {
  if (!summary) return;
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val || '—'; };
  set('sumTrend', summary.trend);
  set('sumRsi',   summary.rsi_desc);
  set('sumOb',    summary.ob_desc);
  set('sumSig',   summary.sig_desc);
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

function showLoading(show) {
  const el = document.getElementById('loadingOverlay');
  if (el) el.classList.toggle('hidden', !show);
}
function showError(msg) {
  const el  = document.getElementById('errorOverlay');
  const msg_el = document.getElementById('errorMsg');
  if (msg_el) msg_el.textContent = msg;
  if (el) el.classList.remove('hidden');
}
function hideError() {
  const el = document.getElementById('errorOverlay');
  if (el) el.classList.add('hidden');
}
function setStatus(msg) {
  const el = document.getElementById('statusText');
  if (el) el.textContent = msg;
}

/* ── Phase 3: Summary + SL/TP panel ── */
function updateSummaryPanel(summary, signals) {
  if (!summary) return;
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val || '—'; };
  set('sumTrend', summary.trend);
  set('sumRsi',   summary.rsi_desc);
  set('sumOb',    summary.ob_desc);
  set('sumSig',   summary.sig_desc);
  set('sumRec',   summary.rec);

  const slTpBlock = document.getElementById('slTpBlock');
  if (slTpBlock) {
    if (signals && signals.length > 0) {
      const last = signals[signals.length - 1];
      slTpBlock.style.display = 'block';
      const slEl = document.getElementById('slValue');
      const tpEl = document.getElementById('tpValue');
      const rrEl = document.getElementById('rrValue');
      if (slEl) slEl.textContent = fmt(last.sl);
      if (tpEl) tpEl.textContent = fmt(last.tp);
      if (rrEl) rrEl.textContent = '1:' + (last.rr || 2).toFixed(1);
    } else {
      slTpBlock.style.display = 'none';
    }
  }
}
