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
let signalMarkers    = [];

/* ── Init ── */
document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  bindControls();
  loadChart();
});

function initCharts() {
  const container = document.getElementById('chart');

  chart = LightweightCharts.createChart(container, {
    width:  container.clientWidth,
    height: container.clientHeight,
    layout: { background: { color: '#0a0b0e' }, textColor: '#8892a8', fontFamily: "'Space Mono', monospace", fontSize: 11 },
    grid:   { vertLines: { color: '#1a1f2e', style: 1 }, horzLines: { color: '#1a1f2e', style: 1 } },
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

  /* RSI chart below */
  const rsiContainer = document.getElementById('rsiChart');
  rsiChart = LightweightCharts.createChart(rsiContainer, {
    width:  rsiContainer.clientWidth,
    height: rsiContainer.clientHeight,
    layout: { background: { color: '#0a0b0e' }, textColor: '#8892a8', fontFamily: "'Space Mono', monospace", fontSize: 10 },
    grid:   { vertLines: { color: '#1a1f2e', style: 1 }, horzLines: { color: '#1a1f2e', style: 1 } },
    rightPriceScale: { borderColor: '#1f2535', scaleMargins: { top: 0.1, bottom: 0.1 } },
    timeScale: { borderColor: '#1f2535', timeVisible: true, secondsVisible: false, rightOffset: 8, visible: false },
    crosshair: {
      vertLine: { color: '#f0b429', width: 1, style: 2, labelBackgroundColor: '#f0b429' },
      horzLine: { color: '#f0b429', width: 1, style: 2, labelBackgroundColor: '#f0b429' },
    },
  });

  rsiSeries = rsiChart.addLineSeries({ color: '#b39ddb', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: true });

  // RSI overbought/oversold lines
  rsiChart.addLineSeries({ color: '#ef535044', lineWidth: 1, priceLineVisible: false, lastValueVisible: false })
    .setData([]);
  rsiChart.addLineSeries({ color: '#26a69a44', lineWidth: 1, priceLineVisible: false, lastValueVisible: false })
    .setData([]);

  // Sync crosshair between charts
  chart.subscribeCrosshairMove(param => {
    if (param.time) rsiChart.setCrosshairPosition(param.point?.x ?? 0, param.point?.y ?? 0, rsiSeries);
    if (param.seriesData) {
      const bar = param.seriesData.get(candleSeries);
      if (bar) {
        document.getElementById('metaO').textContent = fmt(bar.open);
        document.getElementById('metaH').textContent = fmt(bar.high);
        document.getElementById('metaL').textContent = fmt(bar.low);
        document.getElementById('metaC').textContent = fmt(bar.close);
      }
    }
  });

  // Sync time scales
  chart.timeScale().subscribeVisibleTimeRangeChange(range => {
    if (range) rsiChart.timeScale().setVisibleRange(range);
  });

  const ro = new ResizeObserver(() => {
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    rsiChart.applyOptions({ width: rsiContainer.clientWidth, height: rsiContainer.clientHeight });
  });
  ro.observe(container);
  ro.observe(rsiContainer);
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
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Server error');

    renderCandles(data.candles);
    renderEMAs(data.ema_lines);
    renderRSI(data.rsi);
    renderOrderBlocks(data.order_blocks, data.candles);
    renderSignals(data.signals);
    updatePriceStrip(data.meta);
    updateAnalysisPanel(data.meta, data.order_blocks, data.signals);
    updateSummaryPanel(data.summary, data.signals);

    chart.timeScale().fitContent();
    rsiChart.timeScale().fitContent();

    setStatus(`Loaded ${data.meta.bars} bars · ${data.meta.label} · ${currentTimeframe.toUpperCase()} · Bias: ${data.meta.bias.toUpperCase()}`);
  } catch (err) {
    showError(err.message || 'Failed to fetch data');
    setStatus('Error — ' + err.message);
  } finally {
    showLoading(false);
    document.getElementById('refreshBtn').classList.remove('spinning');
  }
}

function renderCandles(candles) {
  candleSeries.setData(candles.map(c => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close })));
  volumeSeries.setData(candles.map(c => ({
    time: c.time, value: c.volume,
    color: c.close >= c.open ? 'rgba(38,166,154,0.35)' : 'rgba(239,83,80,0.35)',
  })));
}

function renderEMAs(emaLines) {
  ema20Series.setData(emaLines.ema20);
  ema50Series.setData(emaLines.ema50);
  ema200Series.setData(emaLines.ema200);
}

function renderRSI(rsiData) {
  rsiSeries.setData(rsiData);
  // Draw static 70/30 lines
  const ob = rsiChart.addLineSeries({ color: '#ef535055', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
  const os = rsiChart.addLineSeries({ color: '#26a69a55', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
  if (rsiData.length > 0) {
    const first = rsiData[0].time;
    const last  = rsiData[rsiData.length - 1].time;
    ob.setData([{ time: first, value: 70 }, { time: last, value: 70 }]);
    os.setData([{ time: first, value: 30 }, { time: last, value: 30 }]);
  }
}

function renderOrderBlocks(orderBlocks, candles) {
  // Remove old OB series
  obSeries.forEach(s => { try { chart.removeSeries(s); } catch(e){} });
  obSeries = [];

  const lastTime  = candles[candles.length - 1].time;
  const firstTime = candles[0].time;

  orderBlocks.forEach(ob => {
    const color   = ob.type === 'bullish' ? 'rgba(38,166,154,0.18)' : 'rgba(239,83,80,0.18)';
    const border  = ob.type === 'bullish' ? 'rgba(38,166,154,0.7)'  : 'rgba(239,83,80,0.7)';

    // Draw as a band using two lines with fill — approximate with area series
    const series = chart.addAreaSeries({
      topColor:     color,
      bottomColor:  color,
      lineColor:    border,
      lineWidth:    1,
      priceLineVisible: false,
      lastValueVisible: false,
      crossHairMarkerVisible: false,
    });

    series.setData([
      { time: ob.time,  value: ob.top    },
      { time: lastTime, value: ob.top    },
    ]);

    // Bottom border line
    const borderSeries = chart.addLineSeries({
      color: border, lineWidth: 1, lineStyle: 1,
      priceLineVisible: false, lastValueVisible: false, crossHairMarkerVisible: false,
    });
    borderSeries.setData([
      { time: ob.time,  value: ob.bottom },
      { time: lastTime, value: ob.bottom },
    ]);

    obSeries.push(series, borderSeries);
  });
}

function renderSignals(signals) {
  const markers = signals.map(s => ({
    time:     s.time,
    position: s.type === 'buy' ? 'belowBar' : 'aboveBar',
    color:    s.type === 'buy' ? '#26a69a'  : '#ef5350',
    shape:    s.type === 'buy' ? 'arrowUp'  : 'arrowDown',
    text:     s.type.toUpperCase() + ' RSI:' + s.rsi,
  }));
  candleSeries.setMarkers(markers);
}

function updatePriceStrip(meta) {
  const priceEl  = document.getElementById('priceMain');
  const changeEl = document.getElementById('priceChange');
  document.getElementById('pairLabel').textContent = meta.label;
  priceEl.textContent  = fmt(meta.last_price);
  const up = meta.change_pct >= 0;
  changeEl.textContent = (up ? '+' : '') + meta.change_pct.toFixed(3) + '%';
  changeEl.className   = 'price-change ' + (up ? 'up' : 'down');
  priceEl.className    = 'price-main '   + (up ? 'up' : 'down');
  document.getElementById('metaC').textContent = fmt(meta.last_price);
  document.getElementById('metaO').textContent = '—';
  document.getElementById('metaH').textContent = '—';
  document.getElementById('metaL').textContent = '—';
}

function updateAnalysisPanel(meta, orderBlocks, signals) {
  const biasEl  = document.getElementById('biasBadge');
  const rsiEl   = document.getElementById('rsiValue');
  const obEl    = document.getElementById('obCount');
  const sigEl   = document.getElementById('sigCount');
  const lastSig = document.getElementById('lastSignal');

  biasEl.textContent  = meta.bias.toUpperCase();
  biasEl.className    = 'badge ' + meta.bias;
  const htfEl = document.getElementById('htfBadge');
  if (htfEl) { htfEl.textContent = (data.htf_bias || 'neutral').toUpperCase(); htfEl.className = 'badge ' + (data.htf_bias || 'neutral'); }
  rsiEl.textContent   = meta.last_rsi;
  rsiEl.style.color   = meta.last_rsi > 70 ? '#ef5350' : meta.last_rsi < 30 ? '#26a69a' : '#e2e6f0';
  obEl.textContent    = orderBlocks.length + ' zones';
  sigEl.textContent   = signals.length + ' signals';

  if (signals.length > 0) {
    const s = signals[signals.length - 1];
    lastSig.textContent = s.type.toUpperCase() + ' @ ' + fmt(s.price);
    lastSig.style.color = s.type === 'buy' ? '#26a69a' : '#ef5350';
  } else {
    lastSig.textContent = 'None';
    lastSig.style.color = '#4a5470';
  }
}

function fmt(val) {
  if (val === null || val === undefined) return '—';
  const n = parseFloat(val);
  if (currentSymbol === 'XAUUSD') return n.toFixed(2);
  if (currentSymbol === 'USDJPY') return n.toFixed(3);
  return n.toFixed(5);
}

function showLoading(show) { document.getElementById('loadingOverlay').classList.toggle('hidden', !show); }
function showError(msg)    { document.getElementById('errorMsg').textContent = msg; document.getElementById('errorOverlay').classList.remove('hidden'); }
function hideError()       { document.getElementById('errorOverlay').classList.add('hidden'); }
function setStatus(msg)    { document.getElementById('statusText').textContent = msg; }

/* ── Phase 3: Summary + SL/TP panel ── */
function updateSummaryPanel(summary, signals) {
  document.getElementById('sumTrend').textContent = summary.trend;
  document.getElementById('sumRsi').textContent   = summary.rsi_desc;
  document.getElementById('sumOb').textContent    = summary.ob_desc;
  document.getElementById('sumSig').textContent   = summary.sig_desc;
  document.getElementById('sumRec').textContent   = summary.rec;

  const slTpBlock = document.getElementById('slTpBlock');
  if (signals && signals.length > 0) {
    const last = signals[signals.length - 1];
    slTpBlock.style.display = 'block';
    document.getElementById('slValue').textContent = fmt(last.sl);
    document.getElementById('tpValue').textContent = fmt(last.tp);
    document.getElementById('rrValue').textContent = '1:' + last.rr.toFixed(1);
  } else {
    slTpBlock.style.display = 'none';
  }
}
