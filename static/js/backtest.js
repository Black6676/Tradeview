let btSymbol    = 'EURUSD';
let btTimeframe = '1d';
let equityChart = null;
let equitySeries = null;

document.addEventListener('DOMContentLoaded', () => {
  bindControls();
  initEquityChart();
});

function bindControls() {
  document.getElementById('btInstrumentTabs').addEventListener('click', e => {
    const btn = e.target.closest('.inst-btn');
    if (!btn) return;
    document.querySelectorAll('#btInstrumentTabs .inst-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    btSymbol = btn.dataset.symbol;
  });

  document.getElementById('btTfTabs').addEventListener('click', e => {
    const btn = e.target.closest('.tf-btn');
    if (!btn) return;
    document.querySelectorAll('#btTfTabs .tf-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    btTimeframe = btn.dataset.tf;
  });

  document.getElementById('runBtn').addEventListener('click', runBacktest);
}

function initEquityChart() {
  const container = document.getElementById('equityChart');
  equityChart = LightweightCharts.createChart(container, {
    width:  container.clientWidth,
    height: 220,
    layout: { background: { color: '#111318' }, textColor: '#8892a8', fontFamily: "'Space Mono', monospace", fontSize: 11 },
    grid:   { vertLines: { color: '#1a1f2e' }, horzLines: { color: '#1a1f2e' } },
    rightPriceScale: { borderColor: '#1f2535' },
    timeScale: { borderColor: '#1f2535', timeVisible: true },
    crosshair: {
      vertLine: { color: '#f0b429', labelBackgroundColor: '#f0b429' },
      horzLine: { color: '#f0b429', labelBackgroundColor: '#f0b429' },
    },
  });

  equitySeries = equityChart.addAreaSeries({
    topColor:    'rgba(38,166,154,0.3)',
    bottomColor: 'rgba(38,166,154,0.02)',
    lineColor:   '#26a69a',
    lineWidth:   2,
    priceLineVisible: false,
  });

  new ResizeObserver(() => {
    equityChart.applyOptions({ width: container.clientWidth });
  }).observe(container);
}

async function runBacktest() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.textContent = 'Running…';

  show('btLoading');
  hide('btEmpty');
  hide('btResults');
  hide('btError');
  setStatus('Fetching historical data and running backtest…');

  try {
    const res  = await fetch(`/api/backtest?symbol=${btSymbol}&timeframe=${btTimeframe}`);
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Server error');

    renderStats(data.stats);
    renderEquity(data.equity);
    renderTradeLog(data.trades);

    show('btResults');
    hide('btLoading');
    setStatus(`Backtest complete — ${data.stats.total_trades} trades · Win rate: ${data.stats.win_rate}% · Total R: ${data.stats.total_r}`);

  } catch (err) {
    document.getElementById('btErrorMsg').textContent = err.message;
    show('btError');
    hide('btLoading');
    setStatus('Error — ' + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Backtest';
  }
}

function renderStats(s) {
  const totalR = s.total_r;
  document.getElementById('stTotal').textContent  = s.total_trades;
  document.getElementById('stWinRate').textContent = s.win_rate + '%';
  document.getElementById('stTotalR').textContent  = (totalR >= 0 ? '+' : '') + totalR + 'R';
  document.getElementById('stPF').textContent      = s.profit_factor;
  document.getElementById('stDD').textContent      = '-' + s.max_drawdown + 'R';
  document.getElementById('stWL').textContent      = s.wins + ' / ' + s.losses;

  const totalREl = document.getElementById('stTotalR');
  totalREl.className = 'stat-value ' + (totalR >= 0 ? 'green' : 'red');

  const pfEl = document.getElementById('stPF');
  pfEl.className = 'stat-value ' + (s.profit_factor >= 1.5 ? 'green' : s.profit_factor >= 1 ? '' : 'red');
}

function renderEquity(equity) {
  if (!equity || equity.length === 0) return;
  equitySeries.setData(equity.map(e => ({ time: e.time, value: e.value })));
  equityChart.timeScale().fitContent();

  // Colour area red if ending negative
  const last = equity[equity.length - 1].value;
  equitySeries.applyOptions(last >= 0
    ? { topColor: 'rgba(38,166,154,0.3)', bottomColor: 'rgba(38,166,154,0.02)', lineColor: '#26a69a' }
    : { topColor: 'rgba(239,83,80,0.3)',  bottomColor: 'rgba(239,83,80,0.02)',  lineColor: '#ef5350' }
  );
}

function renderTradeLog(trades) {
  const tbody = document.getElementById('tradeTableBody');
  tbody.innerHTML = '';

  trades.forEach((t, i) => {
    const tr = document.createElement('tr');
    tr.className = t.result;

    const pnlClass  = t.result === 'win' ? 'pnl-win' : t.result === 'loss' ? 'pnl-loss' : 'pnl-open';
    const pnlText   = t.result === 'open' ? 'Open' : (t.pnl_r >= 0 ? '+' : '') + t.pnl_r + 'R';
    const entryDate = new Date(t.entry_time * 1000).toLocaleDateString();
    const exitDate  = t.exit_time ? new Date(t.exit_time * 1000).toLocaleDateString() : '—';

    tr.innerHTML = `
      <td>${i + 1}</td>
      <td class="${t.type === 'buy' ? 'badge-buy' : 'badge-sell'}">${t.type.toUpperCase()}</td>
      <td>${t.entry_price}</td>
      <td>${t.exit_price ?? '—'}</td>
      <td>${t.sl}</td>
      <td>${t.tp}</td>
      <td>${t.rsi}</td>
      <td>${t.result.toUpperCase()}</td>
      <td class="${pnlClass}">${pnlText}</td>
    `;
    tbody.appendChild(tr);
  });
}

function show(id) { document.getElementById(id).classList.remove('hidden'); }
function hide(id) { document.getElementById(id).classList.add('hidden'); }
function setStatus(msg) { document.getElementById('btStatus').textContent = msg; }
