import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

const WS_URL = 'ws://localhost:8001/ws/eeg';
const MAX_POINTS = 512;
const CHANNELS = 14;

let ws = null;
let chart = null;
let buffer = [];
let selectedChannel = 0;

const statusEl = document.getElementById('eeg-status');
const connectBtn = document.getElementById('eeg-connect');
const disconnectBtn = document.getElementById('eeg-disconnect');
const channelSelect = document.getElementById('eeg-channel');
const canvas = document.getElementById('eeg-chart');

function fillChannelSelect() {
  channelSelect.innerHTML = '';
  for (let i = 0; i < CHANNELS; i++) {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `Ch ${i + 1}`;
    channelSelect.appendChild(opt);
  }
}

function createChart() {
  if (chart) chart.destroy();
  const ctx = canvas.getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: `Channel ${selectedChannel + 1}`,
          data: [],
          borderColor: '#6ee7b7',
          backgroundColor: 'rgba(110, 231, 183, 0.08)',
          borderWidth: 1.5,
          fill: true,
          tension: 0,
          pointRadius: 0,
          pointHoverRadius: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: {
          display: true,
          grid: { color: 'rgba(42, 49, 66, 0.6)' },
          ticks: {
            color: '#8b92a8',
            maxTicksLimit: 10,
            callback(value) {
              return Number(value) != null ? (Number(value) / 128).toFixed(1) + 's' : '';
            },
          },
        },
        y: {
          display: true,
          grid: { color: 'rgba(42, 49, 66, 0.6)' },
          ticks: { color: '#8b92a8' },
        },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });
}

function pushToBuffer(channelsData) {
  const samples = channelsData[selectedChannel];
  if (!samples || !Array.isArray(samples)) return;
  const base = buffer.length;
  samples.forEach((v, i) => {
    buffer.push({ x: base + i, y: v });
  });
  if (buffer.length > MAX_POINTS) {
    buffer = buffer.slice(-MAX_POINTS);
  }
  if (chart?.data?.datasets?.[0]) {
    chart.data.datasets[0].data = buffer;
    chart.data.datasets[0].label = `Channel ${selectedChannel + 1}`;
    chart.update('none');
  }
}

function connect() {
  if (ws?.readyState === WebSocket.OPEN) return;
  statusEl.textContent = 'Connecting…';
  statusEl.classList.remove('connected');
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    statusEl.textContent = 'Connected';
    statusEl.classList.add('connected');
    connectBtn.disabled = true;
    disconnectBtn.disabled = false;
    ws.send('start');
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.channels && Array.isArray(msg.channels)) {
        pushToBuffer(msg.channels);
      }
    } catch (_) {}
  };

  ws.onerror = () => {
    statusEl.textContent = 'Error';
    statusEl.classList.remove('connected');
  };

  ws.onclose = () => {
    statusEl.textContent = 'Disconnected';
    statusEl.classList.remove('connected');
    connectBtn.disabled = false;
    disconnectBtn.disabled = true;
    ws = null;
  };
}

function disconnect() {
  if (ws) {
    ws.close();
    ws = null;
  }
}

export function initEeg() {
  fillChannelSelect();
  createChart();

  channelSelect.addEventListener('change', () => {
    selectedChannel = parseInt(channelSelect.value, 10);
    buffer = [];
    if (chart?.data?.datasets?.[0]) {
      chart.data.datasets[0].data = [];
      chart.data.datasets[0].label = `Channel ${selectedChannel + 1}`;
      chart.update('none');
    }
  });

  connectBtn.addEventListener('click', connect);
  disconnectBtn.addEventListener('click', disconnect);
}
