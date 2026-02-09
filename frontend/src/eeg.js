import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

const WS_URL = 'ws://localhost:8001/ws/eeg';
const CHANNELS = 14;

let ws = null;
let chart = null;

const COLORS = [
  '#22c55e',
  '#0ea5e9',
  '#f97316',
  '#e11d48',
  '#a855f7',
  '#14b8a6',
  '#facc15',
  '#6366f1',
  '#2dd4bf',
  '#f43f5e',
  '#4ade80',
  '#38bdf8',
  '#fb923c',
  '#c4b5fd',
];

const statusEl = document.getElementById('eeg-status');
const connectBtn = document.getElementById('eeg-connect');
const disconnectBtn = document.getElementById('eeg-disconnect');
const canvas = document.getElementById('eeg-chart');

function createChart() {
  if (chart) chart.destroy();
  const ctx = canvas.getContext('2d');
  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: Array.from({ length: CHANNELS }, (_, i) => `Ch ${i + 1}`),
      datasets: [
        {
          label: 'Amplitude',
          data: Array.from({ length: CHANNELS }, () => 0),
          backgroundColor: COLORS.slice(0, CHANNELS),
          borderColor: COLORS.slice(0, CHANNELS),
          borderWidth: 1,
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
          },
        },
        y: {
          display: true,
          grid: { color: 'rgba(42, 49, 66, 0.6)' },
          ticks: { color: '#8b92a8' },
          title: {
            display: true,
            text: 'Amplitude (a.u.)',
            color: '#8b92a8',
          },
        },
      },
      plugins: {
        legend: {
          display: false,
        },
      },
    },
  });
}

function updateAmplitudes(channelsData) {
  if (!Array.isArray(channelsData) || channelsData.length === 0 || !chart) return;

  const amps = [];
  const numChannels = Math.min(CHANNELS, channelsData.length);

  for (let ch = 0; ch < numChannels; ch++) {
    const samples = channelsData[ch];
    if (!Array.isArray(samples) || samples.length === 0) {
      amps.push(0);
      continue;
    }
    // Root-mean-square amplitude per channel over the latest frame
    let sumSq = 0;
    for (let i = 0; i < samples.length; i++) {
      const v = Number(samples[i]) || 0;
      sumSq += v * v;
    }
    amps.push(Math.sqrt(sumSq / samples.length));
  }

  while (amps.length < CHANNELS) {
    amps.push(0);
  }

  chart.data.datasets[0].data = amps;
  chart.update('none');
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
        updateAmplitudes(msg.channels);
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
  createChart();
  connectBtn.addEventListener('click', connect);
  disconnectBtn.addEventListener('click', disconnect);
}
