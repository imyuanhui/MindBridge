const CHAT_API = import.meta.env.VITE_API_BASE_URL;

const messagesEl = document.getElementById('chat-messages');
const inputEl = document.getElementById('chat-input');
const sendBtn = document.getElementById('chat-send');
const workloadEl = document.getElementById('chat-workload');
const workloadInputEl = document.getElementById('chat-workload-input');

function appendMessage(role, text, workload, source) {
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  let html = text;
  if (role === 'assistant' && workload != null) {
    const sourceLabel =
      source === 'user'
        ? ' (user override)'
        : source === 'engine'
        ? ' (brain engine)'
        : '';
    html += `<div class="workload">Workload: ${workload.toFixed(
      1,
    )}${sourceLabel}</div>`;
  }
  div.innerHTML = html;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setWorkloadText(text) {
  workloadEl.textContent = text;
  workloadEl.classList.toggle('empty', !text);
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  appendMessage('user', text);
  inputEl.value = '';
  sendBtn.disabled = true;
  setWorkloadText('Thinking…');

  const body = { user_message: text };
  const rawOverride = workloadInputEl?.value.trim();
  if (rawOverride) {
    const w = Number(rawOverride);
    if (!Number.isNaN(w)) {
      // Clamp to [1, 9] on the client as well
      body.user_workload = Math.max(1, Math.min(9, w));
    }
  }

  try {
    const res = await fetch(CHAT_API, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();

    if (!res.ok) {
      appendMessage('assistant', `Error: ${data.detail || res.statusText}`, null);
      setWorkloadText('');
      return;
    }

    const workload = data.workload_detected ?? null;
    const source = data.workload_source ?? null;
    const aiText = data.ai_response ?? 'No response.';
    appendMessage('assistant', aiText, workload, source);
    if (workload != null) {
      const label =
        source === 'user'
          ? 'User override workload'
          : source === 'engine'
          ? 'Engine workload'
          : 'Workload';
      setWorkloadText(`${label}: ${workload.toFixed(1)}`);
    } else {
      setWorkloadText('');
    }
  } catch (err) {
    appendMessage(
      'assistant',
      `Error: ${err.message}. Is the chat API running on port 8000?`,
      null,
    );
    setWorkloadText('');
  } finally {
    sendBtn.disabled = false;
  }
}

export function initChat() {
  sendBtn.addEventListener('click', sendMessage);
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
}
