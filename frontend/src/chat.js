const CHAT_API = 'http://localhost:8000/chat';

const messagesEl = document.getElementById('chat-messages');
const inputEl = document.getElementById('chat-input');
const sendBtn = document.getElementById('chat-send');
const workloadEl = document.getElementById('chat-workload');

function appendMessage(role, text, workload) {
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  let html = text;
  if (role === 'assistant' && workload != null) {
    html += `<div class="workload">Workload: ${workload.toFixed(1)}</div>`;
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

  try {
    const res = await fetch(CHAT_API, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_message: text }),
    });
    const data = await res.json();

    if (!res.ok) {
      appendMessage('assistant', `Error: ${data.detail || res.statusText}`, null);
      setWorkloadText('');
      return;
    }

    const workload = data.workload_detected ?? null;
    const aiText = data.ai_response ?? 'No response.';
    appendMessage('assistant', aiText, workload);
    setWorkloadText(workload != null ? `Last response workload: ${workload.toFixed(1)}` : '');
  } catch (err) {
    appendMessage('assistant', `Error: ${err.message}. Is the chat API running on port 8000?`, null);
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
