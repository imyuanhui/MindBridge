## MindBridge

Neuro-aware demo stack that streams **simulated EEG data**, runs a **KNN-based brain engine**, and exposes a **Gemini-powered chat API** with workload awareness, all wrapped in a small web UI.

This repo contains:
- **frontend**: Vite app for EEG visualization and chat
- **backend/brain-engine**: KNN “workload” estimator consuming EEG features from Redis
- **backend/chat-api**: HTTP API that talks to Gemini and annotates replies with workload
- **backend/simulator**: Real-time EEG signal simulator pushing data to WebSocket and Redis
- **cache**: Redis instance shared by simulator and brain-engine

---

## Prerequisites

- **Docker** and **Docker Compose** installed (`docker compose` CLI works)
- A **Google Gemini API key**

---

## 1. Configure your environment

In the repo root, create a `.env` file (or edit the existing one) with your Gemini key:

```bash
GEMINI_API_KEY=your-real-api-key-here
```

Docker Compose will inject `GEMINI_API_KEY` into the `chat-api` service.

---

## 2. Start all services with Docker

From the repo root (`MindBridge`):

```bash
docker compose up --build
```

This will start:
- **cache** (Redis) on port **6379**
- **brain-engine** (no public port; internal service)
- **chat-api** on `http://localhost:8000`
- **simulator** on `http://localhost:8001`
- **frontend** on `http://localhost:5173`

When all containers are healthy, open the UI:

```text
http://localhost:5173
```

You can stop everything with:

```bash
docker compose down
```

---

## 3. Exploring the features

### EEG page

- Go to the **EEG** tab in the top navigation.
- Click **Connect** to open a WebSocket to the simulator at:
  - `ws://localhost:8001/ws/eeg`
- You should see a live EEG trace streaming from the simulator at 128 Hz across 14 channels.
- Click **Disconnect** to stop streaming.

Under the hood:
- The **simulator** generates synthetic EEG, streams raw samples over WebSocket to the frontend, and writes extracted features into Redis.
- The **brain-engine** reads features from Redis and predicts a workload score.

### Chat page

- Switch to the **Chat** tab.
- Type a message in the input box and press **Send** (or press Enter).
- The frontend posts to:
  - `POST http://localhost:8000/chat`
- The assistant’s reply will include:
  - A **workload score** (1–9)
  - A label for the source (e.g. “brain engine” or “user override”)

**Workload override:**

- In the **Workload override (1–9, optional)** field, you can manually set a workload value.
- If provided, the value is sent as `user_workload` and is clamped to \[1, 9\] on the client.
  - Low wordload: 1-6.5
  - Medium workload: 6.5-8.5
  - High workload: >= 8.5
- The UI clearly marks responses that used the override vs the engine’s estimate.

---

## 4. Service overview

- **cache**: Redis store for EEG feature frames (`cache-data` named volume).
- **brain-engine**:
  - Reads feature frames from Redis.
  - Predicts workload and serves it to the chat API.
- **chat-api**:
  - Accepts `user_message` (+ optional `user_workload`) on `/chat`.
  - Calls Gemini using `GEMINI_API_KEY`.
  - Returns `ai_response`, `workload_detected`, and `workload_source`.
- **simulator**:
  - Serves WebSocket EEG stream on `/ws/eeg`.
  - Exposes health endpoint on `/health`.
- **frontend**:
  - Single-page app with **EEG** and **Chat** views.

