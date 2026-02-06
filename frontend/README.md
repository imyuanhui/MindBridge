# MindBridge Frontend

Small dashboard to view **real-time EEG** from the simulator and **chat** with the neuro-aware API.

## Run

1. Start backend services (simulator on 8001, chat-api on 8000):

   ```bash
   cd backend && docker-compose up -d
   ```

2. Install and run the frontend:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. Open http://localhost:5173

## Usage

- **EEG**: Click **Connect** to stream from `ws://localhost:8001/ws/eeg`. Use the channel dropdown to view any of the 14 channels. Click **Disconnect** to stop.
- **Chat**: Type a message and press Enter or **Send**. Responses show the current workload score from the brain-engine.

## Config

- EEG WebSocket: `ws://localhost:8001/ws/eeg` (set in `src/eeg.js`)
- Chat API: `http://localhost:8000/chat` (set in `src/chat.js`)
