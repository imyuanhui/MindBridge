## MindBridge (Production)

Neuro-aware demo stack that streams **simulated EEG data**, runs a **KNN-based brain engine**, and exposes a **Gemini-powered chat API** with workload awareness, all wrapped in a small web UI.

🚀 View the [Live Demo](https://mindbridge.xuyuanhui.org/)

---

## Production Architecture

The production version of MindBridge utilizes a robust microservice architecture deployed across specialised cloud providers for optimal performance and scalability:

- Frontend: A React/Vite application deployed on Vercel for global edge delivery.

- Microservices (Render): The following services are deployed independently on Render:

  - Simulator: Generates synthetic EEG signals and feature sets.

  - Chat-API: Manages the Gemini integration and workload-aware prompt logic.

  - Brain-Engine: Processes feature frames to predict cognitive load.

- Global Cache: An Upstash Redis instance connects the simulator to the brain engine.

---

## Key Production Notes
### 1. Real-Time Data Streaming
Due to the command-per-second limitations of the Upstash Redis free tier, the production version does not reflect real-time EEG data at the same high frequency as the local Docker version. The data visualized is processed in throttled batches to maintain service stability.

### 2. Cognitive Load Comparison
This demo is optimized for comparing responses under different predefined cognitive loads. By using the "Workload Override" feature, you can see how the Gemini-powered assistant adjusts its tone, brevity, and complexity when it detects low, medium, or high workload scores.

---

## Features & Usage

### EEG Visualisation
- Navigate to the *EEG* tab and click *Connect*.
- View a simulated 128 Hz EEG trace across 14 channels.
- Note: In production, this serves as a visual representation of the data being fed into the brain engine.

### Workload-Aware Chat
- Navigate to the Chat tab.
- Manual Override (Recommended for Testing): Use the 1–9 scale to simulate different mental states:
  - Low (1–6.5): Detailed, explanatory responses.
  - Medium (6.5–8.5): Balanced, concise feedback.
  - High (≥ 8.5): Brief, action-oriented, and low-cognitive-overhead replies.

---

## Local Development

If you want to experience the local version with full real-time streaming capabilities, please switch to the main branch and follow the setup instructions provided there.

The local version allows you to:

- Stream real-time EEG data at 128 Hz without Upstash limitations.
- Run the full stack via Docker Compose.

- Connect directly to a local Redis instance for zero-latency feature processing.