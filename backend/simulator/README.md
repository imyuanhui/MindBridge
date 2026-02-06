# EEG Simulator

A real-time EEG signal simulator that generates raw EEG data at 128Hz for 14 channels. The simulator sends data to both the frontend (via WebSocket) and the brain-engine microservice (via Redis).

## Features

- **Realistic EEG Signal Generation**: Simulates multi-frequency EEG signals with:
  - Delta (0.5-4 Hz)
  - Theta (4-8 Hz)
  - Alpha (8-13 Hz)
  - Beta (13-30 Hz)
  - Gamma (30-45 Hz)
- **14 Channels**: Generates data for 14 EEG channels simultaneously
- **128Hz Sampling Rate**: Matches typical EEG device sampling rates
- **Real-time WebSocket Streaming**: Broadcasts raw EEG data to connected frontend clients
- **Feature Extraction**: Automatically extracts frequency band powers and sends to brain-engine microservice

## Architecture

```
Simulator → WebSocket → Frontend (real-time display)
         → Redis → Brain-Engine (feature extraction & workload prediction)
```

## API Endpoints

### WebSocket
- **Endpoint**: `ws://localhost:8001/ws/eeg`
- **Protocol**: WebSocket
- **Data Format**: JSON
  ```json
  {
    "timestamp": 1234567890.123,
    "channels": [[ch1_samples], [ch2_samples], ...],
    "sample_rate": 128
  }
  ```

### HTTP Endpoints
- `GET /`: Service information
- `GET /health`: Health check

## Usage

### Running with Docker Compose

The simulator is included in the main `docker-compose.yaml`. Start all services:

```bash
cd backend
docker-compose up --build
```

The simulator will be available at `http://localhost:8001`

### Running Standalone

```bash
cd backend/simulator
pip install -r requirements.txt
python main.py
```

## Frontend Integration Example

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/eeg');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const { timestamp, channels, sample_rate } = data;
    
    // channels is an array of 14 arrays, each containing 128 samples
    // Process and display the EEG data
    channels.forEach((channelData, channelIndex) => {
        // Update visualization for each channel
        updateChannelVisualization(channelIndex, channelData);
    });
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('WebSocket connection closed');
};
```

## Data Flow

1. **Signal Generation**: The simulator generates 128 samples per second for each of 14 channels
2. **Buffering**: Samples are collected into 1-second buffers (128 samples)
3. **Feature Extraction**: Every second, frequency band powers are extracted using Welch's method
4. **Distribution**:
   - Raw data is broadcast to all connected WebSocket clients
   - Extracted features (70 features: 14 channels × 5 bands) are sent to Redis key `current_frame`
5. **Brain-Engine Processing**: The brain-engine microservice reads from Redis and performs workload prediction

## Feature Format

The extracted features follow this format:
- 70 features total (14 channels × 5 frequency bands)
- Order: Ch1_Delta, Ch1_Theta, Ch1_Alpha, Ch1_Beta, Ch1_Gamma, Ch2_Delta, ..., Ch14_Gamma
- Features are sent to Redis as JSON: `{"timestamp": ..., "features": [70 values]}`

## Configuration

Key parameters in `main.py`:
- `SAMPLE_RATE = 128`: Sampling frequency in Hz
- `NUM_CHANNELS = 14`: Number of EEG channels
- `BUFFER_SIZE = 128`: Samples per buffer (1 second at 128Hz)
