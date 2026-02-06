import asyncio
import json
import numpy as np
import redis
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from scipy import signal
import time
from typing import List
import uvicorn

# EEG Simulation Parameters
SAMPLE_RATE = 128  # Hz
NUM_CHANNELS = 14
BUFFER_SIZE = 128  # 1 second of data at 128Hz

# Frequency bands (Hz)
FREQ_BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 45)
}

# Global simulator instance
simulator = None
generation_task = None

# Store active WebSocket connections
active_connections: List[WebSocket] = []

# Redis connection for microservice communication
r = redis.Redis(host='cache', port=6379, decode_responses=True)


class EEGSimulator:
    """Generates realistic EEG signals for 14 channels at 128Hz"""
    
    def __init__(self, sample_rate=128, num_channels=14):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.time = 0.0
        
        # Initialize channel-specific parameters for realistic variation
        self.channel_params = []
        for ch in range(num_channels):
            # Each channel has different base frequencies and amplitudes
            base_freq = np.random.uniform(8, 12)  # Alpha range
            amplitude = np.random.uniform(10, 50)
            phase = np.random.uniform(0, 2 * np.pi)
            
            self.channel_params.append({
                'base_freq': base_freq,
                'amplitude': amplitude,
                'phase': phase,
                'noise_level': np.random.uniform(2, 8)
            })
    
    def generate_sample(self):
        """Generate one sample (all channels) at current time"""
        dt = 1.0 / self.sample_rate
        samples = np.zeros(self.num_channels)
        
        for ch in range(self.num_channels):
            params = self.channel_params[ch]
            
            # Generate multi-frequency signal (realistic EEG)
            # Alpha wave (dominant)
            alpha = params['amplitude'] * np.sin(2 * np.pi * params['base_freq'] * self.time + params['phase'])
            
            # Theta component
            theta = 0.3 * params['amplitude'] * np.sin(2 * np.pi * 6 * self.time + params['phase'] * 0.7)
            
            # Beta component
            beta = 0.2 * params['amplitude'] * np.sin(2 * np.pi * 20 * self.time + params['phase'] * 1.3)
            
            # Delta component (slow wave)
            delta = 0.4 * params['amplitude'] * np.sin(2 * np.pi * 2 * self.time + params['phase'] * 0.5)
            
            # Gamma component (high frequency)
            gamma = 0.1 * params['amplitude'] * np.sin(2 * np.pi * 35 * self.time + params['phase'] * 2.0)
            
            # Add some non-linear modulation for realism
            modulation = 1 + 0.1 * np.sin(2 * np.pi * 0.1 * self.time)
            
            # Combine all components
            signal = (alpha + theta + beta + delta + gamma) * modulation
            
            # Add realistic noise (Gaussian + occasional artifacts)
            noise = np.random.normal(0, params['noise_level'])
            if np.random.random() < 0.01:  # 1% chance of artifact
                noise += np.random.normal(0, 20) * np.exp(-abs(np.random.normal(0, 0.1)))
            
            samples[ch] = signal + noise
        
        self.time += dt
        return samples
    
    def extract_features(self, eeg_data):
        """
        Extract frequency band powers from EEG data
        Returns: List of 70 features (14 channels × 5 bands)
        """
        features = []
        
        for ch in range(self.num_channels):
            channel_data = eeg_data[:, ch]
            
            # Compute power spectral density
            freqs, psd = signal.welch(
                channel_data,
                fs=self.sample_rate,
                nperseg=min(len(channel_data), 256),
                noverlap=None
            )
            
            # Extract power in each frequency band
            for band_name, (low, high) in FREQ_BANDS.items():
                # Find indices for frequency band
                band_mask = (freqs >= low) & (freqs <= high)
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                features.append(float(band_power))
        
        return features


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    global simulator, generation_task
    
    # Startup
    print("Starting EEG Simulator...")
    simulator = EEGSimulator(SAMPLE_RATE, NUM_CHANNELS)
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Channels: {NUM_CHANNELS}")
    print("WebSocket endpoint: ws://localhost:8001/ws/eeg")
    
    # Start the EEG generation loop
    generation_task = asyncio.create_task(eeg_generation_loop())
    
    yield
    
    # Shutdown
    if generation_task:
        generation_task.cancel()
        try:
            await generation_task
        except asyncio.CancelledError:
            pass
    print("EEG Simulator stopped.")


app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def broadcast_eeg_data(data):
    """Broadcast EEG data to all connected WebSocket clients"""
    if active_connections:
        # Transpose data so channels are the outer dimension (14 channels, each with 128 samples)
        channels_data = data.T.tolist()  # Shape: (NUM_CHANNELS, BUFFER_SIZE)
        
        message = json.dumps({
            'timestamp': time.time(),
            'channels': channels_data,
            'sample_rate': SAMPLE_RATE
        })
        
        # Send to all connected clients
        disconnected = []
        for connection in active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            if conn in active_connections:
                active_connections.remove(conn)


async def send_to_microservice(features):
    """Send extracted features to brain-engine via Redis"""
    try:
        frame_data = {
            'timestamp': time.time(),
            'features': features
        }
        r.set("current_frame", json.dumps(frame_data))
    except Exception as e:
        print(f"Error sending to Redis: {e}")


async def eeg_generation_loop():
    """Main loop that generates EEG data continuously"""
    global simulator
    buffer = np.zeros((BUFFER_SIZE, NUM_CHANNELS))
    buffer_index = 0
    
    while True:
        if simulator is None:
            await asyncio.sleep(0.1)
            continue
            
        # Generate one sample for all channels
        sample = simulator.generate_sample()
        buffer[buffer_index] = sample
        
        buffer_index += 1
        
        # When buffer is full (1 second of data), process it
        if buffer_index >= BUFFER_SIZE:
            # Broadcast raw data to frontend
            await broadcast_eeg_data(buffer)
            
            # Extract features from the buffer
            features = simulator.extract_features(buffer)
            
            # Send features to microservice via Redis
            await send_to_microservice(features)
            
            # Reset buffer
            buffer_index = 0
        
        # Sleep to maintain 128Hz sampling rate
        await asyncio.sleep(1.0 / SAMPLE_RATE)


@app.websocket("/ws/eeg")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time EEG data streaming"""
    await websocket.accept()
    active_connections.append(websocket)
    print(f"Client connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # Echo back or handle client messages if needed
            await websocket.send_text(json.dumps({"status": "connected"}))
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print(f"Client disconnected. Total connections: {len(active_connections)}")


@app.get("/")
async def root():
    return {
        "service": "EEG Simulator",
        "sample_rate": SAMPLE_RATE,
        "channels": NUM_CHANNELS,
        "websocket_endpoint": "/ws/eeg"
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "connections": len(active_connections)}




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
