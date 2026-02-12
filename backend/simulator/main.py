import asyncio
import json
import numpy as np
import redis
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from scipy import signal
import time
from typing import List
import uvicorn
import dotenv

dotenv.load_dotenv()

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
# Will be initialized in lifespan
r = None

class EEGSimulator:
    """Generates realistic EEG signals for 14 channels at 128Hz with time-varying cognitive state."""
    
    def __init__(self, sample_rate=128, num_channels=14):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.time = 0.0
        
        # Cognitive state: 0 = relaxed, 1 = high mental workload. Drifts over time so
        # band-power mix (and thus KNN workload rating) fluctuates.
        self.cognitive_state = 0.5
        self._state_random_walk = 0.5  # for smooth random drift
        
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
    
    def _update_cognitive_state(self, dt: float):
        """Drift cognitive state so workload rating varies over time."""
        # Slow deterministic oscillation (periods ~25s and ~40s) for smooth variation
        t = self.time
        deterministic = 0.5 + 0.35 * np.sin(2 * np.pi * 0.04 * t) + 0.2 * np.sin(2 * np.pi * 0.025 * t + 1.0)
        # Small random walk so it doesn't repeat exactly
        self._state_random_walk += np.random.uniform(-0.015, 0.015)
        self._state_random_walk = np.clip(self._state_random_walk, 0.0, 1.0)
        # Blend: mostly deterministic with some random drift
        self.cognitive_state = 0.75 * deterministic + 0.25 * self._state_random_walk
        self.cognitive_state = float(np.clip(self.cognitive_state, 0.0, 1.0))
    
    def generate_sample(self):
        """Generate one sample (all channels) at current time."""
        dt = 1.0 / self.sample_rate
        self._update_cognitive_state(dt)
        
        s = self.cognitive_state  # 0 = relaxed, 1 = high workload
        # Band gains that shift with cognitive state (affects PSD → workload rating)
        # Relaxed: more Alpha/Delta. High load: more Beta/Gamma.
        delta_gain = 0.7 - 0.35 * s
        theta_gain = 0.35 + 0.15 * (1 - s)
        alpha_gain = 1.1 - 0.55 * s
        beta_gain = 0.15 + 0.85 * s
        gamma_gain = 0.08 + 0.5 * s
        
        samples = np.zeros(self.num_channels)
        
        for ch in range(self.num_channels):
            params = self.channel_params[ch]
            
            # Multi-frequency components (gains vary with cognitive state)
            delta = delta_gain * params['amplitude'] * np.sin(2 * np.pi * 2 * self.time + params['phase'] * 0.5)
            theta = theta_gain * params['amplitude'] * np.sin(2 * np.pi * 6 * self.time + params['phase'] * 0.7)
            alpha = alpha_gain * params['amplitude'] * np.sin(2 * np.pi * params['base_freq'] * self.time + params['phase'])
            beta = beta_gain * params['amplitude'] * np.sin(2 * np.pi * 20 * self.time + params['phase'] * 1.3)
            gamma = gamma_gain * params['amplitude'] * np.sin(2 * np.pi * 35 * self.time + params['phase'] * 2.0)
            
            # Extra modulation so each channel isn't perfectly in sync
            modulation = 1 + 0.12 * np.sin(2 * np.pi * 0.08 * self.time + ch * 0.4)
            
            signal = (delta + theta + alpha + beta + gamma) * modulation
            
            # Realistic noise + occasional artifacts
            noise = np.random.normal(0, params['noise_level'])
            if np.random.random() < 0.01:
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
                # Calculate power using trapezoidal integration
                # Manual implementation for compatibility
                if len(psd[band_mask]) > 1:
                    # Trapezoidal rule: sum of (y1 + y2) / 2 * (x2 - x1)
                    band_power = np.sum((psd[band_mask][1:] + psd[band_mask][:-1]) / 2.0 * np.diff(freqs[band_mask]))
                elif len(psd[band_mask]) == 1:
                    band_power = psd[band_mask][0] * (high - low)
                else:
                    band_power = 0.0
                features.append(float(band_power))
        
        return features


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    global simulator, generation_task, r
    
    # Startup
    print("Starting EEG Simulator...")
    
    # Initialize Redis connection
    REDIS_URL = os.getenv('UPSTASH_REDIS_REST_URL')
    print("Connecting to Redis...")
    
    max_retries = 5
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            r = redis.Redis.from_url(REDIS_URL)
            r.ping()  # Test connection
            print(f"✓ Redis connection successful")
            break
        except (redis.ConnectionError, redis.TimeoutError) as e:
            if attempt < max_retries - 1:
                print(f"⚠ Redis connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                print(f"✗ Failed to connect to Redis after {max_retries} attempts")
                print(f"  Error: {e}")
                print(f"  Make sure Redis is running at {redis_host}:{redis_port}")
                r = None
    
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
    global r
    
    if r is None:
        print("⚠ Redis not connected, cannot send features")
        return
    
    try:
        frame_data = {
            'timestamp': time.time(),
            'features': features
        }
        r.set("current_frame", json.dumps(frame_data))
        # Log first few sends and then every 10th
        if not hasattr(send_to_microservice, 'send_count'):
            send_to_microservice.send_count = 0
        send_to_microservice.send_count += 1
        if send_to_microservice.send_count <= 3 or send_to_microservice.send_count % 10 == 0:
            print(f"✓ Sent frame #{send_to_microservice.send_count} to Redis ({len(features)} features)")
    except redis.ConnectionError as e:
        print(f"✗ Redis connection error: {e}")
    except redis.TimeoutError as e:
        print(f"✗ Redis timeout error: {e}")
    except Exception as e:
        print(f"✗ Error sending to Redis: {type(e).__name__}: {e}")


async def eeg_generation_loop():
    """Main loop that generates EEG data continuously"""
    global simulator, r
    buffer = np.zeros((BUFFER_SIZE, NUM_CHANNELS))
    buffer_index = 0
    frame_count = 0
    
    print("EEG generation loop started")
    
    while True:
        if simulator is None:
            await asyncio.sleep(0.1)
            continue
        
        if r is None:
            print("⚠ Waiting for Redis connection...")
            await asyncio.sleep(1.0)
            continue
            
        # Generate one sample for all channels
        sample = simulator.generate_sample()
        buffer[buffer_index] = sample
        
        buffer_index += 1
        
        # When buffer is full (1 second of data), process it
        if buffer_index >= BUFFER_SIZE:
            frame_count += 1
            
            try:
                # Broadcast raw data to frontend
                await broadcast_eeg_data(buffer)
                
                # Extract features from the buffer
                features = simulator.extract_features(buffer)
                
                if len(features) != 70:
                    print(f"⚠ Warning: Expected 70 features, got {len(features)}")
                
                # Send features to microservice via Redis
                await send_to_microservice(features)
                
                if frame_count == 1:
                    print(f"✓ First frame processed and sent to Redis")
                elif frame_count % 10 == 0:  # Log every 10 frames
                    print(f"Processed {frame_count} frames so far...")
                
            except Exception as e:
                print(f"✗ Error processing frame #{frame_count}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            
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


@app.get("/status")
async def status():
    """Get detailed status of the simulator"""
    global r
    
    # Check Redis connection
    redis_status = "disconnected"
    if r is not None:
        try:
            r.ping()
            redis_status = "connected"
        except:
            redis_status = "disconnected"
    
    # Check if simulator is initialized
    simulator_status = "initialized" if simulator is not None else "not initialized"
    
    # Get latest frame info from Redis
    latest_frame_info = None
    try:
        frame_data = r.get("current_frame")
        if frame_data:
            data = json.loads(frame_data)
            latest_frame_info = {
                "timestamp": data.get("timestamp"),
                "feature_count": len(data.get("features", []))
            }
    except:
        pass
    
    return {
        "status": "running",
        "sample_rate": SAMPLE_RATE,
        "channels": NUM_CHANNELS,
        "simulator": simulator_status,
        "redis": redis_status,
        "websocket_connections": len(active_connections),
        "latest_frame": latest_frame_info
    }




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
