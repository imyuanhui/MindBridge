import redis
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import time
import os

dotenv.load_dotenv()

REDIS_URL = os.getenv('UPSTASH_REDIS_REST_URL')

# 1. Setup Redis Connection
# In Docker, 'cache' will be the hostname of the Redis container
r = redis.Redis.from_url(REDIS_URL)

# 2. Load the Reference Library (Your 45-person dataset)
df = pd.read_csv('brain_features_cleaned.csv')
features_ref = df.drop(['rating', 'target_workload'], axis=1).values
ratings_ref = df['rating'].values

# 3. Initialize KNN Model
# We use the 'brute' algorithm for small datasets (45 subjects is tiny for a CPU)
knn = NearestNeighbors(n_neighbors=3, algorithm='brute', metric='euclidean')
knn.fit(features_ref)

def calculate_weighted_workload(new_features):
    """
    Implements your Weighted KNN logic:
    1. Finds the 3 nearest neighbors.
    2. Calculates weights as 1/distance.
    3. Returns a weighted average of their ratings.
    """
    distances, indices = knn.kneighbors([new_features])
    
    # Avoid division by zero with a tiny epsilon
    weights = 1.0 / (distances[0] + 1e-5)
    
    # Weighted average of the ratings from the reference library
    neighbor_ratings = ratings_ref[indices[0]]
    prediction = np.sum(weights * neighbor_ratings) / np.sum(weights)
    
    return float(prediction)

def start_engine():
    print("Brain State Context Engine started. Watching for 'current_frame'...")
    print("Waiting for simulator data...")
    
    frames_processed = 0
    last_timestamp = None
    
    while True:
        # Pull the latest frame pushed by the simulator (or real device)
        frame_data = r.get("current_frame")
        
        if frame_data:
            try:
                data = json.loads(frame_data)
                live_features = data['features']
                frame_timestamp = data.get('timestamp', 0)
                
                # Check if this is a new frame (avoid reprocessing same frame)
                if frame_timestamp != last_timestamp:
                    last_timestamp = frame_timestamp
                    
                    # Validate feature count
                    if len(live_features) != 70:
                        print(f"⚠ Warning: Expected 70 features, got {len(live_features)}")
                    
                    # Perform the inference
                    workload_score = calculate_weighted_workload(live_features)
                    
                    # Save the result to a new Redis key for the Chat API to read
                    r.set("latest_workload_score", round(workload_score, 2))
                    
                    frames_processed += 1
                    
                    # Log the prediction
                    print(f"[Frame #{frames_processed}] Processed Frame. Predicted Workload: {round(workload_score, 2)}")
                # else: same frame, skip processing
                
            except json.JSONDecodeError as e:
                print(f"⚠ Error parsing frame data: {e}")
            except Exception as e:
                print(f"⚠ Error processing frame: {e}")
        else:
            # No data available - this is normal at startup
            if frames_processed == 0:
                # Only print once to avoid spam
                pass
            
        # Poll every second to stay synced with the simulator
        time.sleep(1.0)

if __name__ == "__main__":
    start_engine()