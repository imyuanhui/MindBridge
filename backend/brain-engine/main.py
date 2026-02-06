import redis
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import time

# 1. Setup Redis Connection
# In Docker, 'cache' will be the hostname of the Redis container
r = redis.Redis(host='cache', port=6379, decode_responses=True)

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
    
    while True:
        # Pull the latest frame pushed by the simulator (or real device)
        frame_data = r.get("current_frame")
        
        if frame_data:
            data = json.loads(frame_data)
            live_features = data['features']
            
            # Perform the inference
            workload_score = calculate_weighted_workload(live_features)
            
            # Save the result to a new Redis key for the Chat API to read
            r.set("latest_workload_score", round(workload_score, 2))
            
            # Log the prediction
            print(f"Processed Frame. Predicted Workload: {round(workload_score, 2)}")
            
        # Poll every second to stay synced with the simulator
        time.sleep(1.0)

if __name__ == "__main__":
    start_engine()