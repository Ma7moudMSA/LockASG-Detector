# Replace your current tracking implementation with this improved version
from collections import deque

# Add these at the top with other imports
import numpy as np

# Enhanced tracking variables
tracked_objects = {}
next_object_id = 0
max_disappeared = 10  # Frames an object can disappear before being removed
track_points = {}  # Store recent positions for each object
max_track_points = 20  # How many points to keep in history

def update_tracks(detections):
    global next_object_id, tracked_objects, track_points
    
    target_detections = [det for det in detections if det.get('class') in target_classes]
    current_centroids = []
    current_classes = []
    
    for det in target_detections:
        centroid = get_centroid(det)
        current_centroids.append(centroid)
        current_classes.append(det.get('class'))
    
    # If no objects currently tracked, register all detections
    if len(tracked_objects) == 0:
        for i in range(len(current_centroids)):
            tracked_objects[next_object_id] = (current_centroids[i], current_classes[i])
            track_points[next_object_id] = deque(maxlen=max_track_points)
            track_points[next_object_id].append(current_centroids[i])
            next_object_id += 1
    
    else:
        # Calculate Euclidean distance between existing objects and new detections
        object_ids = list(tracked_objects.keys())
        object_centroids = [tracked_objects[obj_id][0] for obj_id in object_ids]
        
        # Compute distance between each pair of existing and new centroids
        D = np.zeros((len(object_ids), len(current_centroids)))
        for i in range(len(object_ids)):
            for j in range(len(current_centroids)):
                D[i, j] = euclidean_distance(object_centroids[i], current_centroids[j])
        
        # Find optimal assignments using Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(D)
        
        used_rows = set()
        used_cols = set()
        
        # Assign matches
        for (row, col) in zip(rows, cols):
            if D[row, col] < 100:  # Maximum distance threshold
                obj_id = object_ids[row]
                tracked_objects[obj_id] = (current_centroids[col], current_classes[col])
                if obj_id in track_points:
                    track_points[obj_id].append(current_centroids[col])
                used_rows.add(row)
                used_cols.add(col)
        
        # Check for unassigned rows (disappeared objects)
        for row in range(len(object_ids)):
            if row not in used_rows:
                # Mark as disappeared
                pass  # We'll handle disappearance tracking
        
        # Check for unassigned columns (new objects)
        for col in range(len(current_centroids)):
            if col not in used_cols:
                tracked_objects[next_object_id] = (current_centroids[col], current_classes[col])
                track_points[next_object_id] = deque(maxlen=max_track_points)
                track_points[next_object_id].append(current_centroids[col])
                next_object_id += 1
    
    return tracked_objects