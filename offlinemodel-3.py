import cv2
import time
import threading
import math
from queue import Queue
import json
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import numpy as np
from collections import deque
import torch
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# Device detection for optimized inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load your trained local YOLOv8 model weights
model = YOLO('best_3.pt')  # Change this to your model path!

# Configuration - Removed Roboflow API settings
target_classes = ['Lock', 'unassembled_lock', 'Worker']

# Class-specific confidence thresholds
CLASS_CONFIDENCE = {
    'Worker': 0.25,          # Higher confidence for Worker
    'Lock': 0.25,             # Standard confidence for Lock
    'unassembled_lock': 0.2  # Standard confidence for unassembled_lock
}

# Visual settings
BORDER_THICKNESS = 1  # Reduced from 3 to 1 for thinner borders

# Enhanced tracking variables
tracked_objects = {}
next_object_id = 0
counted_ids = set()  # Keep this for backward compatibility but won't be used
cross_count = 0

# NEW: Zone-based counting variables instead of line
counting_zone = None  # Will be defined as (x1, y1, x2, y2)
zone_width = 950  # Width of the counting zone
zone_height = 700  # Height of the counting zone
zone_x = None  # X position of zone (will be set to center)
zone_y = None  # Y position of zone (will be set to center)
# NEW: Track which objects are currently inside the zone
#inside_zone_ids = set()  # Objects currently inside the counting zone

# NEW: Track which objects are currently inside the zone, organized by class

max_disappeared = 10  # Frames an object can disappear before being removed
track_points = {}  # Store recent positions for each object
max_track_points = 20  # How many points to keep in history

# Kalman filter trackers
kalman_trackers = {}

# NEW: Buffer system for locks
lock_buffer = 0
unassembled_lock_buffer = 0
total_lock_buffer = 0  # Combined buffer for both lock types
buffer_increment_times = []  # Track time between buffer increments
last_buffer_increment_time = None

# Zone visibility toggle
show_counting_zone = True  # Changed from line to zone

# Enhanced timing variables
class_appearances = {cls: 0 for cls in target_classes}
object_timings = {}
object_total_time = {}
session_start_time = time.time()

# NEW: CSV data storage
csv_data = []

# Threading setup
frame_queue = Queue(maxsize=2)
prediction_cache = []
last_prediction_time = 0

inside_zone_ids_by_class = {cls: set() for cls in target_classes}


class ObjectTracker:
    def __init__(self, initial_position):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix
        self.kf.F = np.array([[1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        
        # Measurement function
        self.kf.H = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]])
        
        # Covariance matrix
        self.kf.P *= 1000
        
        # Measurement noise
        self.kf.R = np.array([[5, 0],
                             [0, 5]])
        
        # Process noise
        self.kf.Q = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]]) * 0.1
        
        # Initialize state
        self.kf.x = np.array([initial_position[0], initial_position[1], 0, 0])
        
        self.hits = 0
        self.misses = 0
        self.hit_streak = 0
        
    def predict(self):
        self.kf.predict()
        return (int(self.kf.x[0]), int(self.kf.x[1]))
        
    def update(self, measurement):
        self.kf.update(measurement)
        self.hits += 1
        self.hit_streak += 1
        self.misses = 0
        
    def miss(self):
        self.misses += 1
        self.hit_streak = 0

def get_centroid(detection):
    x = detection.get('x', 0)
    y = detection.get('y', 0)
    return (int(x), int(y))

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def predict_next_position(obj_id, current_position):
    if obj_id not in track_points or len(track_points[obj_id]) < 2:
        return current_position
    
    # Calculate velocity from recent positions
    positions = list(track_points[obj_id])
    dx = positions[-1][0] - positions[-2][0]
    dy = positions[-1][1] - positions[-2][1]
    
    # Predict next position based on velocity
    predicted_x = current_position[0] + dx
    predicted_y = current_position[1] + dy
    
    return (int(predicted_x), int(predicted_y))

def is_point_in_zone(point, zone):
    """Check if a point is inside the counting zone"""
    x, y = point
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2

def setup_counting_zone(frame_width, frame_height):
    """Setup the counting zone in the center of the frame"""
    global counting_zone, zone_x, zone_y
    
    # Center the zone
    zone_x = (frame_width - zone_width) // 2
    zone_y = (frame_height - zone_height) // 2
    
    # Define zone as (x1, y1, x2, y2)
    counting_zone = (zone_x, zone_y, zone_x + zone_width, zone_y + zone_height)
    
    print(f"📍 Counting zone setup: {counting_zone}")
    return counting_zone

def log_object_statistics_to_csv():
    """Log detailed object statistics to CSV during session"""
    current_time = time.time()
    session_time = current_time - session_start_time
    
    # Calculate current total times
    total_times = {cls: 0.0 for cls in target_classes}
    all_object_ids = {cls: [] for cls in target_classes}
    
    # Add completed objects
    for obj_id, data in object_total_time.items():
        cls = data['class']
        total_times[cls] += data['total_time']
        all_object_ids[cls].append(obj_id)
    
    # Add currently active objects
    for obj_id, timing_data in object_timings.items():
        if obj_id in tracked_objects:
            cls = timing_data['class']
            active_duration = current_time - timing_data['first_seen']
            total_times[cls] += active_duration
            if obj_id not in all_object_ids[cls]:
                all_object_ids[cls].append(obj_id)
    
    # Log current statistics snapshot
    log_to_csv('Statistics_Snapshot', {
        'class': 'ALL_CLASSES',
        'id': 'SNAPSHOT',
        'info': f"Lock_Time: {total_times.get('Lock', 0):.1f}s, UnassembledLock_Time: {total_times.get('unassembled_lock', 0):.1f}s, Worker_Time: {total_times.get('Worker', 0):.1f}s, Active_Objects: {len(tracked_objects)}"
    })
def log_to_csv(event_type, data):
    """Add event to CSV data list with enhanced object tracking"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_time = time.time() - session_start_time
    
    # Calculate current active object counts
    active_locks = len([obj_id for obj_id, (_, cls) in tracked_objects.items() if cls == 'Lock'])
    active_unassembled = len([obj_id for obj_id, (_, cls) in tracked_objects.items() if cls == 'unassembled_lock'])
    active_workers = len([obj_id for obj_id, (_, cls) in tracked_objects.items() if cls == 'Worker'])
    
    csv_entry = {
        'Timestamp': timestamp,
        'Session_Time': round(session_time, 2),
        'Event_Type': event_type,
        'Object_Class': data.get('class', ''),
        'Object_ID': data.get('id', ''),
        'Lock_Buffer': lock_buffer,
        'Unassembled_Lock_Buffer': unassembled_lock_buffer,
        'Total_Buffer': total_lock_buffer,
        'Time_Since_Last_Increment': data.get('time_since_last', 0),
        'Active_Locks': active_locks,
        'Active_Unassembled_Locks': active_unassembled,
        'Active_Workers': active_workers,
        'Total_Active_Objects': len(tracked_objects),
        'Objects_In_Zone': sum(len(ids) for ids in inside_zone_ids_by_class.values()),
        'Additional_Info': data.get('info', '')
    }
    csv_data.append(csv_entry)


def update_buffer(obj_class, obj_id):
    """Update buffer counters and track timing"""
    global lock_buffer, unassembled_lock_buffer, total_lock_buffer
    global last_buffer_increment_time, buffer_increment_times
    
    current_time = time.time()
    time_since_last = 0
    
    if last_buffer_increment_time is not None:
        time_since_last = current_time - last_buffer_increment_time
        buffer_increment_times.append(time_since_last)
    
    if obj_class == 'Lock':
        lock_buffer += 1
        total_lock_buffer += 1
        print(f"🔒 LOCK BUFFER INCREMENT! Lock Buffer: {lock_buffer} | Total Buffer: {total_lock_buffer}")
        print(f"⏱️  Time since last increment: {time_since_last:.2f}s")
        
    elif obj_class == 'unassembled_lock':
        unassembled_lock_buffer += 1
        total_lock_buffer += 1
        print(f"🔓 UNASSEMBLED LOCK BUFFER INCREMENT! Unassembled Buffer: {unassembled_lock_buffer} | Total Buffer: {total_lock_buffer}")
        print(f"⏱️  Time since last increment: {time_since_last:.2f}s")
    
    last_buffer_increment_time = current_time
    
    # Log to CSV
    log_to_csv('Buffer_Increment', {
        'class': obj_class,
        'id': obj_id,
        'time_since_last': round(time_since_last, 2),
        'info': f"Buffer incremented to {total_lock_buffer}"
    })

def update_tracks(detections):
    global next_object_id, tracked_objects, track_points, kalman_trackers, inside_zone_ids_by_class
    current_time = time.time()
    
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
            kalman_trackers[next_object_id] = ObjectTracker(current_centroids[i])
            
            # Initialize timing data
            class_appearances[current_classes[i]] += 1
            object_timings[next_object_id] = {
                'class': current_classes[i],
                'first_seen': current_time,
                'last_seen': current_time,
                'current_duration': 0.0,
                'session_time': current_time - session_start_time
            }
            print(f"🆕 New {current_classes[i]} detected! ID: {next_object_id} at {current_time - session_start_time:.1f}s into session")
            
            log_to_csv('New_Detection', {
                'class': current_classes[i],
                'id': next_object_id,
                'info': f"First appeared at {current_time - session_start_time:.1f}s"
            })
            
            next_object_id += 1
    else:
        # Calculate Euclidean distance between existing objects and new detections
        object_ids = list(tracked_objects.keys())
        object_centroids = [tracked_objects[obj_id][0] for obj_id in object_ids]
        
        if len(current_centroids) > 0 and len(object_centroids) > 0:
            # Compute distance between each pair of existing and new centroids
            D = np.zeros((len(object_ids), len(current_centroids)))
            for i in range(len(object_ids)):
                for j in range(len(current_centroids)):
                    D[i, j] = euclidean_distance(object_centroids[i], current_centroids[j])
            
            # Find optimal assignments using Hungarian algorithm
            rows, cols = linear_sum_assignment(D)
            
            used_rows = set()
            used_cols = set()
            
            # Assign matches
            for (row, col) in zip(rows, cols):
                if D[row, col] < 200:  # Maximum distance threshold
                    obj_id = object_ids[row]
                    tracked_objects[obj_id] = (current_centroids[col], current_classes[col])
                    
                    # Update Kalman filter
                    if obj_id in kalman_trackers:
                        kalman_trackers[obj_id].update(np.array(current_centroids[col]))
                    
                    # Update track points
                    if obj_id in track_points:
                        track_points[obj_id].append(current_centroids[col])
                    
                    # Update timing
                    if obj_id in object_timings:
                        object_timings[obj_id]['last_seen'] = current_time
                        current_duration = current_time - object_timings[obj_id]['first_seen']
                        object_timings[obj_id]['current_duration'] = current_duration
                    
                    used_rows.add(row)
                    used_cols.add(col)
            
            # Check for unassigned columns (new objects)
            for col in range(len(current_centroids)):
                if col not in used_cols:
                    tracked_objects[next_object_id] = (current_centroids[col], current_classes[col])
                    track_points[next_object_id] = deque(maxlen=max_track_points)
                    track_points[next_object_id].append(current_centroids[col])
                    kalman_trackers[next_object_id] = ObjectTracker(current_centroids[col])
                    
                    class_appearances[current_classes[col]] += 1
                    object_timings[next_object_id] = {
                        'class': current_classes[col],
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'current_duration': 0.0,
                        'session_time': current_time - session_start_time
                    }
                    print(f"🆕 New {current_classes[col]} detected! ID: {next_object_id} at {current_time - session_start_time:.1f}s into session")
                    
                    log_to_csv('New_Detection', {
                        'class': current_classes[col],
                        'id': next_object_id,
                        'info': f"First appeared at {current_time - session_start_time:.1f}s"
                    })
                    
                    next_object_id += 1
    
    # Apply Kalman filter predictions and motion blending
    for obj_id in list(tracked_objects.keys()):
        if obj_id in kalman_trackers:
            # Get Kalman filter prediction
            predicted_pos = kalman_trackers[obj_id].predict()
            
            # Also use our velocity-based prediction
            velocity_prediction = predict_next_position(obj_id, predicted_pos)
            
            # Blend the predictions (weighted average)
            blend_factor = 0.7  # Favor Kalman filter
            final_x = int(blend_factor * predicted_pos[0] + (1 - blend_factor) * velocity_prediction[0])
            final_y = int(blend_factor * predicted_pos[1] + (1 - blend_factor) * velocity_prediction[1])
            
            # Update the tracked object position with the smoothed prediction
            if obj_id in tracked_objects:
                tracked_objects[obj_id] = ((final_x, final_y), tracked_objects[obj_id][1])
    
    # Calculate final times for disappeared objects
    disappeared_objects = []
    for obj_id in list(object_timings.keys()):
        if obj_id not in tracked_objects and obj_id not in object_total_time:
            disappeared_objects.append(obj_id)
    
    for obj_id in disappeared_objects:
        if 'last_seen' in object_timings[obj_id] and 'first_seen' in object_timings[obj_id]:
            total_time = object_timings[obj_id]['last_seen'] - object_timings[obj_id]['first_seen']
            object_total_time[obj_id] = {
                'class': object_timings[obj_id]['class'],
                'total_time': total_time,
                'first_seen': object_timings[obj_id]['first_seen'],
                'last_seen': object_timings[obj_id]['last_seen'],
                'session_start': object_timings[obj_id]['session_time'],
                'session_end': object_timings[obj_id]['last_seen'] - session_start_time
            }
            print(f"📊 Object {obj_id} ({object_timings[obj_id]['class']}) visible for {total_time:.2f}s")
            
            log_to_csv('Object_Completed', {
                'class': object_timings[obj_id]['class'],
                'id': obj_id,
                'info': f"Visible for {total_time:.2f}s"
            })
            
            # Clean up tracking data
            if obj_id in kalman_trackers:
                del kalman_trackers[obj_id]
            if obj_id in track_points:
                del track_points[obj_id]
            # Also remove from inside_zone_ids if it was there
          
            for cls in inside_zone_ids_by_class:
                if obj_id in inside_zone_ids_by_class[cls]:
                    inside_zone_ids_by_class[cls].remove(obj_id)
                    break # Object can only be in one class set, so we can stop checking
            
    return tracked_objects
def save_timing_data():
    """Save timing data to JSON file with enhanced total time statistics"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detection_times_{timestamp}.json"
    
    current_time = time.time()
    session_duration = current_time - session_start_time
    
    # Calculate total times for each class
    total_times = {cls: 0.0 for cls in target_classes}
    all_object_ids = {cls: [] for cls in target_classes}
    
    # Add completed objects
    for obj_id, data in object_total_time.items():
        cls = data['class']
        total_times[cls] += data['total_time']
        all_object_ids[cls].append(obj_id)
    
    # Add currently active objects
    for obj_id, timing_data in object_timings.items():
        if obj_id in tracked_objects:
            cls = timing_data['class']
            active_duration = current_time - timing_data['first_seen']
            total_times[cls] += active_duration
            if obj_id not in all_object_ids[cls]:
                all_object_ids[cls].append(obj_id)
    
    data = {
        'session_info': {
            'start_time': session_start_time,
            'duration': session_duration,
            'timestamp': datetime.now().isoformat()
        },
        'class_appearances': class_appearances,
        'total_time_statistics': {
            'lock_total_time': total_times.get('Lock', 0),
            'unassembled_lock_total_time': total_times.get('unassembled_lock', 0),
            'combined_lock_time': total_times.get('Lock', 0) + total_times.get('unassembled_lock', 0),
            'worker_total_time': total_times.get('Worker', 0),
            'lock_ids': all_object_ids.get('Lock', []),
            'unassembled_lock_ids': all_object_ids.get('unassembled_lock', []),
            'worker_ids': all_object_ids.get('Worker', []),
            'time_percentages': {
                'lock_percentage': (total_times.get('Lock', 0) / session_duration) * 100 if session_duration > 0 else 0,
                'unassembled_lock_percentage': (total_times.get('unassembled_lock', 0) / session_duration) * 100 if session_duration > 0 else 0,
                'combined_lock_percentage': ((total_times.get('Lock', 0) + total_times.get('unassembled_lock', 0)) / session_duration) * 100 if session_duration > 0 else 0,
                'worker_percentage': (total_times.get('Worker', 0) / session_duration) * 100 if session_duration > 0 else 0
            }
        },
        'buffer_data': {
            'lock_buffer': lock_buffer,
            'unassembled_lock_buffer': unassembled_lock_buffer,
            'total_lock_buffer': total_lock_buffer,
            'buffer_increment_times': buffer_increment_times,
            'average_increment_time': sum(buffer_increment_times) / len(buffer_increment_times) if buffer_increment_times else 0
        },
        'completed_objects': {},
        'active_objects': {}
    }
    
    # Completed objects
    for obj_id, timing_data in object_total_time.items():
        data['completed_objects'][str(obj_id)] = timing_data
    
    # Currently active objects
    for obj_id, timing_data in object_timings.items():
        if obj_id in tracked_objects:
            data['active_objects'][str(obj_id)] = {
                'class': timing_data['class'],
                'duration_so_far': current_time - timing_data['first_seen'],
                'first_seen': timing_data['first_seen'],
                'session_time': timing_data['session_time']
            }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Timing data saved to {filename}")
    return filename


def save_csv_data():
    """Save all logged data to CSV file with enhanced total time statistics"""
    if not csv_data:
        print("❌ No CSV data to save")
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detection_log_{timestamp}.csv"
    
    # Calculate total time statistics for CSV summary
    current_time = time.time()
    session_duration = current_time - session_start_time
    
    # Initialize total time accumulators
    total_times = {cls: 0.0 for cls in target_classes}
    all_object_ids = {cls: [] for cls in target_classes}
    
    # Add completed objects
    for obj_id, data in object_total_time.items():
        cls = data['class']
        total_times[cls] += data['total_time']
        all_object_ids[cls].append(obj_id)
    
    # Add currently active objects
    for obj_id, timing_data in object_timings.items():
        if obj_id in tracked_objects:
            cls = timing_data['class']
            active_duration = current_time - timing_data['first_seen']
            total_times[cls] += active_duration
            if obj_id not in all_object_ids[cls]:
                all_object_ids[cls].append(obj_id)
    
    # Add session summary entries to CSV data
    session_summary_entries = [
        {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Session_Time': round(session_duration, 2),
            'Event_Type': 'Session_Summary',
            'Object_Class': 'ALL',
            'Object_ID': '',
            'Lock_Buffer': lock_buffer,
            'Unassembled_Lock_Buffer': unassembled_lock_buffer,
            'Total_Buffer': total_lock_buffer,
            'Time_Since_Last_Increment': 0,
            'Additional_Info': f"Session Duration: {session_duration:.1f}s"
        },
        {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Session_Time': round(session_duration, 2),
            'Event_Type': 'Total_Time_Lock',
            'Object_Class': 'Lock',
            'Object_ID': str(all_object_ids.get('Lock', [])),
            'Lock_Buffer': lock_buffer,
            'Unassembled_Lock_Buffer': unassembled_lock_buffer,
            'Total_Buffer': total_lock_buffer,
            'Time_Since_Last_Increment': round(total_times.get('Lock', 0), 2),
            'Additional_Info': f"Total Lock Time: {total_times.get('Lock', 0):.1f}s, Instances: {len(all_object_ids.get('Lock', []))}, Percentage: {(total_times.get('Lock', 0) / session_duration) * 100:.1f}%"
        },
        {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Session_Time': round(session_duration, 2),
            'Event_Type': 'Total_Time_Unassembled_Lock',
            'Object_Class': 'unassembled_lock',
            'Object_ID': str(all_object_ids.get('unassembled_lock', [])),
            'Lock_Buffer': lock_buffer,
            'Unassembled_Lock_Buffer': unassembled_lock_buffer,
            'Total_Buffer': total_lock_buffer,
            'Time_Since_Last_Increment': round(total_times.get('unassembled_lock', 0), 2),
            'Additional_Info': f"Total Unassembled Lock Time: {total_times.get('unassembled_lock', 0):.1f}s, Instances: {len(all_object_ids.get('unassembled_lock', []))}, Percentage: {(total_times.get('unassembled_lock', 0) / session_duration) * 100:.1f}%"
        },
        {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Session_Time': round(session_duration, 2),
            'Event_Type': 'Total_Time_Worker',
            'Object_Class': 'Worker',
            'Object_ID': str(all_object_ids.get('Worker', [])),
            'Lock_Buffer': lock_buffer,
            'Unassembled_Lock_Buffer': unassembled_lock_buffer,
            'Total_Buffer': total_lock_buffer,
            'Time_Since_Last_Increment': round(total_times.get('Worker', 0), 2),
            'Additional_Info': f"Total Worker Time: {total_times.get('Worker', 0):.1f}s, Instances: {len(all_object_ids.get('Worker', []))}, Percentage: {(total_times.get('Worker', 0) / session_duration) * 100:.1f}%"
        },
        {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Session_Time': round(session_duration, 2),
            'Event_Type': 'Combined_Lock_Time',
            'Object_Class': 'Lock_Combined',
            'Object_ID': str(all_object_ids.get('Lock', []) + all_object_ids.get('unassembled_lock', [])),
            'Lock_Buffer': lock_buffer,
            'Unassembled_Lock_Buffer': unassembled_lock_buffer,
            'Total_Buffer': total_lock_buffer,
            'Time_Since_Last_Increment': round(total_times.get('Lock', 0) + total_times.get('unassembled_lock', 0), 2),
            'Additional_Info': f"Combined Lock Time: {total_times.get('Lock', 0) + total_times.get('unassembled_lock', 0):.1f}s, Total Instances: {len(all_object_ids.get('Lock', [])) + len(all_object_ids.get('unassembled_lock', []))}, Percentage: {((total_times.get('Lock', 0) + total_times.get('unassembled_lock', 0)) / session_duration) * 100:.1f}%"
        }
    ]
    
    # Add summary entries to the CSV data
    csv_data_with_summary = csv_data + session_summary_entries
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data_with_summary)
    df.to_csv(filename, index=False)
    
    print(f"📊 CSV data saved to {filename}")
    print(f"📈 Total events logged: {len(csv_data)}")
    print(f"📋 Summary entries added: {len(session_summary_entries)}")
    print(f"🔒 Lock total time: {total_times.get('Lock', 0):.1f}s (IDs: {all_object_ids.get('Lock', [])})")
    print(f"🔓 Unassembled lock total time: {total_times.get('unassembled_lock', 0):.1f}s (IDs: {all_object_ids.get('unassembled_lock', [])})")
    print(f"👷 Worker total time: {total_times.get('Worker', 0):.1f}s (IDs: {all_object_ids.get('Worker', [])})")
    
    return filename


def calculate_total_times_alternative():
    """Alternative method: Calculate from individual object durations"""
    # Create a comprehensive list of all unique object lifespans
    all_object_lifespans = {}
    
    # Add completed objects
    for obj_id, data in object_total_time.items():
        if obj_id not in all_object_lifespans:
            all_object_lifespans[obj_id] = {
                'class': data['class'],
                'total_time': data['total_time']
            }
    
    # Add active objects (only if not already completed)
    current_time = time.time()
    for obj_id, timing_data in object_timings.items():
        if obj_id in tracked_objects and obj_id not in all_object_lifespans:
            active_duration = current_time - timing_data['first_seen']
            all_object_lifespans[obj_id] = {
                'class': timing_data['class'],
                'total_time': active_duration
            }
    
    # Sum by class
    total_times = {cls: 0.0 for cls in target_classes}
    all_object_ids = {cls: [] for cls in target_classes}
    
    for obj_id, data in all_object_lifespans.items():
        cls = data['class']
        total_times[cls] += data['total_time']
        all_object_ids[cls].append(obj_id)
    
    return total_times, all_object_ids

def print_detailed_statistics():
    """Enhanced statistics with corrected total time calculation"""
    print("\n" + "="*60)
    print("📈 DETAILED DETECTION STATISTICS")
    print("="*60)
    
    session_duration = time.time() - session_start_time
    print(f"📅 Session Duration: {session_duration:.1f} seconds")
    
    # Buffer statistics (unchanged)
    print(f"\n🔒 BUFFER STATISTICS:")
    print(f"   Lock Buffer: {lock_buffer}")
    print(f"   Unassembled Lock Buffer: {unassembled_lock_buffer}")
    print(f"   Total Lock Buffer: {total_lock_buffer}")
    
    if buffer_increment_times:
        avg_increment_time = sum(buffer_increment_times) / len(buffer_increment_times)
        min_increment_time = min(buffer_increment_times)
        max_increment_time = max(buffer_increment_times)
        print(f"   Average time between increments: {avg_increment_time:.2f}s")
        print(f"   Min time between increments: {min_increment_time:.2f}s")
        print(f"   Max time between increments: {max_increment_time:.2f}s")
    
    # Use corrected calculation
    total_times, all_object_ids = calculate_total_times_alternative()
    
    # Display corrected total time statistics
    print("\n📊 CORRECTED TOTAL TIME STATISTICS:")
    
    lock_total_time = total_times.get('Lock', 0)
    unassembled_lock_total_time = total_times.get('unassembled_lock', 0)
    worker_total_time = total_times.get('Worker', 0)
    
    print(f"🔒 Total Lock Time (Assembled): {lock_total_time:.2f}s")
    print(f"   └─ Unique Lock IDs: {all_object_ids.get('Lock', [])}")
    print(f"   └─ Total Lock instances: {len(all_object_ids.get('Lock', []))}")
    
    print(f"🔓 Total Unassembled Lock Time: {unassembled_lock_total_time:.2f}s")
    print(f"   └─ Unique Unassembled Lock IDs: {all_object_ids.get('unassembled_lock', [])}")
    print(f"   └─ Total Unassembled Lock instances: {len(all_object_ids.get('unassembled_lock', []))}")
    
    combined_lock_time = lock_total_time + unassembled_lock_total_time
    print(f"🔧 Combined Lock Time (All Types): {combined_lock_time:.2f}s")
    print(f"   └─ Expected total from your data: 9.94s")
    
    print(f"👷 Total Worker Time: {worker_total_time:.2f}s")
    print(f"   └─ Unique Worker IDs: {all_object_ids.get('Worker', [])}")
    print(f"   └─ Total Worker instances: {len(all_object_ids.get('Worker', []))}")
    
    # Debug information
    print(f"\n🔍 DEBUG INFO:")
    print(f"   Completed objects: {len(object_total_time)}")
    print(f"   Active objects: {len([obj_id for obj_id in object_timings if obj_id in tracked_objects])}")
    print(f"   Objects in both (potential double count): {len([obj_id for obj_id in object_total_time if obj_id in object_timings])}")
    
    # Class appearances
    print("\n🎯 Target Class Appearances:")
    for cls in target_classes:
        print(f"   {cls}: {class_appearances[cls]} times")
    
    # Currently active objects with real-time durations
    if any(obj_id in tracked_objects for obj_id in object_timings):
        print("\n🔄 Currently Active Objects:")
        current_time = time.time()
        for obj_id, timing_data in object_timings.items():
            if obj_id in tracked_objects:
                active_duration = current_time - timing_data['first_seen']
                session_time = timing_data['session_time']
                # Check if object is in any zone
                in_any_zone = any(obj_id in inside_zone_ids_by_class[cls] for cls in inside_zone_ids_by_class)
                in_zone_status = "IN ZONE" if in_any_zone else "OUTSIDE"
                print(f"   ID {obj_id} ({timing_data['class']}): {active_duration:.1f}s active (appeared at {session_time:.1f}s) [{in_zone_status}]")
    
    # Completed objects
    if object_total_time:
        print("\n⏱️ Completed Object Detection Times:")
        class_times = {cls: [] for cls in target_classes}
        
        for obj_id, data in object_total_time.items():
            cls = data['class']
            total_time = data['total_time']
            session_start = data.get('session_start', 0)
            session_end = data.get('session_end', 0)
            class_times[cls].append(total_time)
            print(f"   ID {obj_id} ({cls}): {total_time:.1f}s (appeared at {session_start:.1f}s, left at {session_end:.1f}s)")
        
        # Calculate statistics per class
        print("\n📊 Class Statistics:")
        for cls in target_classes:
            if class_times[cls]:
                avg_time = sum(class_times[cls]) / len(class_times[cls])
                min_time = min(class_times[cls])
                max_time = max(class_times[cls])
                total_class_time = sum(class_times[cls])
                print(f"   {cls}:")
                print(f"     - Average time: {avg_time:.1f}s")
                print(f"     - Min time: {min_time:.1f}s")
                print(f"     - Max time: {max_time:.1f}s")
                print(f"     - Total time: {total_class_time:.1f}s")
                print(f"     - Completed detections: {len(class_times[cls])}")
    
    print("="*60 + "\n")
'''

# Updated inference function using local YOLOv8 model with optimizations
def infer_frame(frame):
    """
    Run inference using local YOLOv8 model with optimizations
    """
    try:
        # Use half precision for faster inference if supported
        half = True if device != 'cpu' else False
        
        # Run YOLOv8 inference with optimizations
        results = model(frame, 
                       imgsz=640,  # Fixed size for consistency
                       conf=0.25,  # Lower initial confidence threshold
                       half=half,  # Use half precision if available
                       verbose=False)  # Disable verbose output
        
        detections = []
        
        # Extract predictions from results
        for box, cls_id, score in zip(results[0].boxes.xyxy.cpu().numpy(),
                                      results[0].boxes.cls.cpu().numpy(),
                                      results[0].boxes.conf.cpu().numpy()):
            
            # Get class name from model
            class_name = model.names[int(cls_id)]
            
            # Apply class-specific confidence thresholds
            threshold = CLASS_CONFIDENCE.get(class_name, 0.4)
            if score >= threshold and class_name in target_classes:
                # Create detection dictionary
                detection = {
                    'x': int((box[0] + box[2]) / 2),  # Center X
                    'y': int((box[1] + box[3]) / 2),  # Center Y
                    'width': int(box[2] - box[0]),    # Width
                    'height': int(box[3] - box[1]),   # Height
                    'class': class_name,
                    'confidence': float(score)
                }
                detections.append(detection)
        
        return detections
        
    except Exception as e:
        print(f"Error in local inference: {e}")
        return []
'''

def infer_frame(frame):
    """Enhanced inference with multi-scale detection for distant objects"""
    try:
        half = True if device != 'cpu' else False
        
        # Multi-scale inference for better distance detection
        all_detections = []
        scales = [640, 832]  # Multiple scales
        
        for scale in scales:
            results = model(frame, 
                           imgsz=scale,
                           conf=0.12,  # Lower confidence for distant objects
                           half=half,
                           verbose=False)
            
            # Extract predictions
            for box, cls_id, score in zip(results[0].boxes.xyxy.cpu().numpy(),
                                          results[0].boxes.cls.cpu().numpy(),
                                          results[0].boxes.conf.cpu().numpy()):
                
                class_name = model.names[int(cls_id)]
                
                # Dynamic confidence based on object size (smaller = more lenient)
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                frame_area = frame.shape[0] * frame.shape[1]
                relative_size = box_area / frame_area
                
                # Lower threshold for smaller objects (distant)
                if relative_size < 0.01:  # Very small objects
                    threshold = CLASS_CONFIDENCE.get(class_name, 0.4) * 0.6
                elif relative_size < 0.05:  # Small objects  
                    threshold = CLASS_CONFIDENCE.get(class_name, 0.4) * 0.8
                else:  # Normal size objects
                    threshold = CLASS_CONFIDENCE.get(class_name, 0.4)
                
                if score >= threshold and class_name in target_classes:
                    detection = {
                        'x': int((box[0] + box[2]) / 2),
                        'y': int((box[1] + box[3]) / 2),
                        'width': int(box[2] - box[0]),
                        'height': int(box[3] - box[1]),
                        'class': class_name,
                        'confidence': float(score)
                    }
                    all_detections.append(detection)
        
        # Remove duplicates (NMS)
        return remove_duplicate_detections(all_detections)
        
    except Exception as e:
        print(f"Error in inference: {e}")
        return []

def remove_duplicate_detections(detections):
    """Remove duplicate detections from multi-scale inference"""
    if len(detections) <= 1:
        return detections
    
    filtered = []
    for i, det1 in enumerate(detections):
        is_duplicate = False
        for j, det2 in enumerate(filtered):
            if det1['class'] == det2['class']:
                # Check if centroids are close
                dist = euclidean_distance((det1['x'], det1['y']), (det2['x'], det2['y']))
                if dist < 50:  # Threshold for duplicates
                    # Keep the one with higher confidence
                    if det1['confidence'] > det2['confidence']:
                        filtered[filtered.index(det2)] = det1
                    is_duplicate = True
                    break
        if not is_duplicate:
            filtered.append(det1)
    
    return filtered

def inference_worker():
    global prediction_cache, last_prediction_time
    
    # Warm up the model
    warmup_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = infer_frame(warmup_frame)
    
    while True:
        try:
            if not frame_queue.empty():
                frame = frame_queue.get()
                current_time = time.time()
                
                # Run inference at a fixed interval for consistent performance
                if current_time - last_prediction_time >=  0.1:  # 0.1 10 FPS inference
                    prediction_cache = infer_frame(frame)
                    last_prediction_time = current_time
                else:
                    # Use the cached prediction but update timing
                    time.sleep(0.01)
            else:
                time.sleep(0.01)  # Reduced sleep time
        except Exception as e:
            print(f"Inference worker error: {e}")
            time.sleep(0.1)


def cleanup_disappeared_objects():
    """Remove objects that haven't been seen for too long"""
    global tracked_objects, object_timings, object_total_time
    
    current_time = time.time()
    disappeared_threshold = 2.0  # Remove objects not seen for 2 seconds
    
    objects_to_remove = []
    
    for obj_id in list(tracked_objects.keys()):
        if obj_id in object_timings:
            time_since_last_seen = current_time - object_timings[obj_id]['last_seen']
            
            if time_since_last_seen > disappeared_threshold:
                objects_to_remove.append(obj_id)
                
                # Calculate final time and move to completed
                total_time = object_timings[obj_id]['last_seen'] - object_timings[obj_id]['first_seen']
                object_total_time[obj_id] = {
                    'class': object_timings[obj_id]['class'],
                    'total_time': total_time,
                    'first_seen': object_timings[obj_id]['first_seen'],
                    'last_seen': object_timings[obj_id]['last_seen'],
                    'session_start': object_timings[obj_id]['session_time'],
                    'session_end': object_timings[obj_id]['last_seen'] - session_start_time
                }
                
                print(f"🗑️ Removing disappeared object {obj_id} ({object_timings[obj_id]['class']}) - not seen for {time_since_last_seen:.1f}s")
                print(f"📊 Object {obj_id} was visible for {total_time:.2f}s")
                
                log_to_csv('Object_Disappeared', {
                    'class': object_timings[obj_id]['class'],
                    'id': obj_id,
                    'info': f"Not seen for {time_since_last_seen:.1f}s, visible for {total_time:.2f}s"
                })
    
    # Remove disappeared objects from all tracking structures
    for obj_id in objects_to_remove:
        if obj_id in tracked_objects:
            del tracked_objects[obj_id]
        if obj_id in kalman_trackers:
            del kalman_trackers[obj_id]
        if obj_id in track_points:
            del track_points[obj_id]
        for cls in inside_zone_ids_by_class:
            if obj_id in inside_zone_ids_by_class[cls]:
             inside_zone_ids_by_class[cls].remove(obj_id)
             break

        # Keep object_timings for reference but it will be used from object_total_time now


# Start background inference thread
inference_thread = threading.Thread(target=inference_worker, daemon=True)
inference_thread.start()

# Connect to phone camera
phone_camera_url = "http://192.168.137.23:8080/video"
cap = cv2.VideoCapture(phone_camera_url)

if not cap.isOpened():
    alternative_urls = [
        "http://192.168.137.23:8080/videofeed",
        "http://192.168.137.23:8080/video.mjpg"
    ]
    for url in alternative_urls:
        print(f"Trying {url}...")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            phone_camera_url = url
            break
        cap.release()

if not cap.isOpened():
    print("❌ Could not connect to phone camera.")
    exit()

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#cap.set(cv2.CAP_PROP_FPS, 30)
#cap.set(cv2.CAP_PROP_FPS, 15)  # Higher FPS
#cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Lower exposure = less blur
#cap.set(cv2.CAP_PROP_GAIN, 50)  # Compensate with gain
#cap.set(cv2.CAP_PROP_EXPOSURE, -7)  # Faster shutter
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # Higher resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 15)
print("✅ Connected to phone camera!")
print(f"Camera URL: {phone_camera_url}")
print("📱 Using LOCAL YOLOv8 model for inference with ENHANCED TRACKING")
print("📱 Tracking object appearance times with LOCK BUFFER SYSTEM")
print("🚀 Features: Kalman Filtering, Motion Prediction, Smooth Tracking")
print("🎯 NEW: Zone-based counting - Objects counted each time they ENTER the zone")
print("Confidence thresholds:")
for cls, conf in CLASS_CONFIDENCE.items():
    print(f"  {cls}: {conf}")
print("Controls:")
print("  'q' - quit | 'r' - reset | 's' - save frame | 'p' - print stats | 'z' - toggle zone | 'd' - save data | 'c' - save CSV")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame read failed")
        time.sleep(0.1)
        continue
    
    frame_count += 1
    
    # Setup counting zone if not already done
    if counting_zone is None:
        setup_counting_zone(frame.shape[1], frame.shape[0])
    
    if not frame_queue.full():
        frame_queue.put(frame.copy())
    
    current_predictions = prediction_cache.copy()
    
    if current_predictions:
        previous_tracks = tracked_objects.copy()
        tracked_objects = update_tracks(current_predictions)
        
        # NEW: Zone-based counting logic with entry/exit tracking
       # NEW: Zone-based counting logic that only counts if the class slot is free
        for obj_id, (current_centroid, cls) in tracked_objects.items():
            if cls in target_classes: # Check all target classes
                in_zone = is_point_in_zone(current_centroid, counting_zone)
                
                # Check for ENTRY
                if in_zone and obj_id not in inside_zone_ids_by_class[cls]:
                    # Only count if no other object of this class is in the zone
                    if len(inside_zone_ids_by_class[cls]) == 0:
                        # This is the first object of its class in the zone -> count it
                        inside_zone_ids_by_class[cls].add(obj_id)
                        print(f"🎯 COUNTED: Object {obj_id} ({cls}) entered zone at {current_centroid}")
                        
                        # Update buffer only for lock types
                        if cls in ['Lock', 'unassembled_lock']:
                            update_buffer(cls, obj_id)
                    else:
                        # Another object of the same class is already in the zone, so just mark it without counting
                        inside_zone_ids_by_class[cls].add(obj_id)
                        print(f"ℹ️ IGNORED: Object {obj_id} ({cls}) entered zone, but another is active.")

                # Check for EXIT
                elif not in_zone and obj_id in inside_zone_ids_by_class[cls]:
                    inside_zone_ids_by_class[cls].remove(obj_id)
                    print(f"🚪 Object {obj_id} ({cls}) EXITED counting zone at {current_centroid}")

                
    
    cleanup_disappeared_objects()
    # Draw visualizations
    if show_counting_zone:
        # Draw counting zone as a rectangle
        x1, y1, x2, y2 = counting_zone
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, "COUNTING ZONE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add zone dimensions text
        zone_info = f"Zone: {zone_width}x{zone_height}"
        cv2.putText(frame, zone_info, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show number of objects currently in zone
        #objects_in_zone = len(inside_zone_ids)
        #zone_count_info = f"Objects in zone: {objects_in_zone}"
        objects_in_zone = sum([len(inside_zone_ids_by_class[cls]) for cls in target_classes])
        zone_count_info = f"Objects in zone: {objects_in_zone}"
        cv2.putText(frame, zone_count_info, (x1, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display session time and buffer information
    session_time = time.time() - session_start_time
    cv2.putText(frame, f"Session: {session_time:.0f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    y_offset = 60
    
    # Display buffer information prominently
    cv2.putText(frame, f"Lock Buffer: {lock_buffer}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    y_offset += 30
    cv2.putText(frame, f"Unassembled Buffer: {unassembled_lock_buffer}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    y_offset += 30
    cv2.putText(frame, f"Total Buffer: {total_lock_buffer}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    y_offset += 40
    
    # Display class appearances
    for cls in target_classes:
        text = f"{cls}: {class_appearances[cls]}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    
    zone_status = "ON" if show_counting_zone else "OFF"
    cv2.putText(frame, f"Zone: {zone_status}", (10, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # Show active object count
    active_count = len([obj_id for obj_id in object_timings if obj_id in tracked_objects])
    cv2.putText(frame, f"Active: {active_count}", (10, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
    
    # FPS
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw detections with enhanced timing info, motion trails, and thinner borders
    for detection in current_predictions:
        if detection['class'] in target_classes:
            x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            
            # Color coding
            if detection['class'] == 'Lock':
                color = (0, 0, 255)  # Red
            elif detection['class'] == 'unassembled_lock':
                color = (255, 0, 0)  # Blue  
            elif detection['class'] == 'Worker':
                color = (0, 255, 0)  # Green
                
            # Draw rectangle with reduced border thickness
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BORDER_THICKNESS)
            
            # Find object ID
            centroid = get_centroid(detection)
            obj_id = None
            for track_id, (track_centroid, track_cls) in tracked_objects.items():
                if abs(centroid[0] - track_centroid[0]) < 20 and abs(centroid[1] - track_centroid[1]) < 20:
                    obj_id = track_id
                    break
            
            # Highlight objects that are in the counting zone
            if obj_id and counting_zone and is_point_in_zone(centroid, counting_zone):
                # Draw a thicker border for objects in the zone
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (255, 255, 0), 3)  # Yellow highlight
                cv2.putText(frame, "IN ZONE", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)
            
            # Draw motion trail and velocity vector
            if obj_id and obj_id in track_points and len(track_points[obj_id]) > 1:
                # Draw motion trail
                points = list(track_points[obj_id])
                for i in range(1, len(points)):
                    thickness = int(np.sqrt(max_track_points / float(i + 1)) * 1.5)
                    cv2.line(frame, points[i-1], points[i], color, thickness)
                
                # Draw velocity vector
                if len(points) >= 2:
                    start_point = points[-1]
                    end_point = (int(start_point[0] + (points[-1][0] - points[-2][0]) * 2),
                                 int(start_point[1] + (points[-1][1] - points[-2][1]) * 2))
                    cv2.arrowedLine(frame, start_point, end_point, (255, 255, 255), 2)
            
            # Enhanced timing display
            if obj_id and obj_id in object_timings:
                current_time = time.time()
                time_detected = current_time - object_timings[obj_id]['first_seen']
                session_time = object_timings[obj_id]['session_time']
                
                # Main label with confidence info
                confidence = detection.get('confidence', 0)
                label = f"{detection['class']} ID:{obj_id} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Time info
                time_label = f"Time: {time_detected:.1f}s"
                cv2.putText(frame, time_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
                
                # Session time (smaller)
                session_label = f"@{session_time:.0f}s"
                cv2.putText(frame, session_label, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    
    cv2.imshow('Enhanced Lock Buffer Tracker - Zone Re-Entry Counting', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Finalize all active objects
        current_time = time.time()
        for obj_id in tracked_objects:
            if obj_id in object_timings and obj_id not in object_total_time:
                total_time = current_time - object_timings[obj_id]['first_seen']
                object_total_time[obj_id] = {
                    'class': object_timings[obj_id]['class'],
                    'total_time': total_time,
                    'first_seen': object_timings[obj_id]['first_seen'],
                    'last_seen': current_time,
                    'session_start': object_timings[obj_id]['session_time'],
                    'session_end': current_time - session_start_time
                }
        break
    elif key == ord('r'):
        # Reset everything including buffers and trackers
        cross_count = 0
        counted_ids.clear()
        inside_zone_ids_by_class = {cls: set() for cls in target_classes}  # Reset the class-based zone tracker
        tracked_objects.clear()
        class_appearances = {cls: 0 for cls in target_classes}
        object_timings.clear()
        object_total_time.clear()
        lock_buffer = 0
        unassembled_lock_buffer = 0
        total_lock_buffer = 0
        buffer_increment_times.clear()
        last_buffer_increment_time = None
        csv_data.clear()
        session_start_time = time.time()
        kalman_trackers.clear()
        track_points.clear()
        print("🔄 All data reset including buffers, trackers, and zone tracking!")
    elif key == ord('s'):
        filename = f"frame_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"💾 Saved: {filename}")
    elif key == ord('p'):
        print_detailed_statistics()
    elif key == ord('d'):
        save_timing_data()
    elif key == ord('c'):
        save_csv_data()
    elif key == ord('z'):  # Changed from 'l' to 'z' for zone toggle
        show_counting_zone = not show_counting_zone
        status = "enabled" if show_counting_zone else "disabled"
        print(f"🎯 Zone {status}")

# Final output
print_detailed_statistics()
save_timing_data()
save_csv_data()

cap.release()
cv2.destroyAllWindows()
print(f"📱 Session completed after {time.time() - session_start_time:.1f} seconds")
print(f"🔒 Final Buffer Counts - Lock: {lock_buffer}, Unassembled: {unassembled_lock_buffer}, Total: {total_lock_buffer}")