import cv2
import time
import threading
import math
from queue import Queue
import json
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# Load your trained local YOLOv8 model weights
model = YOLO('best_1.pt')  # Change this to your model path!

# Configuration - Removed Roboflow API settings
target_classes = ['Lock', 'unassembled_lock', 'Worker']

# Class-specific confidence thresholds
CLASS_CONFIDENCE = {
    'Worker': 0.35,          # Higher confidence for Worker
    'Lock': 0.5,           # Standard confidence for Lock
    'unassembled_lock': 0.3 # Standard confidence for unassembled_lock
}

# Visual settings
BORDER_THICKNESS = 1  # Reduced from 3 to 1 for thinner borders

# Enhanced tracking variables
tracked_objects = {}
next_object_id = 0
counted_ids = set()
cross_count = 0
counting_line_x = None

# NEW: Buffer system for locks
lock_buffer = 0
unassembled_lock_buffer = 0
total_lock_buffer = 0  # Combined buffer for both lock types
buffer_increment_times = []  # Track time between buffer increments
last_buffer_increment_time = None

# Line visibility toggle
show_counting_line = True  # Changed to True by default

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

def get_centroid(detection):
    x = detection.get('x', 0)
    y = detection.get('y', 0)
    return (int(x), int(y))

def log_to_csv(event_type, data):
    """Add event to CSV data list"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_time = time.time() - session_start_time
    
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
    global next_object_id
    current_time = time.time()
    
    target_detections = [det for det in detections if det.get('class') in target_classes]
    current_objects = []
    
    for det in target_detections:
        centroid = get_centroid(det)
        current_objects.append((centroid, det.get('class')))
    
    new_tracked = {}
    used_detections = set()
    
    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # First pass: match existing objects
    for obj_id, (last_centroid, last_cls) in tracked_objects.items():
        min_distance = float('inf')
        matched_idx = None
        
        for idx, (centroid, cls) in enumerate(current_objects):
            if idx in used_detections:
                continue
            if cls == last_cls:
                distance = euclidean_distance(centroid, last_centroid)
                if distance < min_distance and distance < 80:
                    min_distance = distance
                    matched_idx = idx
        
        if matched_idx is not None:
            centroid, cls = current_objects[matched_idx]
            new_tracked[obj_id] = (centroid, cls)
            used_detections.add(matched_idx)
            
            # Update timing
            if obj_id in object_timings:
                object_timings[obj_id]['last_seen'] = current_time
                current_duration = current_time - object_timings[obj_id]['first_seen']
                object_timings[obj_id]['current_duration'] = current_duration
    
    # Second pass: create new objects
    for idx, (centroid, cls) in enumerate(current_objects):
        if idx not in used_detections:
            new_tracked[next_object_id] = (centroid, cls)
            class_appearances[cls] += 1
            object_timings[next_object_id] = {
                'class': cls,
                'first_seen': current_time,
                'last_seen': current_time,
                'current_duration': 0.0,
                'session_time': current_time - session_start_time
            }
            print(f"🆕 New {cls} detected! ID: {next_object_id} at {current_time - session_start_time:.1f}s into session")
            
            # Log new detection to CSV
            log_to_csv('New_Detection', {
                'class': cls,
                'id': next_object_id,
                'info': f"First appeared at {current_time - session_start_time:.1f}s"
            })
            
            next_object_id += 1
    
    # Calculate final times for disappeared objects
    for obj_id in tracked_objects:
        if obj_id not in new_tracked and obj_id in object_timings:
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
                
                # Log object completion to CSV
                log_to_csv('Object_Completed', {
                    'class': object_timings[obj_id]['class'],
                    'id': obj_id,
                    'info': f"Visible for {total_time:.2f}s"
                })
    
    return new_tracked

def save_timing_data():
    """Save timing data to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detection_times_{timestamp}.json"
    
    data = {
        'session_info': {
            'start_time': session_start_time,
            'duration': time.time() - session_start_time,
            'timestamp': datetime.now().isoformat()
        },
        'class_appearances': class_appearances,
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
    current_time = time.time()
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
    """Save all logged data to CSV file"""
    if not csv_data:
        print("❌ No CSV data to save")
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detection_log_{timestamp}.csv"
    
    df = pd.DataFrame(csv_data)
    df.to_csv(filename, index=False)
    
    print(f"📊 CSV data saved to {filename}")
    print(f"📈 Total events logged: {len(csv_data)}")
    return filename

def print_detailed_statistics():
    """Enhanced statistics with buffer information"""
    print("\n" + "="*60)
    print("📈 DETAILED DETECTION STATISTICS")
    print("="*60)
    
    session_duration = time.time() - session_start_time
    print(f"📅 Session Duration: {session_duration:.1f} seconds")
    
    # Buffer statistics
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
                print(f"   ID {obj_id} ({timing_data['class']}): {active_duration:.1f}s active (appeared at {session_time:.1f}s)")
    
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

# Updated inference function using local YOLOv8 model
def infer_frame(frame):
    """
    Run inference using local YOLOv8 model instead of Roboflow API
    """
    try:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        detections = []
        
        # Extract predictions from results
        for box, cls_id, score in zip(results[0].boxes.xyxy.cpu().numpy(),
                                      results[0].boxes.cls.cpu().numpy(),
                                      results[0].boxes.conf.cpu().numpy()):
            
            # Get class name from model
            class_name = model.names[int(cls_id)]
            
            # Create detection dictionary in the same format as before
            detection = {
                'x': int((box[0] + box[2]) / 2),  # Center X
                'y': int((box[1] + box[3]) / 2),  # Center Y
                'width': int(box[2] - box[0]),    # Width
                'height': int(box[3] - box[1]),   # Height
                'class': class_name,
                'confidence': float(score)
            }
            
            # Apply class-specific confidence thresholds
            threshold = CLASS_CONFIDENCE.get(class_name, 0.4)
            if detection['confidence'] >= threshold and class_name in target_classes:
                detections.append(detection)
        
        return detections
        
    except Exception as e:
        print(f"Error in local inference: {e}")
        return []

def inference_worker():
    global prediction_cache, last_prediction_time
    
    while True:
        try:
            if not frame_queue.empty():
                frame = frame_queue.get()
                current_time = time.time()
                
                # Run inference every second (adjust as needed for performance)
                if current_time - last_prediction_time >= 1.0:
                    prediction_cache = infer_frame(frame)
                    last_prediction_time = current_time
                    if prediction_cache:
                        target_objects = [p for p in prediction_cache if p.get('class') in target_classes]
                        # Only print occasionally to reduce spam
                        if int(current_time) % 5 == 0:
                            print(f"🔍 LOCAL: Found {len(target_objects)} target objects")
            else:
                time.sleep(0.1)
        except Exception as e:
            print(f"Inference worker error: {e}")
            time.sleep(1)

# Start background inference thread
inference_thread = threading.Thread(target=inference_worker, daemon=True)
inference_thread.start()

# Connect to phone camera
phone_camera_url = "http://192.168.137.243:8080/video"
cap = cv2.VideoCapture(phone_camera_url)

if not cap.isOpened():
    alternative_urls = [
        "http://192.168.137.107:8080/videofeed",
        "http://192.168.137.107:8080/video.mjpg"
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
cap.set(cv2.CAP_PROP_FPS, 15)

print("✅ Connected to phone camera!")
print(f"Camera URL: {phone_camera_url}")
print("📱 Using LOCAL YOLOv8 model for inference")
print("📱 Tracking object appearance times with LOCK BUFFER SYSTEM")
print("Confidence thresholds:")
for cls, conf in CLASS_CONFIDENCE.items():
    print(f"  {cls}: {conf}")
print("Controls:")
print("  'q' - quit | 'r' - reset | 's' - save frame | 'p' - print stats | 'l' - toggle line | 'd' - save data | 'c' - save CSV")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame read failed")
        time.sleep(0.1)
        continue
    
    frame_count += 1
    
    if counting_line_x is None:
        counting_line_x = frame.shape[1] // 2
    
    if not frame_queue.full():
        frame_queue.put(frame.copy())
    
    current_predictions = prediction_cache.copy()
    
    if current_predictions:
        previous_tracks = tracked_objects.copy()
        tracked_objects = update_tracks(current_predictions)
        
        # FIXED: Line crossing logic - now works regardless of show_counting_line
        # and specifically targets Lock and unassembled_lock
        for obj_id, (current_centroid, cls) in tracked_objects.items():
            if obj_id in previous_tracks:
                previous_centroid, _ = previous_tracks[obj_id]
                current_x = current_centroid[0]
                previous_x = previous_centroid[0]
                
                # Check if Lock or unassembled_lock crossed the line (right to left)
                if (cls in ['Lock', 'unassembled_lock'] and 
                    previous_x > counting_line_x and current_x <= counting_line_x and 
                    obj_id not in counted_ids):
                    
                    counted_ids.add(obj_id)
                    update_buffer(cls, obj_id)  # This handles the buffer increment and timing
    
    # Draw visualizations
    if show_counting_line:
        cv2.line(frame, (counting_line_x, 0), (counting_line_x, frame.shape[0]), (0, 255, 0), 4)
        cv2.putText(frame, "COUNTING LINE", (counting_line_x + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
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
    
    line_status = "ON" if show_counting_line else "OFF"
    cv2.putText(frame, f"Line: {line_status}", (10, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # Show active object count
    active_count = len([obj_id for obj_id in object_timings if obj_id in tracked_objects])
    cv2.putText(frame, f"Active: {active_count}", (10, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
    
    # FPS
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw detections with enhanced timing info and thinner borders
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
    
    cv2.imshow('Lock Buffer Tracker', frame)
    
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
        # Reset everything including buffers
        cross_count = 0
        counted_ids.clear()
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
        print("🔄 All data reset including buffers!")
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
    elif key == ord('l'):
        show_counting_line = not show_counting_line
        status = "enabled" if show_counting_line else "disabled"
        print(f"📏 Line {status}")

# Final output
print_detailed_statistics()
save_timing_data()
save_csv_data()

cap.release()
cv2.destroyAllWindows()
print(f"📱 Session completed after {time.time() - session_start_time:.1f} seconds")
print(f"🔒 Final Buffer Counts - Lock: {lock_buffer}, Unassembled: {unassembled_lock_buffer}, Total: {total_lock_buffer}")
