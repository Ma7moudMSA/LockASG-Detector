from roboflow import Roboflow
import numpy as np

# Initialize Roboflow model
rf = Roboflow(api_key="FkJacBs27YbPy2WDKfuT")
project = rf.workspace().project("my-first-project-tpq1u")
model = project.version("2").model

# Run video prediction
job_id, signed_url, expire_time = model.predict_video(
    "Test.mp4",
    fps=5,
    prediction_type="batch-video",
)

results = model.poll_until_video_results(job_id)

# Configuration
counting_line_x = 100  # Vertical line position on left side
target_classes = ['Lock', 'unassembled_lock']  # Note: using underscore as shown in your output

# Tracking variables
tracked_objects = {}
next_object_id = 0
counted_ids = set()
cross_count = 0

def get_centroid(detection):
    """Extract centroid from Roboflow detection format"""
    # Roboflow detection format: {'x': center_x, 'y': center_y, 'width': w, 'height': h, 'class': 'class_name', 'confidence': conf}
    if 'x' in detection and 'y' in detection:
        return (int(detection['x']), int(detection['y']))
    else:
        # Fallback for other possible formats
        return (0, 0)

def update_tracks(detections):
    """Update object tracking"""
    global next_object_id
    
    if not detections:
        return {}
    
    current_objects = []
    for det in detections:
        if det is None:  # Skip None detections
            continue
        centroid = get_centroid(det)
        class_name = det.get('class', 'unknown')
        current_objects.append((centroid, class_name))
    
    new_tracked = {}
    
    for centroid, cls in current_objects:
        min_distance = float('inf')
        matched_id = None
        
        # Match with existing objects based on x-coordinate proximity
        for obj_id, last_x in tracked_objects.items():
            distance = abs(centroid[0] - last_x)
            if distance < min_distance and distance < 80:
                min_distance = distance
                matched_id = obj_id
        
        if matched_id is not None:
            new_tracked[matched_id] = centroid[0]
        else:
            new_tracked[next_object_id] = centroid[0]
            next_object_id += 1
    
    return new_tracked

# Process results
print("Processing video predictions...")
print(f"Counting line at x = {counting_line_x}")
print(f"Target classes: {target_classes}")
print("-" * 50)

try:
    # Extract the project-specific predictions
    project_key = 'my-first-project-tpq1u'
    
    if project_key in results:
        predictions_data = results[project_key]
        frame_offsets = results.get('frame_offset', [])
        time_offsets = results.get('time_offset', [])
        
        print(f"Found {len(predictions_data)} frames with predictions")
        
        previous_tracks = {}
        
        for frame_idx, frame_predictions in enumerate(predictions_data):
            if frame_predictions is None:
                continue
                
            # Filter for target classes
            filtered_detections = []
            if isinstance(frame_predictions, list):
                for det in frame_predictions:
                    if det and det.get('class') in target_classes:
                        filtered_detections.append(det)
            
            if filtered_detections:
                print(f"Frame {frame_idx + 1}: Found {len(filtered_detections)} target objects")
                for det in filtered_detections:
                    print(f"  - {det.get('class')} at ({det.get('x')}, {det.get('y')}) confidence: {det.get('confidence', 0):.2f}")
            
            # Update tracking
            previous_tracks = tracked_objects.copy()
            tracked_objects = update_tracks(filtered_detections)
            
            # Check for crossings
            for obj_id, current_x in tracked_objects.items():
                if obj_id in previous_tracks:
                    previous_x = previous_tracks[obj_id]
                    # Object crossed from right to left
                    if previous_x > counting_line_x and current_x <= counting_line_x and obj_id not in counted_ids:
                        cross_count += 1
                        counted_ids.add(obj_id)
                        frame_time = time_offsets[frame_idx] if frame_idx < len(time_offsets) else frame_idx
                        print(f"✓ Frame {frame_idx + 1} (t={frame_time}s): Object {obj_id} crossed the line! Total: {cross_count}")
            
            # Print current tracked positions for debugging
            if tracked_objects:
                positions = [f"ID{obj_id}:x={x}" for obj_id, x in tracked_objects.items()]
                print(f"Frame {frame_idx + 1}: Current positions - {', '.join(positions)}")
    
    else:
        print(f"Project key '{project_key}' not found in results")
        print("Available keys:", list(results.keys()))

except Exception as e:
    print(f"Error processing results: {e}")
    import traceback
    traceback.print_exc()

print("-" * 50)
print(f"🎯 FINAL COUNT: {cross_count} objects crossed the counting line")

# Optional: Print summary statistics
if 'frame_offset' in results:
    print(f"📊 SUMMARY:")
    print(f"   - Total frames processed: {len(results['frame_offset'])}")
    print(f"   - Objects that crossed: {len(counted_ids)}")
    print(f"   - Counting line position: x = {counting_line_x}")
