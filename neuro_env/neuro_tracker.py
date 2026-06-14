import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import time
from collections import deque
import os

print("\n--- Initializing Neuro-Focus Tracking Engine ---")

# Configure the Tasks API Face Landmarker model
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Define the 6 landmark indexes for both eyes (Horizontal, Vertical 1, Vertical 2)
LEFT_EYE_IDX = [33, 133, 160, 144, 158, 153]
RIGHT_EYE_IDX = [362, 263, 385, 380, 387, 373]

# --- STATE MACHINE CONFIGURATION ---
EAR_THRESHOLD = 0.24         # Anything below this is a closed eye
FRAME_DEBOUNCE_LIMIT = 3     # Eyes must stay closed for at least 3 frames to count
blink_counter = 0            # Tracks your total score of blinks
frame_counter = 0            # Counts consecutive frames where eyelids are shut

calibration_duration = 5.0
is_calibrated = False
calibration_ear_scores = [] 


# --- NEW VARIABLES FOR DETERMINISTIC LOGIC ---
BASELINE_BPM = 15.0          # Set this to your normal resting blinks per minute
blink_durations = deque(maxlen=5) # Stores the duration of the last 5 blinks for a rolling average
avg_blink_duration = 0.0     # Initialize rolling average variable
current_blink_start = 0.0    # Timer for when a blink begins

start_time = time.perf_counter()  # Start the timer for session duration tracking
blink_times = deque()  # A deque to store timestamps of recent blinks for rate calculation
last_log_time = time.perf_counter() # Timer to manage periodic logging intervals

window_seconds = 60 # Time window for calculating blinks per minute
log_interval_seconds = 5    # Log data every 5 seconds for quick validation during testing
bpm = 0.0 # Initialize BPM variable
state = "Neutral State" # Initialize cognitive state variable
CSV_PATH = os.path.join(os.path.dirname(__file__), "neuro_focus_data.csv")

# Function to compute the Eye Aspect Ratio (EAR) using Euclidean distance
def calculate_EAR(eye_landmarks, img_w, img_h, face_landmarks):
    points = []
    for idx in eye_landmarks:
        pt = face_landmarks[idx]
        points.append(np.array([pt.x * img_w, pt.y * img_h]))
    
    v1 = np.linalg.norm(points[2] - points[3])
    v2 = np.linalg.norm(points[4] - points[5])
    h = np.linalg.norm(points[0] - points[1])
    
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Open connection to the default hardware webcam (0)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1552)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1552)
cv2.namedWindow('Neuro-Focus Tasks Window', cv2.WINDOW_NORMAL)

def attention_classifier(bpm, baseline_bpm, avg_blink_duration, total_blinks, bpm_drop_threshold=0.85, duration_threshold=0.4, is_calibrated=True):
    if not is_calibrated:
        return "Calibrating Baseline..."
    
    """
    Categorizes cognitive load deterministically based on BPM drops and blink duration.
    """
    if total_blinks == 0 and bpm == 0:
        return "Neutral State"

    cond_focus = bpm < (baseline_bpm * bpm_drop_threshold)
    cond_fatigue = avg_blink_duration >= duration_threshold
    
    if cond_focus and cond_fatigue:
        return "High Focus & Fatigue Present"
    elif cond_focus:
        return "High Focus/Inhibited sEBR"
    elif cond_fatigue:
        return "Fatigue Present"
    
    return "Neutral State"
    
def append_to_csv(timestamp, total_blinks, bpm, mean_ear, state):
    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Total_Blinks",
                "Realtime_BPM",
                "Mean_EAR",
                "Cognitive_State"
            ])

        writer.writerow([
            timestamp,
            total_blinks,
            f"{bpm:.2f}",
            f"{mean_ear:.4f}",
            state
        ])


while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    img_h, img_w, _ = frame.shape

    elapsed_time = time.perf_counter() - start_time

    
    if detection_result.face_landmarks:
        face_landmarks = detection_result.face_landmarks[0]
        
        # Calculate individual EAR values
        left_ear = calculate_EAR(LEFT_EYE_IDX, img_w, img_h, face_landmarks)
        right_ear = calculate_EAR(RIGHT_EYE_IDX, img_w, img_h, face_landmarks)
        
        avg_ear = (left_ear + right_ear) / 2.0
        if not is_calibrated:
            if elapsed_time <= calibration_duration:
                calibration_ear_scores.append(avg_ear)
                calibration_ui_text = f"Calibrating... {calibration_duration - elapsed_time:.1f}s left"
            else:
                mean_open_ear = sum(calibration_ear_scores) / len(calibration_ear_scores)
                EAR_THRESHOLD = mean_open_ear * 0.75
                Baseline_BPM = 15.0
                is_calibrated = True

        else:
        # --- THE TEMPORAL STATE MACHINE LOGIC ---
            if avg_ear < EAR_THRESHOLD:
                if frame_counter == 0:
                    current_blink_start = time.perf_counter()
                frame_counter += 1
            else:
                if frame_counter >= FRAME_DEBOUNCE_LIMIT:
                    blink_counter += 1
                        
                    blink_end_time = time.perf_counter()
                    duration = blink_end_time - current_blink_start
                    blink_durations.append(duration)
                        
                    avg_blink_duration = sum(blink_durations) / len(blink_durations)

                    while blink_times and time.perf_counter() - blink_times[0] > window_seconds:
                        blink_times.popleft()

                    blink_times.append(blink_end_time)
                    bpm = (len(blink_times) / window_seconds) * 60.0
                    
                frame_counter = 0

        # Run deterministic classifier
        state = attention_classifier(
            bpm=bpm, 
            baseline_bpm=BASELINE_BPM, 
            avg_blink_duration=avg_blink_duration, 
            total_blinks=blink_counter
        )

        # --- CSV LOGGING INTERVAL (RESTORED) ---
        if time.perf_counter() - last_log_time >= log_interval_seconds:
            append_to_csv(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                blink_counter,
                bpm,
                avg_ear,
                state
            )
            last_log_time = time.perf_counter()

        # --- JARVIS EYE TARGETING RETICLES ---
        for idx in LEFT_EYE_IDX:
            pt = face_landmarks[idx]
            x = int(pt.x * img_w)
            y = int(pt.y * img_h)
            cv2.drawMarker(frame, (x, y), (255, 255, 0), cv2.MARKER_CROSS, markerSize=12, thickness=1)

        for idx in RIGHT_EYE_IDX:
            pt = face_landmarks[idx]
            x = int(pt.x * img_w)
            y = int(pt.y * img_h)
            cv2.drawMarker(frame, (x, y), (255, 255, 0), cv2.MARKER_CROSS, markerSize=12, thickness=1)
        
        # UI Information HUD
        text_to_display = state if is_calibrated else calibration_ui_text
        hud_color = (0, 255, 150) if is_calibrated else (0, 165, 255)

        cv2.putText(frame, f"Mean EAR: {avg_ear:.4f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 160, 0), 2)
        cv2.putText(frame, f"Total Blinks: {blink_counter}", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Realtime BPM: {bpm:.2f}", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Avg Blink Duration: {avg_blink_duration:.2f}s", (30, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, f"Cognitive State: {text_to_display}", (30, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)
        

            # --- BOTTOM RIGHT TELEMETRY HUD ---
        # 1. Format the absolute start time into a readable string
        start_clock_str = time.strftime("%H:%M:%S", time.localtime(start_time))
        
        # 2. Convert raw elapsed seconds into hours, minutes, and seconds
        m, s = divmod(int(elapsed_time), 60)
        h, m = divmod(m, 60)
        elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"
        
        # 3. Create the text strings
        start_text = f"Session Start: {start_clock_str}"
        elapsed_text = f"Elapsed Time: {elapsed_str}"
        
        # 4. Dynamically calculate X position so the text pushes left from the right edge
        # (Assuming standard fonts, we offset by about 450 pixels from the right margin)
        hud_x_position = max(img_w - 450, 30) 
        
        # 5. Render the strings onto the bottom right screen space
        cv2.putText(frame, start_text, (hud_x_position, img_h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 160, 0), 2)
        cv2.putText(frame, elapsed_text, (hud_x_position, img_h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
       
    # HUD Corner Brackets
    cv2.line(frame, (20, 20), (100, 20), (255, 160, 0), 2)
    cv2.line(frame, (20, 20), (20, 100), (255, 160, 0), 2)
    
    cv2.imshow('Neuro-Focus Tasks Window', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()