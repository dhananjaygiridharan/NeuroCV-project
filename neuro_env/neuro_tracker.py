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
EAR_THRESHOLD = 0.24         # The "line in the sand" — anything below this is a closed eye
FRAME_DEBOUNCE_LIMIT = 3     # Eyes must stay closed for at least 3 frames to count
blink_counter = 0            # Tracks your total score of blinks
frame_counter = 0            # Counts consecutive frames where eyelids are shut


start_time = time.perf_counter()  # Start the timer for session duration tracking
blink_times = deque()  # A deque to store timestamps of recent blinks for rate calculation
last_log_time = time.perf_counter() # Timer to manage periodic logging intervals

window_seconds = 60 # Time window for calculating blinks per minute
log_interval_seconds = 5    # Log data every 5 seconds for quick validation during testing
bpm = 0.0 # Initialize BPM variable
state = "Neutral State" # Initialize cognitive state variable
CSV_PATH = os.path.join(os.path.dirname(__file__), "neuro_focus_data.csv")

#Function to compute the Eye Aspect Ratio (EAR) using Euclidean distance
def calculate_EAR(eye_landmarks, img_w, img_h):
    # Convert the 6 specific landmark points into scaled X, Y pixel arrays
    points = []
    for idx in eye_landmarks:
        pt = face_landmarks[idx]
        points.append(np.array([pt.x * img_w, pt.y * img_h]))
    
    # Extract points based on our list order:
    # points[0]=P1 (outer), points[1]=P4 (inner)
    # points[2]=P2, points[3]=P6 (vertical pair 1)
    # points[4]=P3, points[5]=P5 (vertical pair 2)
    
    # Calculate vertical distances
    v1 = np.linalg.norm(points[2] - points[3])
    v2 = np.linalg.norm(points[4] - points[5])
    
    # Calculate horizontal distance
    h = np.linalg.norm(points[0] - points[1])
    
    # Compute the scale-invariant ratio
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Open connection to the default hardware webcam (0)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cv2.namedWindow('Neuro-Focus Tasks Window', cv2.WINDOW_NORMAL)

def attention_classifier(avg_ear, bpm, total_blinks):
    # Avoid a false "High Focus" label at startup before real blink data exists.
    if total_blinks == 0 and bpm == 0:
        return "Neutral State"

    if avg_ear < 0.18:
        return "Fatigue Present/High sEBR"
    if avg_ear > EAR_THRESHOLD and bpm < 12:
        return "High Focus/Inhibited sEBR"
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

    if detection_result.face_landmarks:
        face_landmarks = detection_result.face_landmarks[0]
        
        # Calculate individual EAR values
        left_ear = calculate_EAR(LEFT_EYE_IDX, img_w, img_h)
        right_ear = calculate_EAR(RIGHT_EYE_IDX, img_w, img_h)
        
        # Task 3: Average the EAR outputs to protect against head rotation artifacts
        avg_ear = (left_ear + right_ear) / 2.0
        
        # --- THE TEMPORAL STATE MACHINE LOGIC ---
        if avg_ear < EAR_THRESHOLD:
            # Eyelids are closed! Tick the consecutive frame timer up by 1
            frame_counter += 1
        else:
            # Eyelids are open! Check if they were just closed long enough for a valid blink
            if frame_counter >= FRAME_DEBOUNCE_LIMIT:
                blink_counter += 1

                while blink_times and time.perf_counter() - blink_times[0] > window_seconds:
                    blink_times.popleft()  # Remove blinks outside the time window

                blink_times.append(time.perf_counter())  # Record the timestamp of the blink
                bpm = (len(blink_times) / 60.0) * 60  # Calculate blinks per minute
            
            # Reset the frame counter to 0 since your eyes are now open
            frame_counter = 0

        state = attention_classifier(avg_ear, bpm, blink_counter)

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
        # Draw crosshairs on the Left Eye points
        for idx in LEFT_EYE_IDX:
            pt = face_landmarks[idx]
            x = int(pt.x * img_w)
            y = int(pt.y * img_h)
            # MARKER_CROSS creates a precise cybernetic crosshair target
            cv2.drawMarker(frame, (x, y), (255, 255, 0), cv2.MARKER_CROSS, markerSize=12, thickness=1)

        # Draw crosshairs on the Right Eye points
        for idx in RIGHT_EYE_IDX:
            pt = face_landmarks[idx]
            x = int(pt.x * img_w)
            y = int(pt.y * img_h)
            cv2.drawMarker(frame, (x, y), (255, 255, 0), cv2.MARKER_CROSS, markerSize=12, thickness=1)
        
        # Put visual indicator text on the screen
        cv2.putText(frame, f"Mean EAR: {avg_ear:.4f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 160, 0), 2)
        cv2.putText(frame, f"Total Blinks: {blink_counter}", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Realtime BPM: {bpm:.2f}", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Cognitive State: {state}", (30, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 150), 2)
        
    # --- HUD CORNER BRACKETS ---
    # Top-Left Corner Brackets (X, Y)
    cv2.line(frame, (20, 20), (100, 20), (255, 160, 0), 2)  # Horizontal line
    cv2.line(frame, (20, 20), (20, 100), (255, 160, 0), 2)  # Vertical line
    
    cv2.imshow('Neuro-Focus Tasks Window', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()