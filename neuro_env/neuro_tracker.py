import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
                print(f"Blink Registered! Total Count: {blink_counter}")
            
            # Reset the frame counter to 0 since your eyes are now open
            frame_counter = 0

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
        cv2.putText(frame, f"AVG EAR: {avg_ear:.2f}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 160, 0), 2)
        # Draw the Blink Score HUD overlay right below the EAR text
        cv2.putText(frame, f"Blinks: {blink_counter}", (30, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
    # --- HUD CORNER BRACKETS ---
    # Top-Left Corner Brackets (X, Y)
    cv2.line(frame, (20, 20), (100, 20), (255, 160, 0), 2)  # Horizontal line
    cv2.line(frame, (20, 20), (20, 100), (255, 160, 0), 2)  # Vertical line
    
    cv2.imshow('Neuro-Focus Tasks Window', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()