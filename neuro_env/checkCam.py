import cv2

# Open connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Force the webcam to its absolute limit by requesting an impossible resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)

# Read back what the hardware actually selected as its maximum
max_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
max_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\n=============================================")
print(f" Your webcam's MAX hardware resolution is:")
print(f" {max_width} x {max_height} pixels")
print("=============================================")

# Clean up
cap.release()