import cv2
# --- CORRECTED IMPORT ---
import pyapriltags as apriltag # Import the library and give it a shorter alias

# --- 1. Initialize the Camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- 2. Initialize the AprilTag Detector ---
# Create a detector object with the correct family
# The 'families' argument is passed directly to the Detector.
detector = apriltag.Detector(families="tag36h11")

print("Starting webcam feed. Hold up a tag. Press 'q' to quit.")

# --- 3. Main Loop ---
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect tags
    results = detector.detect(gray)

    # --- 4. Draw Detections ---
    if results:
        print(f"Found {len(results)} tags.")
        for r in results:
            # Get the four corner points of the tag
            (ptA, ptB, ptC, ptD) = r.corners
            ptA = (int(ptA[0]), int(ptA[1]))
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))

            # Draw the bounding box
            cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
            cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
            cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
            cv2.line(frame, ptD, ptA, (0, 255, 0), 2)

            # Draw the center and put the tag ID
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(r.tag_id), (ptA[0], ptA[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # --- 5. Display the Result ---
    cv2.imshow("AprilTag Detector", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. Clean Up ---
cap.release()
cv2.destroyAllWindows()
print("Webcam feed stopped.")
