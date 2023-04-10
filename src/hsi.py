import cv2
import numpy as np

# Initialize variables
cap = cv2.VideoCapture('narutin.gif')
sift = cv2.SIFT_create()
farneback_params = dict(
    pyr_scale=0.5,
    levels=1,
    winsize=2,
    iterations=5,
    poly_n=1,
    poly_sigma=3,
    flags=3
)
prev_frame = None

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is not None:
        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, **farneback_params)

        # Compute magnitude of the flow
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])

        # Create binary mask using threshold
        thresh = cv2.threshold(mag, 5, 255, cv2.THRESH_BINARY)[1]

        # Apply binary mask to original frame to extract moving objects
        moving_objects = cv2.bitwise_and(frame, frame, mask=thresh)

        # Display results
        cv2.imshow('Original', frame)
        cv2.imshow('Moving objects', moving_objects)

    prev_frame = gray

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
