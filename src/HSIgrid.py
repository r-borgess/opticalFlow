import cv2
import numpy as np

# Define new size
new_size = (640, 360) # for example

# Open video file
cap = cv2.VideoCapture('kai.mp4')

# Define parameters for Lucas-Kanade algorithm
lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize variables
prev_frame = None
prev_points = None
prev_rgb = None

# Loop over video frames
while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, new_size)

    # Convert frame to HSI color space
    hsi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, i = cv2.split(hsi)
    
    # Combine channels for display
    hsv = np.concatenate([h[..., np.newaxis], s[..., np.newaxis], i[..., np.newaxis]], axis=-1)

    # Convert HSI image to RGB color space
    rgb = cv2.cvtColor(hsi, cv2.COLOR_HSV2BGR)

    # Scale pixel values to range [0,1]
    rgb = cv2.normalize(rgb, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # Calculate optical flow using Lucas-Kanade algorithm
    curr_points, st, err = cv2.calcOpticalFlowPyrLK(prev_rgb, rgb, prev_points, None, **lk_params)
    
    # Initialize points in first frame
    if prev_frame is None:
        prev_frame = i
        prev_points = cv2.goodFeaturesToTrack(prev_frame, maxCorners=2000, qualityLevel=0.03, minDistance=7)
        prev_rgb = frame
        continue

    # Select only good points
    good_new = curr_points[st == 1]
    good_old = prev_points[st == 1]
    
    # Draw the tracks as arrows
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        prev_rgb = cv2.arrowedLine(prev_rgb, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2, tipLength=0.5)
    
    # Update previous frame and points
    prev_frame = i
    prev_points = good_new.reshape(-1, 1, 2)
    
    # Display the resulting image
    cv2.imshow('Lucas-Kanade Optical Flow', prev_rgb)
    
    # Check for key press and exit if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release video file and close windows
cap.release()
cv2.destroyAllWindows()
