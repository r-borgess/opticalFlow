import cv2
import numpy as np

# Define new size
new_size = (640, 640) # for example

# Open video file
cap = cv2.VideoCapture('narutin.gif')
cap2 = cap
i = 0
#telaPreta = np.zeros((new_size[0], new_size[1], 3))

# Define parameters for Lucas-Kanade algorithm
lk_params = dict(winSize=(25, 25),
                 maxLevel=10,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.3))

# Define search grid
grid_size = int((cap.get(cv2.CAP_PROP_FRAME_WIDTH) * cap.get(cv2.CAP_PROP_FRAME_HEIGHT))/5000)
#grid_size = int((new_size[0] * new_size[1])/5000)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
x = np.linspace(0, width, grid_size)
y = np.linspace(0, height, grid_size)
search_points = np.array(np.meshgrid(np.float32(x), np.float32(y))).T.reshape(-1, 1, 2)

# Initialize variables
prev_frame = None
prev_points = None

# Loop over video frames
while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        break
    
    #frame = cv2.resize(frame, new_size)

     # Apply Gaussian blur to the entire frame
    frame = cv2.GaussianBlur(frame, (15, 15), 0)
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    #gray = frame
    
    # Initialize points in first frame
    if prev_frame is None:
        prev_frame = eq
        prev_points = search_points
        continue
    '''
    if prev_frame is None:
        prev_frame = gray
        prev_points = cv2.goodFeaturesToTrack(prev_frame, maxCorners=100, qualityLevel=0.03, minDistance=10)
        continue
    '''    
    # Check if prev_points is not None and has a valid shape and data type
    #if prev_points is not None and prev_points.shape[1:] != (1, 2) or prev_points.dtype != np.float32:
    #    prev_points = None

    # Calculate optical flow using Lucas-Kanade algorithm
    curr_points, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, eq, prev_points, None, **lk_params)

    # Select only good points
    good_new = curr_points[st == 1]
    good_old = prev_points[st == 1]
    
    # Draw the tracks as arrows
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        if abs(a - c) > 20 or abs(b - d) > 20:
            continue
        frame = cv2.arrowedLine(frame ,(int(c), int(d)), (int(a), int(b)), (0, 255, 0), thickness=1, line_type=1,tipLength=0.15)
    
    # Update previous frame and points
    prev_frame = eq
    prev_points = good_new.reshape(-1, 1, 2)

    ret2, frameReserva = cap.read()

    cv2.imwrite('C:/Users/rodri/OneDrive/Documentos/visao2/frames/frame'+str(i)+'.png', frame)
    i += 4
    
    # Display the resulting image
    cv2.imshow('Original', frameReserva)
    cv2.imshow('Optical Flow', frame)
    
    # Check for key press and exit if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release video file and close windows