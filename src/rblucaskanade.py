import cv2
import numpy as np

resize = False
scale = 0.3
gauss = True
bw = True
hsv = not bw
save = True

cap = cv2.VideoCapture('input.gif')
j = 0

if resize:
    new_size = (int((cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)), int((cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)))

sift = cv2.SIFT_create()

lk_params = dict(winSize=(50, 50), 
                 maxLevel=10, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.003))

prev_frame = None
prev_points = None

while True:
    ret, frame = cap.read()
    frameReserva = frame
    if not ret:
        break

    if resize:
        frame = cv2.resize(frame, new_size)
    if gauss:
        frame = cv2.GaussianBlur(frame, (9, 9), 0)  
    if bw:
        eq = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('bw', eq)
    if hsv:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        # Split the HSI frame into its channels
        h, s, v = cv2.split(frame)
        eq = h + s + v
        cv2.imshow('yuv', eq)
    
    if prev_frame is None:
        prev_frame = eq
        prev_kp, prev_desc = sift.detectAndCompute(prev_frame, None)
        prev_points = np.float32([kp.pt for kp in prev_kp]).reshape(-1, 1, 2)
        continue

    kp, desc = sift.detectAndCompute(eq, None)
    curr_points, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, eq, prev_points, None, **lk_params)

    good_new = curr_points[st == 1]
    good_old = prev_points[st == 1]

    blank = np.zeros_like(frame)

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        if abs(a - c) > 20 or abs(b - d) > 20:
            continue
        blank = cv2.arrowedLine(blank ,(int(c), int(d)), (int(a), int(b)), (255, 255, 255), thickness=1, line_type=4,tipLength=0.35)
        frame = cv2.arrowedLine(frame ,(int(c), int(d)), (int(a), int(b)), (255, 255, 255), thickness=1, line_type=4,tipLength=0.35)
    
    prev_frame = eq
    prev_kp = kp
    prev_desc = desc
    prev_points = good_new.reshape(-1, 1, 2)

    cv2.imshow('Original', frameReserva)
    cv2.imshow('Merged', frame)
    cv2.imshow('Optical Flow', blank)

    if save:
        cv2.imwrite('C:/Users/rodri/OneDrive/Documentos/visao2/lucas kanade/originais/frame'+str(j)+'.png',frameReserva)
        cv2.imwrite('C:/Users/rodri/OneDrive/Documentos/visao2/lucas kanade/frames/frame'+str(j)+'.png', frame)
        cv2.imwrite('C:/Users/rodri/OneDrive/Documentos/visao2/lucas kanade/hsi/frame'+str(j)+'.png', eq)
        cv2.imwrite('C:/Users/rodri/OneDrive/Documentos/visao2/lucas kanade/segs/frame'+str(j)+'.png', blank)
        j += 1

    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break