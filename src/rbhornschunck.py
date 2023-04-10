import cv2
import numpy as np

resize = False
scale = 0.5
gauss = True
bw = True
yuv = not bw
save = True

cap = cv2.VideoCapture('bron.mp4')
cap2 = cap
j = 0

if resize:
    new_size = (int((cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)), int((cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)))

sift = cv2.SIFT_create()

# set parameters for dense optical flow
farneback_params = dict(
    pyr_scale=0.5,
    levels=1,
    winsize=10,
    iterations=5,
    poly_n=1,
    poly_sigma=0.5,
    flags=3
)

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
        frame = cv2.GaussianBlur(frame, (11, 11), 0)  
    if bw:
        eq = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('bw', eq)
    if yuv:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(frame)
        eq = y + u + v
        cv2.imshow('yuv', eq)
    
    if prev_frame is None:
        prev_frame = eq
        prev_kp, prev_desc = sift.detectAndCompute(prev_frame, None)
        continue

    # compute dense optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_frame, eq, None, **farneback_params)

    # visualize the dense optical flow
    hsv = np.zeros_like(frame)
    hsv[...,1] = 255

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    ret3 ,thresh = cv2.threshold(bgr,20,255,cv2.THRESH_BINARY)

    cv2.imshow('Optical Flow', bgr)
    cv2.imshow('segmented', thresh)

    prev_frame = eq

    cv2.imshow('Original', frameReserva)

    if save:
        cv2.imwrite('C:/Users/rodri/OneDrive/Documentos/visao2/horn schunck/originais/frame'+str(j)+'.png',frameReserva)
        cv2.imwrite('C:/Users/rodri/OneDrive/Documentos/visao2/horn schunck/frames/frame'+str(j)+'.png', bgr)
        cv2.imwrite('C:/Users/rodri/OneDrive/Documentos/visao2/horn schunck/hsi/frame'+str(j)+'.png', eq)
        cv2.imwrite('C:/Users/rodri/OneDrive/Documentos/visao2/horn schunck/segs/frame'+str(j)+'.png',thresh)
        j += 1
    
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
