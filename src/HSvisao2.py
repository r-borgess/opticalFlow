import cv2
import numpy as np
from pyflow import pyflow
cap = cv2.VideoCapture('narutin.gif')
i = 0

ret, frame1 = cap.read()
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
FRAME1 = np.float32(frame1)

while True:
    #read next frame
    ret, frame2 = cap.read()

    if not ret:
        break

    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame2 = np.float32(frame2)
    
    flow = pyflow(frame1, frame2, alpha=0.012, ratio=0.75, minWidth=20, nOuterFPIterations=7, nInnerFPIterations=1, nSORIterations=30, colType=0)

    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('Optical Flow', rgb)

    frame1 = frame2

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break