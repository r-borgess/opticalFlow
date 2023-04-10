# opticalFlow
Motion detection based on both Horn-Schunck and Lucas-Kanade optical flow calculation methods.

- Image processing
- Color space conversion and channel splitting:
  - RGB to YUV
- Feature detection (SIFT)
- Optical flow calculation:
  - Dense flow (HS)
  - Sparse flow (LK)
- Motion based segmentation

## Input
A sequence of consecutive frames (gif, mp4, etc) defined in the code.

## Output

- Original frame
- Color conversion results
- Splitted channels
- Optical flow calculation results
- Segmented frame
