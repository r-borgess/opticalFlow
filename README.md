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

## Execution

`python <rblucaskanade.py/rbhornschunck.py>`

Or simply run the code on your favorite IDE.

## Notes

### Parameters

This program was made to be tested with different parameters combinations. It is possible to tweak options related to:
- Resizing/scaling
- Gaussian filtering
- Color format
  - RGB
  - HSV
  - YUV
  - GRAY 
- Saving the output frames

I would like to add also options for the feature detection, search grid and optical flow method. Maybe later.

### Color information

Instead of using black and white frames, the program uses a combination of the luminance (Y) and both crominance (U and V) channels of the frame in order to obtain more accurate flow calculation by using color and intensity information combined. The combination of the channels can be modified in the code. It is also possible to use the RGB format or even the bw version.
