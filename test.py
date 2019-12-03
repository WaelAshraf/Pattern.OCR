import cv2
import numpy as np
# Load as greyscale
im = cv2.imread('0.png', cv2.IMREAD_GRAYSCALE)

# Invert
im = 255 - im

# Calculate horizontal projection
proj = np.sum(im,0)

# Create output image same height as text, 500 px wide
m = np.max(proj)
w = 49
result = np.zeros((proj.shape[0],49))

# Draw a line for each row
for row in range(im.shape[0]):
    cv2.line(result, (0,row), (int(proj[row]*w/m),row), (255,255,255), 1)

cv2.imshow('marked areas', result)