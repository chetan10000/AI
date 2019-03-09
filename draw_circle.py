import cv2
import numpy as np

# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),200,(255,0,0),-1)#### these are the perameter like x,y, radius,colour,thickness####

# Create a black image, a window and bind the function to window
img = np.zeros((700,700,3), np.uint8)###it creates a screen window###
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)####function to perform ####

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break