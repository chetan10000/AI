import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('C:\\\\Users\chetan\Desktop\Python\haarcascades\\\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\\\Users\chetan\Desktop\Python\haarcascades\\\\haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        while (eyes.any()):
            
            cap2=cv2.VideoCapture('C:\\\\Users\chetan\Desktop\Python\\\\funny.3gp')
            while(cap2.isOpened()):
                
                ret, frame = cap2.read()

                gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                cv2.imshow('frame',gray1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()

            cap2.release()
    
        cv2.destroyAllWindows()
        
            
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
