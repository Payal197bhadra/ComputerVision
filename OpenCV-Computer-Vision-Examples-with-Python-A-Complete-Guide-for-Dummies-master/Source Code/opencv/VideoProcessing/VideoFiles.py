import cv2
import time

cap=cv2.VideoCapture('mysupervideo.mp4')
if cap.isOpened()==False:
    print("Error File Not found")

while cap.isOpened():
    ret,frame=cap.read()
    if ret==True:
        #Writer 20FPS
        time.sleep(1/20) #to sleep the frames for 20FPS
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord(
                'q'):  # we will wait 1 millisecond and until we press q we will break out of that
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()