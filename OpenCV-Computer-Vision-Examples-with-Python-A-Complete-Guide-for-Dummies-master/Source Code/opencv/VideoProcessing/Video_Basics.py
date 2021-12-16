import cv2

cap=cv2.VideoCapture(0)


width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('mysupervideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20,(width,height)) #20 is frame per second

while True:
    ret,frame=cap.read()   # it return a tuple with ret and frame
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #operations drawig
    writer.write(frame)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  #we will wait 1 millisecond and until we press q we will break out of that
        break

cap.release() #stop capturing the videos
writer.release()
cv2.destroyAllWindows()