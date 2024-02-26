import cv2 as cv
import imutils 

cam=cv.VideoCapture(0)

firstframe=None
area=500

while True:
    _,img=cam.read()
    text="Normal"

    img=imutils.resize(img,width=1000)
    grayimg=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gaussianimg=cv.GaussianBlur(grayimg,(21,21),0)

    if firstframe is None:
        firstframe=gaussianimg
        continue

    imgdiff=cv.absdiff(firstframe,gaussianimg)
    threshimg=cv.threshold(imgdiff,25,255,cv.THRESH_BINARY)[1]
    threshimg=cv.dilate(threshimg,None,iterations=2)

    cnts=cv.findContours(threshimg.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)

    for c in cnts:
        if cv.contourArea(c)<area:
            continue
        (x,y,w,h)=cv.boundingRect(c)
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        text="Moving Object detected"
    print(text)
    cv.putText(img,text,(10,20),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)
    cv.imshow("CameraFeed",img)

    key=cv.waitKey(10)
    print(key)
    if key==ord("a"):
        break
cam.release()
cv.destroyAllWindows()

