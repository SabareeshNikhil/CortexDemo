#Identify pupils. Based on beta 1

import numpy as np
import cv2
import time
import matplotlib
matplotlib.use("Tkagg")
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0) #640,480
w = 640
h = 480

#ret, frame = cap.read()
#pupilFrame = frame
#pupilO = frame
pupilX = list()
pupilY = list()
f, (ax1, ax2) = plt.subplots(1, 2)

while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
        #downsample
        #frameD = cv2.pyrDown(cv2.pyrDown(frame))
        #frameDBW = cv2.cvtColor(frameD,cv2.COLOR_RGB2GRAY)

                #detect face
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
                faces = cv2.CascadeClassifier('haarcascade_eye.xml')
                detected = faces.detectMultiScale(frame, 1.3, 5)

                #faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                #detected2 = faces.detectMultiScale(frameDBW, 1.3, 5)
                
                pupilFrame = frame
                pupilO = frame
                windowClose = np.ones((5,5),np.uint8)
                windowOpen = np.ones((2,2),np.uint8)
                windowErode = np.ones((2,2),np.uint8)

                #draw square
                for (x,y,w,h) in detected:
                        cv2.rectangle(frame, (x,y), ((x+w),(y+h)), (0,0,255),1)        
                        cv2.line(frame, (x,y), ((x+w,y+h)), (0,0,255),1)
                        cv2.line(frame, (x+w,y), ((x,y+h)), (0,0,255),1)
                        pupilFrame = cv2.equalizeHist(frame[int(y+(h*.25)):(y+h), x:(x+w)])
                        pupilO = pupilFrame
                        pupilFrame = cv2.adaptiveThreshold(pupilFrame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2) 
                        #pupilO = pupilFrame
                        ret, pupilFrame = cv2.threshold(pupilFrame,55,255,cv2.THRESH_BINARY)                #50 ..nothin 70 is better
                        pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_CLOSE, windowClose)
                        pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_ERODE, windowErode)
                        pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_OPEN, windowOpen)

                        #so above we do image processing to get the pupil..
                        #now we find the biggest blob and get the centriod
                        threshold = cv2.inRange(pupilFrame,250,255)                #get the blobs
                        _, contours, hierarchy = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                        
                        #if there are 3 or more blobs, delete the biggest and delete the left most for the right eye
                        #if there are 2 blob, take the second largest
                        #if there are 1 or less blobs, do nothing
                        
                        if len(contours) >= 2:
                                #find biggest blob
                                maxArea = 0
                                MAindex = 0                        #to get the unwanted frame 
                                distanceX = []                #delete the left most (for right eye)
                                currentIndex = 0 
                                for cnt in contours:
                                        area = cv2.contourArea(cnt)
                                        center = cv2.moments(cnt)
                                        if center['m00'] != 0:
                                                cx,cy = int(center['m10']/center['m00']), int(center['m01']/center['m00'])
                                        else :
                                                cx,cy = 0, 0
                                        distanceX.append(cx)        
                                        if area > maxArea:
                                                maxArea = area
                                                MAindex = currentIndex
                                        currentIndex = currentIndex + 1
                
                                del contours[MAindex]                #remove the picture frame contour
                                del distanceX[MAindex]
                        
                        eye = 'right'

                        if len(contours) >= 2:                #delete the left most blob for right eye
                                if eye == 'right':
                                        edgeOfEye = distanceX.index(min(distanceX))
                                else:
                                        edgeOfEye = distanceX.index(max(distanceX))        
                                del contours[edgeOfEye]
                                del distanceX[edgeOfEye]

                        if len(contours) >= 1:                #get largest blob
                                maxArea = 0
                                for cnt in contours:
                                        area = cv2.contourArea(cnt)
                                        if area > maxArea:
                                                maxArea = area
                                                largeBlob = cnt
                                        
                        if len(largeBlob) > 0:        
                                cv2.drawContours(pupilO,[largeBlob],-1,(0,255,0),3)
                                #cv2.drawKeypoints(pupilO, largeBlob, pupilO)
                                center = cv2.moments(largeBlob)
                                cx,cy = int(center['m10']/center['m00']), int(center['m01']/center['m00'])
                                cv2.circle(pupilO,(cx,cy),5,255,-1)

                                pupilX.append(cx)
                                pupilY.append(cy)
                                #show picture
                                cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
                                cv2.resizeWindow('frame', 640, 480)
                                cv2.imshow('frame',pupilO)
                                #cv2.waitKey(30)
                #cv2.imshow('frame2',pupilFrame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if len(pupilX) > 30 :
                        pupilX_conv = np.convolve(pupilX, np.ones(30)/30, mode='valid')
                        pupilY_conv = np.convolve(pupilY, np.ones(30)/30, mode='valid')
                else :
                        pupilX_conv = pupilX
                        pupilY_conv = pupilY
                if len(pupilX) > 100 :
                        pupilX_conv100 = np.convolve(pupilX, np.ones(100)/100, mode='valid')
                        pupilY_conv100 = np.convolve(pupilY, np.ones(100)/100, mode='valid')
                else :
                        pupilX_conv100 = pupilX_conv
                        pupilY_conv100 = pupilY_conv

                #ax1.scatter(range(len(pupilX_conv)), pupilX_conv)
                #ax2.scatter(range(len(pupilY_conv)), pupilY_conv)
                #plt.show()
                #cv2.waitKey(30)
        #else:
                #break
cap.release()
cv2.destroyAllWindows()

ax1.invert_yaxis()
ax1.scatter(pupilX_conv, range(len(pupilX_conv)))
#ax1.plot(pupilX_conv100, range(len(pupilX_conv100)))
ax2.scatter(range(len(pupilY_conv)), pupilY_conv, color = 'orange')
#ax2.plot(range(len(pupilY_conv100)), pupilY_conv100, color = 'orange')
#ax3.scatter(pupilX_conv, pupilY_conv)
#ax3.plot(pupilX_conv, pupilY_conv, color='orange')
plt.show()

