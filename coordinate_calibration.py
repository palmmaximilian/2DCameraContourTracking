#%%
import cv2
import imutils
import numpy as np
import pandas as pd
import configparser
# import pprint
from imutils import perspective
from imutils import contours
import math
import json
from helper_functions import nothing,midpoint,pixelToCartesian
from scipy.spatial import distance as dist


config = configparser.ConfigParser()
config.read('config.ini')
#%%
def getCoordTrackbars():
    global coordMinsize
    global coordMaxsize
    global coordMincanny
    global coordMaxcanny
    global coordBlur
    global distanceOriginToX
    coordMinsize=cv2.getTrackbarPos("minsize","coord")
    coordMaxsize=cv2.getTrackbarPos("maxsize","coord")
    coordMincanny=cv2.getTrackbarPos("min","coord")
    coordMaxcanny=cv2.getTrackbarPos("max","coord")
    coordBlur=cv2.getTrackbarPos("Blur","coord")
    distanceOriginToX=cv2.getTrackbarPos("Distance Origin to X","coord")
    count = coordBlur % 2 
    if (count == 0):
        blur += 1


def createCoordWindow():

    coordMincanny=int(config['COORD']['mincanny'])
    coordMaxcanny=int(config['COORD']['maxcanny'])
    coordMinsize=int(config['COORD']['minsize'])
    coordMaxsize=int(config['COORD']['maxsize'])
    coordBlur=int(config['COORD']['blur'])
    distanceOriginToX=int(config['COORD']['distanceOriginToX'])

    cv2.namedWindow('coord')
    cv2.createTrackbar("min","coord",0,400,nothing)
    cv2.createTrackbar("max","coord",0,400,nothing)
    cv2.createTrackbar("minsize","coord",0,1000,nothing)
    cv2.createTrackbar("maxsize","coord",0,1000,nothing)
    cv2.createTrackbar("Blur", "coord", 1, 21, nothing)
    cv2.createTrackbar("Distance Origin to X", "coord", 1, 500, nothing)


    cv2.setTrackbarPos("min","coord",coordMincanny)
    cv2.setTrackbarPos("max","coord",coordMaxcanny)
    cv2.setTrackbarPos("minsize","coord",coordMinsize)
    cv2.setTrackbarPos("maxsize","coord",coordMaxsize)
    cv2.setTrackbarPos("Blur","coord",coordBlur)
    cv2.setTrackbarPos("Distance Origin to X","coord",distanceOriginToX)


def calibrateCoordinateSystem():
    camera=int(config['General']['camera'])
    videoCaptureObject = cv2.VideoCapture(camera)
    tolerance=50
    pointList=[]
    originpoint=[0,0]
    xpoint=[0,0]
    ypoint=[0,0]
    rotation=0
    incline=0
    createCoordWindow()
    while(True):
        ret,frame = videoCaptureObject.read()
        
        image = frame
        orig = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        getCoordTrackbars()

        gray = cv2.GaussianBlur(gray, (coordBlur, coordBlur), 0)

        edged = cv2.Canny(gray, coordMincanny, coordMaxcanny)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        
        # find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if(len(cnts)>=1):
            # sort the contours from left-to-right and initialize the
            # 'pixels per metric' calibration variable
            (cnts, _) = contours.sort_contours(cnts)
            
            pointList=[]     
                       
            # loop over the contours individually
            for c in cnts:


                # if the contour is not sufficiently large, ignore it
                area=cv2.contourArea(c)
                
                if  area< coordMinsize or area > coordMaxsize:
                    continue
                
                
                # compute the rotated bounding box of the contour

                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)


                cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
                # loop over the original points and draw them
                for (x, y) in box:
                    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                centerpoint=midpoint(tl,br)

                

                cv2.putText(orig, str(cv2.contourArea(c)),
                    (int(trbrX + 20), int(trbrY+20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)

                # Reduce list to relevant points, remove duplicates
                if(len(pointList)==0):
                    pointList.append(centerpoint)
                included=False
                for point in pointList:
                    if(
                        (centerpoint[0]>point[0]-tolerance and centerpoint[0]<point[0]+tolerance) 
                        and 
                        (centerpoint[1]>point[1]-tolerance and centerpoint[1]<point[1]+tolerance)
                        ):
                        included=True
                if(not included):
                    pointList.append(centerpoint)
                    



        (h, w) = orig.shape[:2]

        # print(pointList)  
        # figure out which point is which
        if(len(pointList)==3):

            for idx, point in enumerate (pointList):
                pointList[idx]=pixelToCartesian(pointList[idx],gray.shape)

            # find Y Point
            currentPos=0
            for i in range(1,2):
                if(pointList[currentPos][1]<pointList[i][1]):
                    currentPos=i
            ypoint=pointList[currentPos]
            del pointList[currentPos]


            if(pointList[0][0]>pointList[1][0]):
                xpoint=pointList[0]
                originpoint=pointList[1]
            else:
                xpoint=pointList[1]
                originpoint=pointList[0]


            convertedPoint=pixelToCartesian(originpoint,gray.shape)
            cv2.putText(orig, "origin",
                    (int(convertedPoint[0]+50), int(convertedPoint[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)
            cv2.putText(orig, str(originpoint),
                    (int(convertedPoint[0]), int(convertedPoint[1]-50)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)

            convertedPointX=pixelToCartesian(xpoint,gray.shape)
            cv2.putText(orig, "x",
                    (int(convertedPointX[0]+50), int(convertedPointX[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)
            cv2.putText(orig, str(xpoint),
                    (int(convertedPointX[0]), int(convertedPointX[1]-50)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)

            convertedPointY=pixelToCartesian(ypoint,gray.shape)
            cv2.putText(orig, "y",
                    (int(convertedPointY[0]+50), int(convertedPointY[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)     
            cv2.putText(orig, str(ypoint),
                    (int(convertedPointY[0]), int(convertedPointY[1]-50)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)

            
            incline=(xpoint[1]-originpoint[1])/(xpoint[0]-originpoint[0])
            rotation=math.atan(incline)
            distanceOriginX=math.dist(convertedPoint,convertedPointX)
            pixelsPerMetric = distanceOriginX /distanceOriginToX
            # print(pixelsPerMetric)
        
        cv2.imshow('Capturing Video',orig)
        cv2.imshow('coord',edged)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            videoCaptureObject.release()
            cv2.destroyAllWindows()
            break


    
    config['COORD']['mincanny']=str(coordMincanny)
    config['COORD']['maxcanny']=str(coordMaxcanny)
    config['COORD']['minsize']=str(coordMinsize)
    config['COORD']['maxsize']=str(coordMaxsize)
    config['COORD']['blur']=str(coordBlur)
    config['COORD']['originpoint']=str(originpoint)
    config['COORD']['xpoint']=str(xpoint)
    config['COORD']['ypoint']=str(ypoint)
    config['COORD']['rotation']=str(rotation)
    config['COORD']['incline']=str(incline)
    config['CAMERA']['pixelpermm']=str(pixelsPerMetric)
    config['COORD']['distanceOriginToX']=str(distanceOriginToX)

    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    print("Settings saved. Calibration done!")
