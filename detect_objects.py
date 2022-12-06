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
from helper_functions import midpoint,pixelToCartesian,transformCoordinates


config = configparser.ConfigParser()
config.read('config.ini')
#%%




def detectObjects():
    #calibrate camera
    
    camera=int(config['General']['camera'])
    cameraPixelPerMM=float(config['CAMERA']['pixelpermm'])
    cameraBlur=int(config['CAMERA']['blur'])
    cameraMincanny=int(config['CAMERA']['mincanny'])
    cameraMaxcanny=int(config['CAMERA']['maxcanny'])
    cameraMinsize=int(config['CAMERA']['minsize'])
    cameraMaxsize=int(config['CAMERA']['maxsize'])
    cameraBorder=int(config['CAMERA']['border'])


    originpoint=json.loads(config['COORD']['originpoint'])
    xpoint=json.loads(config['COORD']['xpoint'])
    ypoint=json.loads(config['COORD']['ypoint'])
    rotationOrigin=float(config['COORD']['rotation'])
    



    videoCaptureObject = cv2.VideoCapture(camera)
    objectList=[]
    
    while(True):
        ret,frame = videoCaptureObject.read()
        image = frame
        orig = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Draw Coordinate System
        convertedPoint=pixelToCartesian(originpoint,gray.shape)  
        convertedXPoint=pixelToCartesian(xpoint,gray.shape)  
        convertedYPoint=pixelToCartesian(ypoint,gray.shape)  
        cv2.line(orig, (int(convertedPoint[0]), int(convertedPoint[1])),(int(convertedXPoint[0]), int(convertedXPoint[1])),(255,0,0),2)
        cv2.line(orig, (int(convertedPoint[0]), int(convertedPoint[1])),(int(convertedYPoint[0]), int(convertedYPoint[1])),(0,255,0),2)
        
        cv2.circle(orig, (int(convertedPoint[0]), int(convertedPoint[1])), 10, (0, 0, 255), -1)
        cv2.circle(orig, (int(convertedXPoint[0]), int(convertedXPoint[1])), 10, (0, 0, 255), -1)
        cv2.circle(orig, (int(convertedYPoint[0]), int(convertedYPoint[1])), 10, (0, 0, 255), -1)


        

        gray = cv2.GaussianBlur(gray, (cameraBlur, cameraBlur), 0)

        edged = cv2.Canny(gray, cameraMincanny, cameraMaxcanny)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        # find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if(len(cnts)>=1):

            # sort the contours from left-to-right and initialize the
            # 'pixels per metric' calibration variable
            (cnts, _) = contours.sort_contours(cnts)
                                
            # loop over the contours individually
            for c in cnts:

                # if the contour is not sufficiently large, ignore it
                area=cv2.contourArea(c)
                
                if  area< cameraMinsize or area > cameraMaxsize:
                    continue

                # compute the rotated bounding box of the contour
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)

                # #check that object is fully in the image

                coord=np.asarray(box)
                (h, w) = orig.shape[:2]
                if(coord[0,0]<=cameraBorder):
                    continue
                if(coord[0,1]<=cameraBorder):
                    continue
                if(coord[1,0]>=w-cameraBorder):
                    continue
                if(coord[1,1]<=cameraBorder):
                    continue
                if(coord[2,0]>=w-cameraBorder):
                    continue
                if(coord[2,1]>=h-cameraBorder):
                    continue
                if(coord[3,0]<=cameraBorder):
                    continue
                if(coord[3,1]>=h-cameraBorder):
                    continue

                cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
                # loop over the original points and draw them
                for (x, y) in box:
                    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)
                dA = math.dist((tltrX, tltrY), (blbrX, blbrY))
                dB = math.dist((tlblX, tlblY), (trbrX, trbrY))


                centerpoint=midpoint(tl,br)
                cv2.circle(orig, (int(centerpoint[0]), int(centerpoint[1])), 5, (0, 0, 255), -1)
                convertedPoint=pixelToCartesian(centerpoint,gray.shape)
                convertedPoint=transformCoordinates(convertedPoint,originpoint,rotationOrigin)

                
                convertedPoint[0]=convertedPoint[0]/ cameraPixelPerMM
                convertedPoint[1]=convertedPoint[1]/ cameraPixelPerMM

                convertedPoint[0]=float(f'{convertedPoint[0]:.2f}')
                convertedPoint[1]=float(f'{convertedPoint[1]:.2f}')

                convertedTr=pixelToCartesian(tr,gray.shape)
                convertedTl=pixelToCartesian(tl,gray.shape)
                convertedBl=pixelToCartesian(bl,gray.shape)

                if(math.dist(convertedTl,convertedTr)>math.dist(convertedTl,convertedBl)):
                    incline=(convertedTr[1]-convertedTl[1])/(convertedTr[0]-convertedTl[0])
                    rotation=math.atan(incline)-rotationOrigin
                else:
                    incline=(convertedBl[1]-convertedTl[1])/(convertedBl[0]-convertedTl[0])
                    rotation=math.atan(incline)-rotationOrigin

                if(rotation<(-1.5708)):
                    rotation=rotation+3.14159
                # print(rotation)
                
                objectList.append([convertedPoint,rotation])



                if(dA>0 and dB>0):
                    dimA = dA / cameraPixelPerMM
                    dimB = dB / cameraPixelPerMM
                    # cv2.putText(orig, "{:.1f}mm".format(dimA),
                    #     (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.65, (0, 0, 0), 2)
                    # cv2.putText(orig, "{:.1f}mm".format(dimB),
                    #     (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.65, (0, 0, 0), 2)
                cv2.putText(orig, str(convertedPoint),
                    (int(trbrX + 20), int(trbrY+20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)
                cv2.putText(orig, str(math.degrees(rotation)),
                    (int(trbrX + 20), int(trbrY+40)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)
                



        (h, w) = orig.shape[:2]
        cv2.rectangle(orig, (cameraBorder,cameraBorder), (w-cameraBorder,h-cameraBorder), (0, 0, 255) , 2)




        cv2.imshow('Capturing Video',orig)  
        # print(objectList)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            videoCaptureObject.release()
            cv2.destroyAllWindows()
            break