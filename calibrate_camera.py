#%%
import cv2
import imutils
import numpy as np
import configparser
# import pprint
from imutils import perspective
from imutils import contours
import math
import json
from helper_functions import nothing,midpoint


config = configparser.ConfigParser()
config.read('config.ini')
#%%


def getCameraTrackbars():
    global cameraMinsize
    global cameraMaxsize
    global cameraMincanny
    global cameraMaxcanny
    global cameraBorder
    global cameraBlur
    cameraMinsize=cv2.getTrackbarPos("minsize","contour")
    cameraMaxsize=cv2.getTrackbarPos("maxsize","contour")
    cameraMincanny=cv2.getTrackbarPos("min","contour")
    cameraMaxcanny=cv2.getTrackbarPos("max","contour")

    cameraBorder=cv2.getTrackbarPos("border","contour")
    cameraBlur=cv2.getTrackbarPos("Blur","contour")
    count = cameraBlur % 2 
    if (count == 0):
        blur += 1

def createCameraWindow():

    cameraMincanny=int(config['CAMERA']['mincanny'])
    cameraMaxcanny=int(config['CAMERA']['maxcanny'])
    cameraMinsize=int(config['CAMERA']['minsize'])
    cameraMaxsize=int(config['CAMERA']['maxsize'])
    cameraBorder=int(config['CAMERA']['border'])
    cameraBlur=int(config['CAMERA']['blur'])

    cv2.namedWindow('contour')
    cv2.createTrackbar("min","contour",0,400,nothing)
    cv2.createTrackbar("max","contour",0,400,nothing)
    cv2.createTrackbar("minsize","contour",100,100000,nothing)
    cv2.createTrackbar("maxsize","contour",100,100000,nothing)
    cv2.createTrackbar("border","contour",0,200,nothing)
    cv2.createTrackbar("Blur", "contour", 1, 21, nothing)


    cv2.setTrackbarPos("min","contour",cameraMincanny)
    cv2.setTrackbarPos("max","contour",cameraMaxcanny)
    cv2.setTrackbarPos("minsize","contour",cameraMinsize)
    cv2.setTrackbarPos("maxsize","contour",cameraMaxsize)
    cv2.setTrackbarPos("border","contour",cameraBorder)
    cv2.setTrackbarPos("Blur","contour",cameraBlur)

def calibrateCamera():
    #calibrate camera
    camera=int(config['General']['camera'])
    cameraPixelPerMM=float(config['CAMERA']['pixelpermm'])
    videoCaptureObject = cv2.VideoCapture(camera)
    createCameraWindow()
    while(True):
        ret,frame = videoCaptureObject.read()
        image = frame
        orig = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        getCameraTrackbars()

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

                if(dA>0 and dB>0):
                    dimA = dA / cameraPixelPerMM
                    dimB = dB / cameraPixelPerMM
                    cv2.putText(orig, "{:.1f}mm".format(dimA),
                        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 2)
                    cv2.putText(orig, "{:.1f}mm".format(dimB),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 2)
                cv2.putText(orig, str(cv2.contourArea(c)),
                    (int(trbrX + 20), int(trbrY+20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)


        (h, w) = orig.shape[:2]
        cv2.rectangle(orig, (cameraBorder,cameraBorder), (w-cameraBorder,h-cameraBorder), (0, 0, 255) , 2)

        cv2.imshow('Capturing Video',orig)
        cv2.imshow('contour',edged)                        

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            videoCaptureObject.release()
            cv2.destroyAllWindows()
            break


    config['CAMERA']['pixelpermm']=str(cameraPixelPerMM)
    config['CAMERA']['mincanny']=str(cameraMincanny)
    config['CAMERA']['maxcanny']=str(cameraMaxcanny)
    config['CAMERA']['minsize']=str(cameraMinsize)
    config['CAMERA']['maxsize']=str(cameraMaxsize)
    config['CAMERA']['border']=str(cameraBorder)
    config['CAMERA']['blur']=str(cameraBlur)

    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    print("Settings saved. Calibration done!")
