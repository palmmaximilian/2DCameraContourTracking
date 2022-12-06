import math

def nothing(x):
    # print("value changed")
    pass

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def pixelToCartesian(point,resolution):
    convertedX=point[0]
    convertedY=resolution[0]-point[1]
    return([convertedX,convertedY])

def transformCoordinates(point,originPoint,rotationOrigin):
    distance=math.dist(point,originPoint)
    incline=(point[1]-originPoint[1])/(point[0]-originPoint[0])
    rotation=math.atan(incline)
    newX=distance*math.cos(rotation-rotationOrigin)
    newY=distance*math.sin(rotation-rotationOrigin)
    return([newX,newY])
