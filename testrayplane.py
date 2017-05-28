'''
This code is to trying to test the ray plane intersection
by assuming a ray, a plane equation and a point on the plane.

I've also mentioned some of the links that I used as a reference.
#https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection
#https://www.khanacademy.org/partner-content/pixar/rendering/rendering-2/e/ray-intersection-with-plane
#https://www.siggraph.org/education/materials/HyperGraph/raytrace/rayplane_intersection.htm
#https://github.com/mattdesl/ray-plane-intersection

Contributors - Naveen Kumar Rai (nkrai@cs.stonybrook.edu)
'''
import sys
import cv2
import numpy as np
import aruco
import yaml
from numpy.linalg import inv

#marker1 and marker5 are just normal vectors. Nothing to be afraid of.
def computedotproduct(normalVector, marker5, marker1):
    vector = [marker5[0][0]-marker1[0][0], marker5[1][0]-marker1[1][0], marker5[2][0]-marker1[2][0]]
    t = vector[0]*normalVector[0]+vector[1]*normalVector[1]+vector[2]*normalVector[2]
    return abs(t)

'''
There are two functions named as "computeRayPlaneIntersectionNotUnit" and "computeRayPlaneIntersectionUnit".
I was just trying to check if the rayDirection vector should be a unit vector or not. Hence, the two functions.
Result - The magnitude of the rayDirection vector doesn't affect anything.
'''
def computeRayPlaneIntersectionNotUnit(planePoint, planeNormal, rayDirection):
    rayDirection = np.squeeze(np.asarray(rayDirection))
    magnitude = rayDirection[0]*rayDirection[0] + rayDirection[1]*rayDirection[1] + rayDirection[2]*rayDirection[2]
    magnitude = np.sqrt(magnitude)
    rayDirection = [rayDirection[0]/magnitude, rayDirection[1]/magnitude, rayDirection[2]/magnitude]
    magnitude = rayDirection[0]*rayDirection[0] + rayDirection[1]*rayDirection[1] + rayDirection[2]*rayDirection[2]
    print("rayDirection ", magnitude, rayDirection)
    print("planePoint ", planePoint)
    print("planeNormal ", planeNormal)
    numerator = planePoint[0]*planeNormal[0] +planePoint[1]*planeNormal[1] +planePoint[2]*planeNormal[2]
    denominator = rayDirection[0]*planeNormal[0] +rayDirection[1]*planeNormal[1] +rayDirection[2]*planeNormal[2]
    t = numerator/denominator
    return np.array([np.array(rayDirection[0]*t), np.array(rayDirection[1]*t), np.array(rayDirection[2]*t)])

def computeRayPlaneIntersectionUnit(planePoint, planeNormal, rayDirection):
    rayDirection = np.squeeze(np.asarray(rayDirection))
    magnitude = rayDirection[0]*rayDirection[0] + rayDirection[1]*rayDirection[1] + rayDirection[2]*rayDirection[2]
    print("rayDirection ", magnitude, rayDirection)
    print("planePoint ", planePoint)
    print("planeNormal ", planeNormal)
    numerator = planePoint[0]*planeNormal[0] +planePoint[1]*planeNormal[1] +planePoint[2]*planeNormal[2]
    denominator = rayDirection[0]*planeNormal[0] +rayDirection[1]*planeNormal[1] +rayDirection[2]*planeNormal[2]
    t = numerator/denominator
    return np.array([np.array(rayDirection[0]*t), np.array(rayDirection[1]*t), np.array(rayDirection[2]*t)])

def main():
# x + y + z = 1; (0,0,0) -> (1,2,3)
# x + y + z = 1; (0,0,0) -> (5,2,1)
	planePoint = [0.0, 0.0, 1.0]
	planeNormal = [1.0, 1.0, 1.0]
	#rayDirection = [1.0, 2.0, 3.0]
	rayDirection = [5.0, 2.0, 1.0]
	computed3dpoint = computeRayPlaneIntersectionUnit(planePoint, planeNormal, rayDirection)
	print("computed3dpoint not a unit vector", computed3dpoint)
	planeNormal = [2.0, 2.0, 2.0]
	computed3dpoint = computeRayPlaneIntersectionUnit(planePoint, planeNormal, rayDirection)
	print("computed3dpoint unit vector", computed3dpoint)

main()
