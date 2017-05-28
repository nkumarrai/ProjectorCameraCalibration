import sys
import cv2
import numpy as np
import aruco
import yaml
from numpy.linalg import inv

def computedotproduct(normalVector, marker5, marker1):
	vector = [marker5[0][0]-marker1[0][0], marker5[1][0]-marker1[1][0], marker5[2][0]-marker1[2][0]]
	t = vector[0]*normalVector[0]+vector[1]*normalVector[1]+vector[2]*normalVector[2]
	return abs(t)

def computeRayPlaneIntersection(planePoint, planeNormal, rayDirection):
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

def findRotationAndTranslation():
	with open("stereo_calibration.yaml", 'r') as stream:
		try:
			data = yaml.load(stream)
		except yaml.YAMLError as exc:
			print(exc)
			exit()
	return np.matrix(data["rotation"]), np.matrix(data["translation"])

def reProjectImage(R, T):
	with open("stereo_calibration.yaml", 'r') as stream:
		try:
			data = yaml.load(stream)
		except yaml.YAMLError as exc:
			print(exc)
			exit()
	camparam = aruco.CameraParameters()  
	camparam.setParams(np.array(data["camera_matrix1"]), 
					np.array(data["distortion_coefficients1"]), 
					#np.array([ data["image_width"], data["image_height"] ]))
					np.array([ 640, 480 ]))

	with open("stereo_calibration.yaml", 'r') as stream:
		try:
			data = yaml.load(stream)
		except yaml.YAMLError as exc:
			print(exc)
			exit()
	projMatrix = np.array(data["camera_matrix2"])
	projDistortion = np.array(data["distortion_coefficients2"])

	# create detector and set parameters
	detector = aruco.MarkerDetector()
	params = detector.getParams()

	#open first camera
	v = cv2.VideoCapture(0)
	if not v.isOpened():
		print "unable to open video stream"
		exit()


	imageSize = None

	while True:
		projectedImagePoints = []
		projectedImagePoints2 = [[[100, 100]]]
		ret, img = v.read()
		imageSize = img.shape[::-1]
		orig = np.copy(img)

		markers = detector.detect(img)
		# Try all the corners of marker5 
		marker1 = None
		marker5 = None
		marker3 = None
		pixelValue = [200, 200]
		for marker in markers:
			if marker.id == 1:
				marker1 = marker
			if marker.id == 2:  #marker id is changed.. for sanity check. revert it to 5 later
			 	marker5 = marker
			if marker.id == 3:
				marker3 = marker
		if marker1 != None and marker5 != None and marker3 != None:
			pixelValue = marker5.getCenter()
			cameraMatrix = camparam.CameraMatrix
			distortion = camparam.Distorsion
			print("camera matrix", cameraMatrix)
			print("camera distortion", distortion)
			print("projector matrix", projMatrix)
			print("projector distortion", projDistortion)
			test = np.zeros((1,1,2), dtype=np.float32)
			test[0][0][0] = pixelValue[0]
			test[0][0][1] = pixelValue[1]
			print("before correction Pixel values ")
			print(test)
			pixelCoordinates = cv2.undistortPoints(test, cameraMatrix, distortion)
			print("corrected Pixel values ")
			print(pixelCoordinates)
			pixelCoordinates = np.array([pixelCoordinates[0][0][0], pixelCoordinates[0][0][1], 0.004])
			marker1.calculateExtrinsics(0.030, camparam)
			rvec = marker1.Rvec
			tvec = marker1.Tvec
	#		print("marker translation vector ", tvec, type(tvec))
			rotationMatrix, jacb = cv2.Rodrigues(rvec)
	#		print("rotation matrix")
	#		print(rotationMatrix)
	#		print("normal vector c")
	#		print(rotationMatrix[:, 2])
			normalVector2 = rotationMatrix[:,2]
	#		print("normal vector r")
	#		print(rotationMatrix[2])
			normalVector1 = rotationMatrix[2]
			#####################################################
			rvecForRepr = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
			tvecForRepr = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
			#####################################################
			marker5.calculateExtrinsics(0.030, camparam)
			marker3.calculateExtrinsics(0.030, camparam)
			dotproduct15 = computedotproduct(normalVector1, marker5.Tvec, marker1.Tvec);
			print("dot product 5 and 1 row", dotproduct15)
			dotproduct13 = computedotproduct(normalVector1, marker1.Tvec, marker3.Tvec);
			print("dot product 1 and 3 row", dotproduct13)
			dotproduct15 = computedotproduct(normalVector2, marker5.Tvec, marker1.Tvec);
			print("dot product 5 and 1 column", dotproduct15)
			dotproduct13 = computedotproduct(normalVector2, marker1.Tvec, marker3.Tvec);
			print("dot product 1 and 3 column", dotproduct13)
			if abs(dotproduct13) < 0.05 and abs(dotproduct15) < 0.05:
				print("inside the value")
			else:
				print("probably exiting...")
				#continue
			print("continue")
			computed3Dpoints = computeRayPlaneIntersection(tvec, normalVector1, pixelCoordinates)
			print("computed 3d points c", pixelCoordinates)
			print(computed3Dpoints)
			projectedImagePoints, jacb = cv2.projectPoints(np.array([computed3Dpoints]), rvecForRepr, tvecForRepr, cameraMatrix, distortion)
			print("projected 3d points on image plane using plane intersection")
			print(projectedImagePoints)
			#####################################################
			# Verified that using Tvec works as 3d point.
			computed3Dpoints = marker5.Tvec
			print("computed 3d points just tvec")
			print(computed3Dpoints)
			projectedImagePoints2, jacb = cv2.projectPoints(np.array([computed3Dpoints]), rvecForRepr, tvecForRepr, cameraMatrix, distortion)
			print("projected 3d points on image plane using tvec")
			print(projectedImagePoints2)
			#####################################################
			tempRT = np.concatenate((np.matrix(R), np.matrix(T)), axis=1)
			lastRow = np.matrix([0,0,0,1])
			transformMat = np.concatenate((tempRT,lastRow))
			computed3Dpoints = np.matrix([computed3Dpoints[0][0], computed3Dpoints[1][0], computed3Dpoints[2][0], 1.0])
			computed3Dpoints = transformMat * np.transpose(computed3Dpoints)
			computed3Dpoints = np.array(computed3Dpoints)
			computed3Dpoints = np.array([computed3Dpoints[0][0], computed3Dpoints[1][0], computed3Dpoints[2][0]])
			print("computed 3d points c after transformation", type(normalVector1))
			print(computed3Dpoints)
	
			projectedImagePoints, jacb = cv2.projectPoints(np.array([computed3Dpoints]), rvecForRepr, tvecForRepr, projMatrix, projDistortion)
			print("projected 3d points on image plane for projector")
			projectedImagePoints = (int(projectedImagePoints[0][0][0]), int(projectedImagePoints[0][0][1]))
			print(projectedImagePoints)
			#markerToProject = np.full([800, 1280], 160, dtype=np.uint8)
			#cv2.circle(markerToProject, projectedImagePoints, 20, (255, 0, 0), -1)
			#cv2.namedWindow("marker projection", cv2.WND_PROP_FULLSCREEN)
			#cv2.setWindowProperty("marker projection",cv2.WND_PROP_FULLSCREEN,cv2.cv.CV_WINDOW_FULLSCREEN)
			#cv2.imshow("marker projection", markerToProject)
		
		# show frame
		pixelProj = (int(pixelValue[0]), int(pixelValue[1]))
		#print("projectedImagePoints2 ", projectedImagePoints2[0][0][0], projectedImagePoints2[0][0][1])
		tempdraw = (int(projectedImagePoints2[0][0][0]), int(projectedImagePoints2[0][0][1]))
		cv2.circle(img, pixelProj, 15, (255, 0, 0), -1)
		cv2.circle(img, tempdraw, 15, (0, 255, 0), -1)
		cv2.imshow("frame", cv2.resize(img, None, fx=0.5, fy=0.5))
		key = cv2.waitKey(30)
		if key == 1048689 or key == 1048603: # ESC or 'q'
			break
	return projectedImagePoints


def main():
	R, T = findRotationAndTranslation()
	projectedImagePoints = reProjectImage(R, T)

main();
