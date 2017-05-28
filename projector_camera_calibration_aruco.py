'''
This code is our main code which is doing the following things -
1. Generate the markers to project. 
2. Create the image to project (which is nothing but a gray image with value = 160)
	and two markers around the center.
3. Total markers that I am using right now is 6. Id's are from 1-6.

Contributors - 
1. Naveen Kumar Rai (nkrai@cs.stonybrook.edu)
2. Roy Shilkrot (roys@cs.stonybrook.edu)
'''
import sys
import os
import cv2
import numpy as np
import aruco
import yaml
from numpy.linalg import inv

'''
1. Draw cube on all of the markers except marker id 5 and 6.
2. Here, it was assumed that the cardboard (plane) is at (0,0,0).
3. Rvec and Tvec of the markers are computed using calculateExtrinsics
	function of the aruco markers.
'''
def drawCube(markers, camparam, img):
	for m in markers:
		if m.id != 5 and m.id != 6:
			m.calculateExtrinsics(0.030, camparam, False)
			halfSize = m.ssize/2
			cubePoints = [[-halfSize, -halfSize, 0], [halfSize, -halfSize, 0], [halfSize, halfSize, 0], [-halfSize, halfSize, 0], 
						[-halfSize, -halfSize, m.ssize], [halfSize, -halfSize, m.ssize], [halfSize, halfSize, m.ssize], [-halfSize, halfSize, m.ssize]]
			imagePoints, jacb = cv2.projectPoints(np.array([cubePoints]), m.Rvec, m.Tvec, camparam.CameraMatrix, camparam.Distorsion)
			imagePoints = np.int32(imagePoints).reshape(-1,2)
			for i in range(4):
				cv2.line(img, tuple(imagePoints[i].ravel()), tuple(imagePoints[(i+1)%4].ravel()), (0, 0, 255, 255), 1, cv2.CV_AA)
			for i in range(4):
				cv2.line(img, tuple(imagePoints[i+4].ravel()), tuple(imagePoints[4+(i+1)%4].ravel()), (0, 0, 255, 255), 1, cv2.CV_AA)
			for i in range(4):
				cv2.line(img, tuple(imagePoints[i].ravel()), tuple(imagePoints[i+4].ravel()), (0, 0, 255, 255), 1, cv2.CV_AA)

	'''
	#This is using the inbuilt functions from aruco library.
	for marker in markers:
		marker.calculateExtrinsics(0.030, camparam, False)
		aruco.CvDrawingUtils.draw3dCube(img, marker, camparam)
		#aruco.CvDrawingUtils.draw3dAxis(img, marker, camparam)
	'''
	return img



'''
General Info - 
1. For stereo calibration, we need 2d points of the projected markers from projector plane,
	2d points of the projected markers from the camera plane and 3d points of the projected markers.
2. I am using ray plane intersection method to compute 3d points for the projected markers.
3. To apply ray plane intersection method, I need atleast one marker from either id 5 or 6 
	(which are the projected markers) for rayDirection vector, and atleast one marker from id 1 to 4 
	(which are the cardboard markers) for plane equation, plane point and plane normal.
	
Here, I am trying to do sanity checks to make sure that the computations aren't wrong.
Sanity checks - 
1. Get the center of the marker. 
2. Use undistortPoints to convert the pixel coordinates (x,y) to real world units in mm.
3. Calculate Extrinsics to get Rvec and Tvec.
4. You've to calculate Rvec and Tvec w.r.t camera (which means camera is at (0,0,0)).
5. Use calcualted pixel coordiantes in the real world as rayDirection, Rvec and Tvec as plane normal
	and plane equation. 
6. Once you've computed the 3d point, use cv2.ProjectPoints to project the 3dpoint back in the camera 
	plane. ****The sanity check here is that the projected point should coincide with the original one.****
7. ****Another sanity check**** is manually checking the rotationMatrix, Tvec and 3dpoint values whether
	they align or represent the actual setup that I have or not. Use inch tape to see how much distant
	the table is from the projector and camera.
8. ****One more sanity check**** Tvec represent the translation vector so if we use it as the 3d point
	of the marker in the real world and compute 2d point in the camera plane using projectPoints, the resultant
	co-ordinates should coincide with the original getCenter coordinates or should be around that only.
'''
def sanityCheck(markers, camparam, img):
	marker1 = None
	marker5 = None
	for marker in markers:
		if marker.id == 1:
			marker1 = marker
		if marker.id == 5: 
			marker5 = marker
	marker5corners = []
	objPoints = []
	if marker1 != None and marker5 != None:
		pixelValue = marker5.getCenter()
		cameraMatrix = camparam.CameraMatrix
		distortion = camparam.Distorsion
		test = np.zeros((1,1,2), dtype=np.float32)
		# This is done to handle the specific structure required in undistortPoints.
		test[0][0][0] = pixelValue[0]
		test[0][0][1] = pixelValue[1]
		print("***************** start *****************")
		print("before correction Pixel values ")
		print(test)
		pixelCoordinates = cv2.undistortPoints(test, cameraMatrix, distortion)
		print("corrected Pixel values ")
		print(pixelCoordinates)
		pixelCoordinates = np.array([pixelCoordinates[0][0][0], pixelCoordinates[0][0][1], 1])
		marker1.calculateExtrinsics(0.030, camparam, False)
		rvec = marker1.Rvec
		tvec = marker1.Tvec
		print("marker translation vector")
		print(tvec)
		print("rvec from marker")
		print(rvec)
		rotationMatrix, jacb = cv2.Rodrigues(rvec)
		print("rotation matrix")
		print(rotationMatrix)
		print("row vector since we are using a transpose.")
		print(rotationMatrix[2])
		normalVector = rotationMatrix[2]
		'''
		Convert the rotation matrix and Tvec for the co-ordinate system where camera is at (0,0,0).
		'''
		#####################################################
		unityMatrix = np.identity(3)
		#rvecForRepr = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
		rvecForRepr, jacb = cv2.Rodrigues(unityMatrix)
		print("rvecForRepr", rvecForRepr)
		tvecForRepr = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
		#####################################################
		newTvec = np.matrix(rotationMatrix) * np.matrix(tvec)
		print("newTvec", newTvec)
		newTvec = -1 * newTvec
		newTvec = np.array(newTvec)
		print("newTvec", newTvec)
		#####################################################
		projectedImagePoints, jacb = cv2.projectPoints(np.array([newTvec]), rvecForRepr, tvecForRepr, cameraMatrix, distortion)
		print("projected newTvec back on the camera plane")
		print(projectedImagePoints)
		print("projected newTvec -> marker1 getcenter")
		print(marker1.getCenter())
		computed3Dpoints = computeRayPlaneIntersection(newTvec, normalVector, pixelCoordinates)
		print("computed 3d points")
		print(computed3Dpoints)
		projectedImagePoints, jacb = cv2.projectPoints(np.array([computed3Dpoints]), rvecForRepr, tvecForRepr, cameraMatrix, distortion)
		print("projected 3d points on image plane c")
		print(projectedImagePoints)

'''
Just a copy of the above code to return projectedImagePoints.
'''
def getReprojectedImagePoint(pixelValue, marker1, camparam, zAxis):
	marker1.calculateExtrinsics(0.030, camparam, False)
	halfSize = 0
	if zAxis == True:
		halfSize = marker1.ssize/2
	cameraMatrix = camparam.CameraMatrix
	distortion = camparam.Distorsion
	test = np.zeros((1,1,2), dtype=np.float32)
	test[0][0][0] = pixelValue[0]
	test[0][0][1] = pixelValue[1]
	pixelCoordinates = cv2.undistortPoints(test, cameraMatrix, distortion)
	pixelCoordinates = np.array([pixelCoordinates[0][0][0], pixelCoordinates[0][0][1], 1])
	marker1.calculateExtrinsics(0.030, camparam, False)
	rvec = marker1.Rvec
	tvec = marker1.Tvec
	rotationMatrix, jacb = cv2.Rodrigues(rvec)
	normalVector = rotationMatrix[:,2]
	#####################################################
	rvecForRepr = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
	tvecForRepr = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
	#####################################################
	newTvec = np.matrix(rotationMatrix) * np.matrix(tvec)
	newTvec = -1 * newTvec
	newTvec = np.array(newTvec)
	#####################################################
	computed3Dpoints = computeRayPlaneIntersection(tvec, normalVector, pixelCoordinates)
	#@TODO: check this out
	computed3Dpoints[2] = computed3Dpoints[2] + halfSize
	projectedImagePoints, jacb = cv2.projectPoints(np.array([computed3Dpoints]), rvecForRepr, tvecForRepr, cameraMatrix, distortion)
	return projectedImagePoints
		

def getImagePoints(marker1, m, camparam, img):
	corners = []
	for i, point in enumerate(m):
		corners.append(point)

	for i in range(len(corners)):
		pixelValue = corners[i]
		pixelValue = getReprojectedImagePoint(pixelValue, marker1, camparam, True)
		corners.append(pixelValue)
	return corners

'''
1. Draw cubes on projected markers (id 5 and 6). It uses marker1 (which is on cardboard)
	for the plane equation and plane normal. 
2. Computes the 2d points in camera plane for the projected markers to draw the cube.
'''
def drawCubeForProjectedMarkers(markers, camparam, img):
	marker1 = None
	sanityCheck(markers, camparam, img)
	for m in markers:
		if m.id == 1:
			marker1 = m
	for m in markers:
		if marker1 != None:
			if m.id == 5 or m.id == 6:
				m.calculateExtrinsics(0.030, camparam, False)
				imgPts = getImagePoints(marker1, m, camparam, img)
				imagePoints = []
				for i in range(4):
					imagePoints.append(imgPts[i].tolist())
				for i in range(4, len(imgPts)):	
					imagePoints.append((imgPts[i].tolist())[0][0])
				imagePoints = np.int32(imagePoints).reshape(-1,2)
				for i in range(4):
					cv2.line(img, tuple(imagePoints[i].ravel()), tuple(imagePoints[(i+1)%4].ravel()), (0, 0, 255, 255), 1, cv2.CV_AA)
				for i in range(4):
					cv2.line(img, tuple(imagePoints[i+4].ravel()), tuple(imagePoints[4+(i+1)%4].ravel()), (0, 0, 255, 255), 1, cv2.CV_AA)
				for i in range(4):
					cv2.line(img, tuple(imagePoints[i].ravel()), tuple(imagePoints[i+4].ravel()), (0, 0, 255, 255), 1, cv2.CV_AA)
	return img

'''
code to compute the ray plane intersection.
'''
def computeRayPlaneIntersection(planePoint, planeNormal, rayDirection):
	rayDirection = np.squeeze(np.asarray(rayDirection))
	magnitude = rayDirection[0]*rayDirection[0] + rayDirection[1]*rayDirection[1] + rayDirection[2]*rayDirection[2]
	magnitude = np.sqrt(magnitude)
	rayDirection = [rayDirection[0]/magnitude, rayDirection[1]/magnitude, rayDirection[2]/magnitude]
	magnitude = rayDirection[0]*rayDirection[0] + rayDirection[1]*rayDirection[1] + rayDirection[2]*rayDirection[2]
	#print("rayDirection ", magnitude, rayDirection)
	numerator = planePoint[0]*planeNormal[0] +planePoint[1]*planeNormal[1] +planePoint[2]*planeNormal[2]
	#print("value of numerator", numerator)
	denominator = rayDirection[0]*planeNormal[0] +rayDirection[1]*planeNormal[1] +rayDirection[2]*planeNormal[2]
	#print("value of denominator", denominator)
	t = numerator/denominator
	#print("value of t", t)
	return np.array([np.array(rayDirection[0]*t), np.array(rayDirection[1]*t), np.array(rayDirection[2]*t)])

def calculatePixelValues(pixelCoordinates, cameraMatrix):
	c = np.matrix(cameraMatrix)
	d = np.transpose(np.matrix(pixelCoordinates))
	#print("shape of matrices", c.shape, d.shape)	
	final = c * d
	return final

'''
main code
'''
def findRotationAndTranslation():
	# load board and camera parameters
	#boardconfig = aruco.BoardConfiguration("chessboardinfo_small_meters.yml")

	'''
	Create the image to project from projector with two markers on it (id 5 and 6).
	'''
	dictionary = aruco.Dictionary_loadPredefined(aruco.Dictionary.ARUCO)
	markerToProject = np.full([800, 1280], 0, dtype=np.uint8)
	#print(markerToProject.shape) (800, 1280)

	for i in [400,700]:
		mi = cv2.resize(dictionary.getMarkerImage_id(5 if i == 400 else 6, 3),None,fx=5, fy=5, interpolation = cv2.INTER_NEAREST)
		white_image = np.full([mi.shape[0]+40, mi.shape[1]+40], 255, dtype=np.uint8)
		markerToProject[380:20+400+mi.shape[0],i-20:i+20+mi.shape[1]] = white_image
		#markerToProject[400:400+mi.shape[0],i:i+mi.shape[1]] = np.invert(mi, dtype=np.uint8)
		markerToProject[400:400+mi.shape[0],i:i+mi.shape[1]] = mi
		markerToProject[markerToProject == 255] = 160

	print("marker size ", mi.shape[0], mi.shape[1])
	cv2.namedWindow("marker projection", cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty("marker projection",cv2.WND_PROP_FULLSCREEN,cv2.cv.CV_WINDOW_FULLSCREEN)
	cv2.imshow("marker projection", markerToProject)

	with open("camera_calibration.yaml", 'r') as stream:
		try:
			data = yaml.load(stream)
		except yaml.YAMLError as exc:
			print(exc)
			exit()

	camparam = aruco.CameraParameters()  
	camparam.setParams(np.array(data["camera_matrix"]), 
					np.array(data["distortion_coefficients"]), 
					#np.array([ data["image_width"], data["image_height"] ]))
					np.array([ 1280, 720 ]))

	print("camera matrix", camparam.CameraMatrix)
	# create detector and set parameters
	detector = aruco.MarkerDetector()
	params = detector.getParams()

	#open first camera
	'''
	Tried to set the width and height as 1280x720 but it didn't work so using a
	recorded video rather than using the camera directly.
	'''
	os.system("v4l2-ctl --set-fmt-video=width=1280,height=720,pixelformat=1")
#	v = cv2.VideoCapture(0)
	v = cv2.VideoCapture("marker.avi")
	'''	
	if not v.isOpened():
		print "unable to open video stream"
		exit()

	# This will help in recording the frames to a video.
	ret = v.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280)
	ret = v.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 960)
	print("return value", ret)
	w = v.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH);
	h = v.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT);
	print("width and height", w, h)
	fourcc = cv2.cv.CV_FOURCC(*'XVID')
	out = cv2.VideoWriter('output.avi',fourcc, 25.0, (int(w),int(h)))
	'''
	projpoints1 = np.array([[400], [400]], dtype=np.float32)
	projpoints2 = np.array([[400 + 105], [400]], dtype=np.float32)
	projpoints3 = np.array([[400], [400 + 105]], dtype=np.float32)
	projpoints4 = np.array([[400 + 105], [400 + 105]], dtype=np.float32)
	imageSize = None

	count = 0
	while True:
		ret, img = v.read()
		imageSize = img.shape[::-1]
		print("Image size ", imageSize)
		orig = np.copy(img)

		markers = detector.detect(img)
 
		for marker in markers:
			marker.draw(img, np.array([255, 255, 255]), 2)

		# for all markers except 5 and 6
		img = drawCube(markers, camparam, img)
		img = drawCubeForProjectedMarkers(markers, camparam, img)
		#out.write(img)

		# show frame
		cv2.imshow("frame", cv2.resize(img, None, fx=0.5, fy=0.5))
		key = cv2.waitKey(30)
		if key == 1048689 or key == 1048603: # ESC or 'q'
			break
		if key == 1048608: #space
			cv2.imwrite("out.png", img)

	#out.release()

	'''
	For stereo calibration, I have assumed the distortion and intrinsic matrix 
	so that it would help in the process.
	'''
	distortionPr = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
	cameraMatrixPr = np.matrix('1000.0 0.0 1280; 0.0 1000.0 200.0; 0.0 0.0 1.0')
	cameraMatrixPr = np.array(cameraMatrixPr, dtype=np.float32)
	imageSize = (1280, 720)
	'''	
	# Look at our old code "projector_camera_calibration_aruco_old_code.py" in the same
	# directory to find out how to fill these structures (ImagePointsP, ImagePointsC, ObjectPoints)
	# required for stereo calibration.
	print("Image size ************************************ ")
	print(imageSize)
	print("Object Points ********************************* ", len(ObjectPoints))
	print(ObjectPoints)
	print("imageC points ********************************* ", len(ImagePointsC))
	print(ImagePointsC)
	print("imageP points ********************************* ", len(ImagePointsP))
	print(ImagePointsP)

	retval = 0
	cameraMatrix1 = []
	cameraMatrix2 = []
	distCoeffs1 = []
	distCoeffs2 = []
	R = []
	T = []
	E = []
	F = []

	retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(ObjectPoints, ImagePointsP, ImagePointsC, imageSize, cameraMatrix, distortion, cameraMatrixPr, distortionPr, None, None, None, None, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,30,1e-6), cv2.CALIB_USE_INTRINSIC_GUESS)
	print("----------ret----------", ret)
	print("########## Camera matrix 1")
	print(cameraMatrix1)
	print("########## Camera matrix 2")
	print(cameraMatrix2)
	print("########## Dist coeff 1")
	print(distCoeffs1)
	print("########## Dist coeff 2")
	print(distCoeffs2)
	print("########## Rotation matrix")
	print(np.matrix(R))
	print("########## Translation vector")
	print(np.matrix(T))
	print("##########")

	data = {"camera_matrix1": cameraMatrix1.tolist(), "camera_matrix2": cameraMatrix2.tolist(), 
			"distortion_coefficients1": distCoeffs1.tolist(),"distortion_coefficients2": distCoeffs2.tolist(),
			"rotation" : R.tolist(), "translation":T.tolist()}
	with open("stereo_calibration.yaml", "w") as f:
		yaml.dump(data, f)
	'''
	R = []
	T = []
	return R, T

def main():
	R, T = findRotationAndTranslation()

main();
