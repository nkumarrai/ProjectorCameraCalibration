import sys
import cv2
import numpy as np
import aruco
import yaml
from numpy.linalg import inv

def computeRayPlaneIntersection(planePoint, planeNormal, rayDirection):
	rayDirection = np.squeeze(np.asarray(rayDirection))
	magnitude = rayDirection[0]*rayDirection[0] + rayDirection[1]*rayDirection[1] + rayDirection[2]*rayDirection[2]
	magnitude = np.sqrt(magnitude)
	rayDirection = [rayDirection[0]/magnitude, rayDirection[1]/magnitude, rayDirection[2]/magnitude]
	magnitude = rayDirection[0]*rayDirection[0] + rayDirection[1]*rayDirection[1] + rayDirection[2]*rayDirection[2]
	print("rayDirection ", magnitude, rayDirection)
	numerator = planePoint[0]*planeNormal[0] +planePoint[1]*planeNormal[1] +planePoint[2]*planeNormal[2]
	print("value of numerator", numerator)
	denominator = rayDirection[0]*planeNormal[0] +rayDirection[1]*planeNormal[1] +rayDirection[2]*planeNormal[2]
	print("value of denominator", denominator)
	t = numerator/denominator
	print("value of t", t)
	return np.array([np.array(rayDirection[0]*t), np.array(rayDirection[1]*t), np.array(rayDirection[2]*t)])

def calculatePixelValues(pixelCoordinates, cameraMatrix):
	c = np.matrix(cameraMatrix)
	d = np.transpose(np.matrix(pixelCoordinates))
	#print("shape of matrices", c.shape, d.shape)	
	final = c * d
	return final

def findRotationAndTranslation():
	# load board and camera parameters
	#boardconfig = aruco.BoardConfiguration("chessboardinfo_small_meters.yml")

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

	# create detector and set parameters
	detector = aruco.MarkerDetector()
	params = detector.getParams()

	#open first camera
	#v = cv2.VideoCapture("stereo.avi")
	v = cv2.VideoCapture(0)
	if not v.isOpened():
		print "unable to open video stream"
		exit()


	'''
	w = v.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH);
	h = v.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT); 
	fourcc = cv2.cv.CV_FOURCC(*'XVID')
	out = cv2.VideoWriter('output.avi',fourcc, 25.0, (int(w),int(h)))
	'''
	ImagePointsP = []
	ImagePointsC = []
	ObjectPoints = []
	projpoints1 = np.array([[400], [400]], dtype=np.float32)
	projpoints2 = np.array([[400 + 105], [400]], dtype=np.float32)
	projpoints3 = np.array([[400], [400 + 105]], dtype=np.float32)
	projpoints4 = np.array([[400 + 105], [400 + 105]], dtype=np.float32)
	imageSize = None

	count = 0
	while True:
		ret, img = v.read()
		imageSize = img.shape[::-1]
		orig = np.copy(img)

		markers = detector.detect(img)
		for marker in markers:
			# print marker ID and point positions
			#print("Marker: {:d}".format(marker.id))
			#for i, point in enumerate(marker):
			#  print("\t{:d} {}".format(i, str(point)))
			marker.draw(img, np.array([255, 255, 255]), 2)

		 	# calculate marker extrinsics for marker size of 30mm
			marker.calculateExtrinsics(0.030, camparam)
			#print("Marker extrinsics:{:d} iteration {:d} \n{:s}\n{:s}".format(marker.id, count, marker.Tvec, marker.Rvec))
			#print("detected ids: {}".format(", ".join(str(m.id) for m in markers)))

		# Try all the corners of marker5 
		marker1 = None
		marker5 = None
		for marker in markers:
			if marker.id == 1:
				marker1 = marker
			if marker.id == 5:  #marker id is changed.. for sanity check. revert it to 5 later
			 	marker5 = marker
		marker5corners = []
		objPoints = []
		if marker1 != None and marker5 != None:
			for i, point in enumerate(marker5):
				marker5corners.append(point)
			print(marker5corners)
			count = count + 1
			for i in range(len(marker5corners)):
				#pixelValue = marker5.getCenter()
				pixelValue = marker5corners[i]
				cameraMatrix = camparam.CameraMatrix
				distortion = camparam.Distorsion
				test = np.zeros((1,1,2), dtype=np.float32)
				test[0][0][0] = pixelValue[0]
				test[0][0][1] = pixelValue[1]
				print("***************** start *****************")
				print("before correction Pixel values ")
				print(test)
				pixelCoordinates = cv2.undistortPoints(test, cameraMatrix, distortion)
				print("corrected Pixel values ")
				print(pixelCoordinates)
				pixelCoordinates = np.array([pixelCoordinates[0][0][0], pixelCoordinates[0][0][1], 1])
				inversePixelValues = calculatePixelValues(pixelCoordinates, cameraMatrix)
				print("Inverse pixel values... calculate")
				print(inversePixelValues)
				marker1.calculateExtrinsics(0.030, camparam)
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
				''' (not using it anymore)
				print("marker translation vector ", tvec, type(tvec))
				rotationMatrix, jacb = cv2.Rodrigues(rvec)
				print("rotation matrix")
				print(rotationMatrix)
				print("normal vector c")
				print(rotationMatrix[:, 2])
				normalVector1 = rotationMatrix[:,2]
				print("normal vector r")
				print(rotationMatrix[2])
				normalVector2 = rotationMatrix[2]
				#####################################################
				rvecForRepr = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
				tvecForRepr = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
				#####################################################
				computed3Dpoints = computeRayPlaneIntersection(tvec, normalVector1, pixelCoordinates)
				print("computed 3d points c")
				print(computed3Dpoints)
				projectedImagePoints, jacb = cv2.projectPoints(np.array([computed3Dpoints]), rvecForRepr, tvecForRepr, cameraMatrix, distortion)
				print("projected 3d points on image plane c")
				print(projectedImagePoints)
				computed3Dpoints = computeRayPlaneIntersection(tvec, normalVector2, pixelCoordinates)
				print("computed 3d points r")
				print(computed3Dpoints)
				projectedImagePoints, jacb = cv2.projectPoints(np.array([computed3Dpoints]), rvecForRepr, tvecForRepr, cameraMatrix, distortion)
				print("projected 3d points on image plane r")
				print(projectedImagePoints)
				print("***************** end *****************")
				'''
				objPoints.append(computed3Dpoints)
			ImagePointsP.append(np.array([projpoints1, projpoints2, projpoints3, projpoints4]))
			ImagePointsC.append(np.array(marker5corners))
			ObjectPoints.append(np.array(objPoints))
			if count == 50:
				break
	

		#out.write(img)

		# show frame
		cv2.imshow("frame", cv2.resize(img, None, fx=0.5, fy=0.5))
		key = cv2.waitKey(30)
		if key == 1048689 or key == 1048603: # ESC or 'q'
			break
		if key == 1048608: #space
			cv2.imwrite("out.png", img)

	#out.release()

	distortionPr = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
	cameraMatrixPr = np.matrix('1000.0 0.0 1280; 0.0 1000.0 100.0; 0.0 0.0 1.0')
	cameraMatrixPr = np.array(cameraMatrixPr, dtype=np.float32)
	imageSize = (1280, 720)
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
	return R, T

def main():
	R, T = findRotationAndTranslation()

main();
