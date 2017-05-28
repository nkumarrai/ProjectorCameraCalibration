import numpy as np
import cv2
import glob
import yaml

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((4 * 9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:4].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
images = []

#open first camera
#v = cv2.VideoCapture(0)
#Read from recorded video - 1280x720
v = cv2.VideoCapture("chessboard.avi")

while True:
  ret, img = v.read()
  orig = np.copy(img)

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  print(gray.shape[::-1])
  # Find the chess board corners
  corners = 0
  ret, corners = cv2.findChessboardCorners(gray, (9, 4), corners,
                         cv2.cv.CV_CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)

  #If found, add object points, image points (after refining them)
  if ret == True:
    cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (9, 4), corners, ret)
    objpoints.append(objp)
    images.append(orig)
    imgpoints.append(corners)
    print "Save image {0}".format(len(images))
  
  cv2.imshow('img',img)
  key = cv2.waitKey(30)
  print("length of images", len(images))

  if key == 27: # ESC or 'q'
    break

cv2.destroyAllWindows()

if len(images) == 0:
  exit()

print("size ", gray.shape[::-1])
print("***********************************")
print("Object points ", objpoints)
print("***********************************")
print("Image points ", imgpoints)
print("***********************************")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print("----------ret----------")
print(mtx)
print("##########")
print(dist)
print("##########")
print(rvecs)
print("##########")
print(tvecs)
print("----------ret----------")

data = {"camera_matrix": mtx.tolist(), 
        "distortion_coefficients": dist.tolist(),
        "image_size": np.shape(images[0])}
with open("camera_calibration.yaml", "w") as f:
  yaml.dump(data, f)

img = images[0]
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
 
# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imshow('calibration result',dst)
cv2.waitKey(0)

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
 
# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imshow('calibration result',dst)
cv2.waitKey(0)

mean_error = 0
tot_error = 0
for i in xrange(len(objpoints)):
	imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
	error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
	tot_error += error
print("total error: ", tot_error/len(objpoints))
