#include <opencv2/opencv.hpp>
#include <iostream>
#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;
namespace fs = boost::filesystem;

Mat img;
deque<Point> projRectInImage;
Mat homographyCamToProj;
Mat mousePoint = (Mat_<double>(3, 1) << 0,0,1);

static void onMouse( int event, int x, int y, int flags, void* ) {
    if( event == EVENT_LBUTTONDOWN ) {
        if (projRectInImage.size() >= 4) {
            projRectInImage.clear();
        }
        projRectInImage.push_back(Point(x, y));
    }
    if( event == EVENT_RBUTTONDOWN ) {
        mousePoint.at<double>(0) = x;
        mousePoint.at<double>(1) = y;
    }
}

void calibrate(Size projectorScreenSize, Size camScreenSize) {
    if (projRectInImage.size() != 4) {
        cout << "need 4 points" << endl;
        return;
    }

    vector<Point2f> camPoints;
    for (Point p : projRectInImage) {
        camPoints.push_back(Point2f(p.x, p.y));
    }
    vector<Point2f> projPoints(4);
    projPoints[0] = Point2f(0,                         0);
    projPoints[1] = Point2f(projectorScreenSize.width, 0);
    projPoints[2] = Point2f(projectorScreenSize.width, projectorScreenSize.height);
    projPoints[3] = Point2f(0,                         projectorScreenSize.height);

    cout << "cam points: " << endl << Mat(camPoints) << endl;
    cout << "proj points: " << endl << Mat(projPoints) << endl;

    homographyCamToProj = findHomography(camPoints, projPoints);

    cout << "found homography " << endl << homographyCamToProj << endl;

    FileStorage fs("pro_cam_calibration.yaml", FileStorage::WRITE);
    fs << "homographyCamToProj" << homographyCamToProj;
    fs << "proj_width" << projectorScreenSize.width;
    fs << "proj_height" << projectorScreenSize.height;
    fs << "camPoints" << Mat(camPoints);
    fs << "camSize" << camScreenSize;
}

int main(int argc, char const *argv[]) {
    cv::CommandLineParser parser(argc, argv, "{help h||}"
                                             "{@source||}"
                                             "{@screenw|1280|width of the projector screen}"
                                             "{@screenh|800 |height of the projector screen}");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (not parser.check()) {
        parser.printErrors();
        return 1;
    }

    const string source = parser.get<string>("@source");

    cout << "Device opening ..." << endl;
    VideoCapture capture;
    if( fs::is_regular_file(source)) {
        capture.open(source);
    } else {
        capture.open( CAP_OPENNI2 );
        if (!capture.isOpened()) {
            capture.open( CAP_OPENNI );
        }
    }

    if (not capture.isOpened()) {
        cerr << "Can't open OpenNI cap" << endl;
        return 1;
    }

    namedWindow("color", cv::WINDOW_GUI_NORMAL + cv::WINDOW_AUTOSIZE);
    setMouseCallback("color", onMouse, 0);

    for (;;) {
        if (not capture.grab()) {
            cout << "Can not grab images." << endl;
            return -1;
        } else {
            if (capture.retrieve(img, CAP_OPENNI_BGR_IMAGE)) {
                if (projRectInImage.size() > 1) {
                    //draw projector extents on cam image
                    for (size_t i = 0; i < projRectInImage.size(); i++) {
                        line(img, projRectInImage[i], projRectInImage[(i+1) %  projRectInImage.size()], Scalar::all(255));
                        stringstream ss; ss << projRectInImage[i];
                        putText(img, ss.str(), projRectInImage[i] - Point(10,10), cv::FONT_HERSHEY_PLAIN, 1.0, Scalar::all(255));
                    }
                } else {
                    circle(img, projRectInImage[0], 10, Scalar::all(255), CV_FILLED);
                }

                //draw mouse right-click point
                const Point mp = Point(mousePoint.at<double>(0), mousePoint.at<double>(1));
                circle(img, mp, 5, Scalar(255,0,0), CV_FILLED);
                if (not homographyCamToProj.empty()) { //translate to projector coordinates
                    Mat projPoint = homographyCamToProj * mousePoint;
                    convertPointsFromHomogeneous(projPoint.t(), projPoint);
                    projPoint.convertTo(projPoint, CV_16SC1);
                    stringstream ss; ss << mousePoint << " -> " << projPoint;
                    putText(img, ss.str(), mp - Point(50, 10), cv::FONT_HERSHEY_PLAIN, 1.0, Scalar(255,0,0));
                }

                imshow( "color", img );
            }
        }

        const int key = waitKey(30);
        if (key == 'q' or key == 27) {
            break;
        }
        if (key == 'c') {
            calibrate(Size(parser.get<int>("@screenw"), parser.get<int>("@screenh")), img.size());
        }
    }

    return 0;
}
