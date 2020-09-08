#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>     // for SURF
#include <opencv2/features2d.hpp>              // for BRUTEFORCE MATCHER
#include <opencv2/calib3d.hpp>                 // for F and E matrix

#include <iostream>

#include "ReadImgData.h"

using namespace cv;
using namespace std;

struct CloudPoint {
    Point3d pt;
    vector<int> index_of_2d_origin;
};

bool isValidRotation(const Mat_<double> & rot);
double TriangulatePoints(
    const vector<KeyPoint>& keypoints,
    const vector<KeyPoint>& keypoints1,
    const Mat& K,
    const Mat& K_inv,
    const Matx34d& P,
    const Matx34d& P1,
    vector<Point3d>& pointcloud);
Mat_<double> LinearLSTriangulation( Point3d u, Matx34d P, Point3d u1, Matx34d P1);


int main( int argc, char** argv )
{
    // if( argc != 2)
    // {
    //  cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
    //  return -1;
    // }

    //global 3D point cloud
    vector<CloudPoint> pcloud;

    // Read and view image
    Mat image1, image2;
    image1 = imread("../data/rdimage.000.ppm", IMREAD_COLOR);   // Read the file
    image2 = imread("../data/rdimage.001.ppm", IMREAD_COLOR);   // Read the file

    if( !image1.data || !image2.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    // namedWindow( "Display window", WINDOW_NORMAL );         // Create a window for display.
    // resizeWindow( "Display window", 600,600 );
    // imshow( "Display window", image );                   // Show our image inside it.

    // int k1 = waitKey(0);                                          // Wait for a keystroke in the window


    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 10000;
    Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create( minHessian);

    vector<KeyPoint> keypoints1, keypoints2;

    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);

    // //-- Draw keypoints
    // Mat img_keypoints1, img_keypoints2;
    // drawKeypoints(image1, keypoints1, img_keypoints1);
    // drawKeypoints(image2, keypoints2, img_keypoints2);
    
    // //-- Show detected (drawn) keypoints
    // namedWindow( "SURF Keypoints 1", WINDOW_NORMAL );         // Create a window for display.
    // resizeWindow( "SURF Keypoints 1", 1000,1000 );
    // imshow("SURF Keypoints 1", img_keypoints1);

    // //-- Show detected (drawn) keypoints
    // namedWindow( "SURF Keypoints 2", WINDOW_NORMAL );         // Create a window for display.
    // resizeWindow( "SURF Keypoints 2", 1000,1000 );
    // imshow("SURF Keypoints 2", img_keypoints2);

    // int k = waitKey(0);

    // Create Descriptors
    Mat descriptors1, descriptors2;
    detector->compute( image1, keypoints1, descriptors1 );
    detector->compute( image2, keypoints2, descriptors2 );

    // Match descriptors by Bruteforce L2
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "BruteForce" );
    vector<DMatch> matches;
    matcher->match( descriptors1, descriptors2, matches );

    // Adding keypoints in the matched order to new vector of type Points2f
    vector<Point2f> imgpts1, imgpts2;
    for (auto tmp: matches){
        // cout << tmp.queryIdx << ", " << tmp.trainIdx << ", " << tmp.distance << endl;
        // cout << norm(descriptors1.row(tmp.queryIdx),descriptors2.row(tmp.trainIdx)) << endl;
        imgpts1.push_back(keypoints1[tmp.queryIdx].pt);
        imgpts2.push_back(keypoints2[tmp.trainIdx].pt);
    }

    // Display the matched keypoints on the image
    // namedWindow( "Matched old results", WINDOW_NORMAL );
    // resizeWindow( "Matched old results", 1000, 1000 );
    // Mat img_matched;
    // drawMatches( image1, keypoints1, image2, keypoints2, matches, img_matched );
    // imshow( "Matched old results", img_matched );

    // Calculate Fundamental and Essential matrix from the above matched vectors
    vector<uchar> inliers; // store the mask in a vector of uchar
    Mat F = findFundamentalMat(imgpts1, imgpts2, FM_RANSAC, 2, 0.99, 10, inliers); 
    Mat K = ( Mat_<double>(3,3) << 2780.1700000000000728, 0, 1539.25, 0, 2773.5399999999999636, 1001.2699999999999818, 0, 0, 1 ); // intrinsic camera matrix
    Mat_<double> E = K.t() * F * K;



    // Display the matched keypoints on the image
    // vector<char> inliers_c(inliers.begin(), inliers.end()); //drawMatches requires vector of char

    // namedWindow( "Matched new results", WINDOW_NORMAL );
    // resizeWindow( "Matched new results", 1000, 1000 );
    // Mat img_matched_new;
    // drawMatches( image1, keypoints1, image2, keypoints2, matches, img_matched_new, Scalar::all(-1), Scalar::all(-1), inliers_c );
    // imshow( "Matched new results", img_matched_new );

    SVD svd(E);
    Matx33d W(  0, -1, 0,
                1, 0, 0,
                0, 0, 1);
    
    Mat_<double> R = svd.u * Mat(W) * svd.vt; //Hartley Zisserman 9.19
    Mat_<double> t = svd.u.col(2); //u3
    Matx34d P1( R(0,0),R(0,1), R(0,2), t(0),
                R(1,0),R(1,1), R(1,2), t(1),
                R(2,0),R(2,1), R(2,2), t(2));

    if (!isValidRotation(R)){
        cout << "ERROR" << endl;
        return 0;
    }
    int k = waitKey();

    return 0;
}

bool isValidRotation(const Mat_<double> & rot){
    if (fabs(determinant(rot)) - 1 > 1.0e-7){
        cout << "INVALID ROTATION MATRIX" << endl;
        return false;
    }
    return true;
}

double TriangulatePoints(
    const vector<KeyPoint>& keypoints,
    const vector<KeyPoint>& keypoints1,
    const Mat& K,
    const Mat& K_inv,
    const Matx34d& P,
    const Matx34d& P1,
    vector<Point3d>& pointcloud)
{
    vector<double> reprojection_error;
    for (unsigned int i = 0; i < keypoints.size(); ++i)
    {
        Point2f kp = keypoints[i].pt;
        Point3d u(kp.x, kp.y, 1.0);
        Mat_<double> um = K_inv * Mat_<double>(u);
        u = um.at<Point3d>(0);

        Point2f kp1 = keypoints1[i].pt;
        Point3d u1(kp1.x, kp1.y, 1.0);
        Mat_<double> um1 = K_inv * Mat_<double>(u1);
        u = um1.at<Point3d>(0);

        //triangulation
        Mat_<double> X = LinearLSTriangulation(u, P, u1, P1);

        //calc reprojection error
        Mat_<double> xPt_img = K * Mat(P1)* X;
        Point2f xPt_img_(xPt_img(0)/xPt_img(2), xPt_img(1)/xPt_img(2));
        reprojection_error.push_back(norm(xPt_img_ - kp1));

        //store 3D point
        pointcloud.push_back(Point3d(X(0),X(1),X(2))); //try at() on Mat X

    }

    //return mean reprojection error
    Scalar me = mean(reprojection_error);
    cout << me[0] << ", " << me[1] << ", " << me[2] << ", " << me[3] << endl;
    return me[0];
}

Mat_<double> LinearLSTriangulation( Point3d u, Matx34d P, Point3d u1, Matx34d P1)
{
    //build A matrix
    Matx43d A(  u.x*P(2,0)-P(0,0),      u.x*P(2,1)-P(0,1),      u.x*P(2,2)-P(0,2),
                u.y*P(2,0)-P(1,0),      u.y*P(2,1)-P(1,1),      u.y*P(2,2)-P(1,2),
                u1.x*P1(2,0)-P1(0,0),   u1.x*P1(2,1)-P1(0,1),   u1.x*P1(2,2)-P1(0,2),
                u1.y*P1(2,0)-P1(1,0),   u1.y*P1(2,1)-P1(1,1),   u1.y*P1(2,2)-P1(1,2)
                );
 
    //build B vector
    Matx41d B(  -(u.x*P(2,3)-P(0,3)),
                -(u.y*P(2,3)-P(1,3)),
                -(u1.x*P1(2,3)-P1(0,3)),
                -(u1.y*P1(2,3)-P1(1,3)));

    //solve for X
    Mat_<double> X;
    solve(A,B,X,DECOMP_SVD);
    return X;
}
