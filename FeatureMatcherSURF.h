#pragma once

#include <opencv2/xfeatures2d/nonfree.hpp>     // for SURF

using namespace std;
using namespace cv;

class SURF{
public:

    int minHessian = 1000;
    Ptr<xfeatures2d::SURF> _featuredetector = xfeatures2d::SURF::create( minHessian);
    
    Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;
	
	vector<Mat> descriptors;
	
	vector<Mat>& imgs; 
	vector<vector<KeyPoint> >& imgpts;

    SURF();
}