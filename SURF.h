#pragma once

#include <opencv2/xfeatures2d/nonfree.hpp>     // for SURF

using namespace std;
using namespace cv;

class SURF{
public:

    int minHessian = 1000;
    Ptr<xfeatures2d::SURF> _featuredetector = xfeatures2d::SURF::create( minHessian);
	
	
	vector<Mat>& _imgs; 
	vector<vector<KeyPoint>>& _imgpts;
	vector<Mat>& _descriptors;

    SURF(vector<Mat>& imgs, vector<vector<KeyPoint>>& imgpts, vector<Mat>& descriptors): _imgs(imgs), _imgpts(imgpts), _descriptors(descriptors)
    {
        _featuredetector->detect(_imgs, _imgpts);
        _featuredetector->compute(_imgs, _imgpts, _descriptors);
    }
};