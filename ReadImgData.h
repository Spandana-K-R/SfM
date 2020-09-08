#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>         // for Mat
#include <opencv2/core/utility.hpp> // for glob
#include <opencv2/highgui.hpp>      // for imread

using namespace std;
using namespace cv;

class Dataset{
public:

    string _path;
    vector<string> _filenames;
    vector<Mat> _images;

    Dataset(const string& path);
    
    void ReadImgs();
};