#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>     // for SURF
#include <opencv2/features2d.hpp>              // for BRUTEFORCE MATCHER
#include <opencv2/calib3d.hpp>                 // for F and E matrix


#include "ReadImgData.h"
#include "FeatureMatcherSURF.h"

using namespace std;
using namespace cv;


class FrameManager{
public:
    Ptr<Dataset> _dataset;
    vector<vector<KeyPoint>> all_keypoints;
    SURF feature_matcher;

    FrameManager(Ptr<Dataset> dataset): _dataset(dataset){}
    
    vector<DMatch> flipMatches(const vector<DMatch>& matches) {
        vector<DMatch> flip;
        for(int i=0; i<matches.size(); ++i) {
            //TODO
            flip.push_back(matches[i]);
            swap(flip.back().queryIdx,flip.back().trainIdx);
        }
	    return flip;
    }

    void matchFeatures(){
        int loop1_top = _dataset->_images.size() - 1;
        int loop2_top = _dataset->_images.size();
	    int frame_num_i = 0;

        vector<KeyPoint> keypoints1, keypoints2;

        //TODO parallelize
		for (frame_num_i = 0; frame_num_i < loop1_top; frame_num_i++) {
			for (int frame_num_j = frame_num_i + 1; frame_num_j < loop2_top; frame_num_j++){
				
                keypoints1.clear(); keypoints2.clear();

                // cout << "------------ Match " << imgs_names[frame_num_i] << ","<<imgs_names[frame_num_j]<<" ------------\n";
				vector<DMatch> matches_tmp;
                
                //-- Step 1: Detect the keypoints using SURF Detector

                detector->detect(image1, keypoints1);
                detector->detect(image2, keypoints2);


				feature_matcher->MatchFeatures(frame_num_i,frame_num_j,&matches_tmp);
				matches_matrix[std::make_pair(frame_num_i,frame_num_j)] = matches_tmp;

				std::vector<cv::DMatch> matches_tmp_flip = FlipMatches(matches_tmp);
				matches_matrix[std::make_pair(frame_num_j,frame_num_i)] = matches_tmp_flip;
			}
		}
    }

	//}

	features_matched = true;
}

}

std::vector<cv::DMatch> matches_tmp;

std::map<std::pair<int,int> ,std::vector<cv::DMatch> > matches_matrix;

matches_matrix[std::make_pair(frame_num_i,frame_num_j)] = matches_tmp;

std::vector<cv::DMatch> matches_tmp_flip = FlipMatches(matches_tmp);
matches_matrix[std::make_pair(frame_num_j,frame_num_i)] = matches_tmp_flip;