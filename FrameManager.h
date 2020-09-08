#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>     // for SURF
#include <opencv2/features2d.hpp>              // for BRUTEFORCE MATCHER
#include <opencv2/calib3d.hpp>                 // for F and E matrix


#include "Dataset.h"
#include "SURF.h"

using namespace std;
using namespace cv;


class FrameManager{
public:
    vector<Mat>& _imgs;
    vector<vector<KeyPoint>>& _imgpts;
    vector<Mat>& _descriptors;

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "BruteForce" );
    map<pair<int,int> , vector<DMatch> > matches_matrix;

    bool features_matched_already;

    FrameManager(vector<Mat>& imgs, vector<vector<KeyPoint>>& imgpts, vector<Mat>& descriptors): _imgs(imgs), _imgpts(imgpts), _descriptors(descriptors){
        features_matched_already = false;
    }
    
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
        int loop1_top = _imgs.size() - 1;
        int loop2_top = _imgs.size();
	    int frame_num_i = 0;

        //TODO parallelize
		for (frame_num_i = 0; frame_num_i < loop1_top; frame_num_i++) {
			for (int frame_num_j = frame_num_i + 1; frame_num_j < loop2_top; frame_num_j++){                
                //-- Step 1: Let's match features
				vector<DMatch> matches_tmp;
				matcher->match(_descriptors[frame_num_i], _descriptors[frame_num_j], matches_tmp);
				matches_matrix[std::make_pair(frame_num_i,frame_num_j)] = matches_tmp;

				vector<DMatch> matches_tmp_flip = flipMatches(matches_tmp);
				matches_matrix[make_pair(frame_num_j,frame_num_i)] = matches_tmp_flip;

                // cout << "frame: " << frame_num_i << " frame: " << frame_num_j << ", matches tmp size: " << matches_tmp.size() << endl;
			}
		}
	    features_matched_already = true;
    }


};