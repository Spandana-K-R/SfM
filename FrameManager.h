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
#include "utils.h"

using namespace std;
using namespace cv;


class FrameManager{
public:
  vector<Mat>& _imgs;
  vector<vector<KeyPoint>>& _imgpts;
  vector<vector<KeyPoint>>& _imgpts_pruned;
  vector<Mat>& _descriptors;

  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "BruteForce" );
  map<pair<int,int> , vector<DMatch> > matches_matrix;

  bool features_matched_already;

  FrameManager(vector<Mat>& imgs, vector<vector<KeyPoint>>& imgpts, vector<vector<KeyPoint>>& imgpts_pruned, vector<Mat>& descriptors): _imgs(imgs), _imgpts(imgpts), _imgpts_pruned(imgpts_pruned), _descriptors(descriptors){
    features_matched_already = false;
    matchFeatures();
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

  void matchFeatures()
  {
    // TODO parallelize
    if (!features_matched_already)
    {
      for (int frame_num_i = 0; frame_num_i < _imgs.size() - 1; frame_num_i++) {
        for (int frame_num_j = frame_num_i + 1; frame_num_j < _imgs.size(); frame_num_j++){                
          // Let's match features
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
  }

  void GetBaseline3D(){};

  void PruneMatchesBasedOnF(){
    //prune the match between <_i> and all views using the Fundamental matrix to prune
    for (int _i=0; _i < _imgs.size() - 1; ++_i){
		  for (unsigned int _j=_i+1; _j < _imgs.size(); ++_j){
			  int older_view = _i, working_view = _j;

        GetFundamentalMat(_imgpts[older_view], _imgpts[working_view], _imgpts_pruned[older_view], _imgpts_pruned[working_view], matches_matrix[std::make_pair(older_view,working_view)]);
			//update flip matches as well
			  matches_matrix[std::make_pair(working_view,older_view)] = flipMatches(matches_matrix[std::make_pair(older_view,working_view)]);
		  }
	  }
  }

  Mat GetFundamentalMat(const vector<KeyPoint>& imgpts1, const vector<KeyPoint>& imgpts2, vector<KeyPoint>& imgpts1_pruned, vector<KeyPoint>& imgpts2_pruned, vector<DMatch>& matches){
    //Try to eliminate keypoints based on the fundamental matrix
    vector<uchar> status(imgpts1.size());
  	imgpts1_pruned.clear(); imgpts2_pruned.clear();
	
    vector<KeyPoint> imgpts1_tmp;
    vector<KeyPoint> imgpts2_tmp;
    if (matches.size() <= 0) { 
      //points already aligned...
      imgpts1_tmp = imgpts1;
      imgpts2_tmp = imgpts2;
    } else {
      GetAlignedPointsFromMatch(imgpts1, imgpts2, matches, imgpts1_tmp, imgpts2_tmp);
    }
	
	  Mat F;
	  {
      vector<Point2f> pts1,pts2;
      KeyPointsToPoints(imgpts1_tmp, pts1);
      KeyPointsToPoints(imgpts2_tmp, pts2);
      double minVal,maxVal;
      minMaxIdx(pts1,&minVal,&maxVal);
      F = findFundamentalMat(pts1, pts2, FM_RANSAC, 0.006 * maxVal, 0.99, status); //threshold from [Snavely07 4.1]
	  }
	
    vector<DMatch> new_matches;
    cout << "F keeping " << countNonZero(status) << " / " << status.size() << endl;	
    for (unsigned int i=0; i<status.size(); i++) {
      if (status[i]) 
      {
        imgpts1_pruned.push_back(imgpts1_tmp[i]);
        imgpts2_pruned.push_back(imgpts2_tmp[i]);

        if (matches.size() <= 0) { //points already aligned...
          new_matches.push_back(DMatch(matches[i].queryIdx,matches[i].trainIdx,matches[i].distance));
        } else {
          new_matches.push_back(matches[i]);
        }
      }
    }	
	
    cout << matches.size() << " matches before, " << new_matches.size() << " new matches after Fundamental Matrix\n";
    matches = new_matches; //keep only those points who survived the fundamental matrix
	  return F;
  }
};