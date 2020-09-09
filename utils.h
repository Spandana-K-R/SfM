#pragma once

#include <vector>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

void GetAlignedPointsFromMatch(const vector<KeyPoint>& imgpts1, const vector<KeyPoint>& imgpts2, const vector<DMatch>& matches, vector<KeyPoint>& pt_set1, vector<KeyPoint>& pt_set2){
	for (unsigned int i=0; i<matches.size(); i++) {
		assert(matches[i].queryIdx < imgpts1.size());
		pt_set1.push_back(imgpts1[matches[i].queryIdx]);
		assert(matches[i].trainIdx < imgpts2.size());
		pt_set2.push_back(imgpts2[matches[i].trainIdx]);
	}	
}

void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps) {
	ps.clear();
	for (unsigned int i=0; i<kps.size(); i++) ps.push_back(kps[i].pt);
}

void PointsToKeyPoints(const vector<Point2f>& ps, vector<KeyPoint>& kps) {
	kps.clear();
	for (unsigned int i=0; i<ps.size(); i++) kps.push_back(KeyPoint(ps[i],1.0f));
}