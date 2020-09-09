#include <iostream>

#include "Dataset.h"
#include "FrameManager.h"
#include "SURF.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv){
  if( argc != 2)
    {
     cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

  // Import all images
  Dataset dataset(argv[1]);
  if(dataset._images.size() == 0) { 
		cerr << "Can't get image files" << endl;
		return 1;
	}
  if(dataset._images.size() < dataset._filenames.size()) { 
		cerr << "Can't read all files! Read " << dataset._images.size() << " files but should've read " << dataset._filenames.size() << " files." << endl;
		return 1;
	}

  // keypoint detection and extracting decriptors within class SURF
  vector<vector<KeyPoint>> imgpts;
  vector<vector<KeyPoint>> imgpts_pruned;
  vector<Mat> descriptors;
  
  SURF surf(dataset._images, imgpts, descriptors);  
  // for (int i = 0; i < dataset._images.size(); ++i){
  //   cout << "size of each keypoint vector " << imgpts[i].size() << endl;
  // }

  // for (int i = 0; i < dataset._images.size(); ++i){
  //   cout << "size of descriptor matrix "<< descriptors[i].size << endl;
  // }

  //
  FrameManager fm(dataset._images, imgpts, imgpts_pruned, descriptors);
  if (!fm.features_matched_already)
    fm.matchFeatures();

  fm.GetBaseline3D();

  return 0;
}