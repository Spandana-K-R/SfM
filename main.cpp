#include <iostream>

#include "ReadImgData.h"
#include "PnP.h"

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
    if(dataset._images.size() <= dataset._filenames.size()) { 
		cerr << "Can't read all files" << endl;
		return 1;
	}

    //
    FrameManager fm(&dataset);

    return 0;
}