#include "Dataset.h"

Dataset::Dataset(const string& path):_path(path){
    ReadImgs();
}

void Dataset::ReadImgs(){
    glob(_path + "/*.ppm", _filenames, false);
    size_t count = _filenames.size();

    for (size_t i =0; i<count; ++i)
        _images.push_back(imread(_filenames[i]));
}