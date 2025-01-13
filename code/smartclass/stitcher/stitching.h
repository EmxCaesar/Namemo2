#ifndef _STITCHING_H_
#define _STITCHING_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include "../common/image_utils.h"
#include "stitcher.h"

namespace stitching {

int stitching(std::vector<Image>& vec_stImage,
              Stitcher& stitcher, cv::Mat& pano, std::vector<FaceBox>& stitch_box,
              int col, int person_num);

}


#endif
