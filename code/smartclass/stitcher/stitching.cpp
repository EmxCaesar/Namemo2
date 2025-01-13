#include "stitching.h"

namespace stitching {

int stitching(std::vector<Image>& vec_stImage,
              Stitcher& stitcher, cv::Mat& pano, std::vector<FaceBox> &stitch_box,
              int col,int person_num)
{
    int ret = 0;
    //std::cout << "stitching col : " << col << std::endl;
    stitcher.images_feed(&vec_stImage);

    stitcher.Images_match(0,col);

    std::vector<int> reserved_idx;
    stitcher.images_filter(reserved_idx);

    std::vector<cv::detail::CameraParams> cameras(reserved_idx.size());
    stitcher.images_estimate_cameras(reserved_idx, col, cameras);

    cv::Point top_left;
    stitcher.images_warp(cameras, top_left);

    stitcher.images_exposure_compensate();

    stitcher.images_find_seam();

    std::vector<std::vector<WarpFaceBox>> reserved_face_box(person_num);
    stitcher.images_filter_facebox(reserved_face_box);

    stitcher.images_optim_seam(reserved_face_box);

    stitcher.images_blend(top_left,pano, stitch_box);

    return ret;
}

}
