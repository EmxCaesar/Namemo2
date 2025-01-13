#ifndef _IMAGE_UTILS_H_
#define _IMAGE_UTILS_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct FaceDetection
{
    float bbox[4];  //x1 y1 x2 y2
    float class_confidence;
    float landmark[10];
};// floatx15

struct FaceInfo
{
    // det result
    FaceDetection face_detection;// detect result :box landmark face_score
    cv::Mat face_aligned;// 112x112
    cv::Mat face_env_aligned;//112x112

    // recog result
    cv::Mat face_descriptor;// 512
    cv::Mat face_env_descriptor;// 512
    int id;// recog label	
    float score;// recog similarity
};

struct FaceBox
{
    float x1,y1,x2,y2;
    int id;
    float score;
};

struct WarpFaceBox
{
    FaceBox aft_warp;
    int img_id;
    float nonzero_num;
};

namespace stitching{

class Image
{
public:
	int id;
    int round_num;

    // face
    bool b_use_env = false;
    std::vector<FaceInfo> vec_faceinfo;// face detect and recog result
    std::vector<FaceBox> vec_facebox;// copy from faceinfo in Stitcher warp
    std::vector<FaceBox> vec_warpfacebox;// warp result

	//feature
    cv::detail::ImageFeatures feature;
    std::vector<cv::detail::MatchesInfo> pairwise_match;

	//img
    cv::Mat img_original;
    cv::Mat img_seam;
    cv::Mat img_warp;
    cv::Mat img_warped_f;

	//mask
    cv::Mat mask;
    cv::Mat mask_warp;
    cv::Mat mask_seam;

	//camera params
    cv::detail::CameraParams camera;

    cv::Point corner;
    cv::Size size;

    bool is_filter=false;
    bool have_face=false;

public:
	Image(){}
    Image(cv::Mat& img, int idx, int round_number){
       img_original = img.clone();
       id = idx;
       round_num = round_number;
	}
};//class

}//namespace



#endif
