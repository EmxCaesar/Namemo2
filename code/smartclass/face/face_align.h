#ifndef _FACE_ALIGN_H_
#define _FACE_ALIGN_H_

#include <opencv2/opencv.hpp>
#include"../common/image_utils.h"

void face_align(cv::Mat& src, cv::Mat& aligned, FaceDetection& face_detection);
void face_env_align(cv::Mat& src, cv::Mat& aligned, float bbox[4],
                    float offset_x, float offset_y, float ratio);

#endif
