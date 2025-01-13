#ifndef _CAMERA_BASE_H_
#define _CAMERA_BASE_H_

#include <opencv2/opencv.hpp>

class CameraBase
{
public:
    virtual ~CameraBase(){

    }

    virtual void capture(cv::Mat& img, int index) = 0;
    virtual void PTZMove(const float pantiltX,const float pantiltY, const float zoom)=0;

};

#endif
