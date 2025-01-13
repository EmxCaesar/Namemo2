#ifndef _OBSERVER_IMG_H_
#define _OBSERVER_IMG_H_

#include <opencv2/opencv.hpp>
#include "image_utils.h"

class ImgObserver
{
public:
    virtual void process(stitching::Image* image)=0;
    virtual void setParam(void* param) { }
    virtual void clearParam(){}

    virtual ~ImgObserver(){ }
};


#endif
