#ifndef _FACE_DECTECTOR_H_
#define _FACE_DECTECTOR_H_

#include "../common/image_utils.h"
#include "facenn/retinaface.h"

class FaceDetector
{
private:
    RetinaFace* pFaceDetector;
	float* data;
	float* prob;

    bool m_isRelatedROISet = false;
    float m_ROIAnchorOffsetX = 0;
    float m_ROIAnchorOffsetY = 0;
    float m_ROIRatio = 2;

    void nms(std::vector<FaceDetection>& res, float* prob);

public:
    FaceDetector(const char *engine_path);
	~FaceDetector();

    void setRelatedROI(float roi_anchor_offset_x,
                       float roi_anchor_offset_y,
                       float roi_ratio);
    void process(stitching::Image& image);
};

#endif
