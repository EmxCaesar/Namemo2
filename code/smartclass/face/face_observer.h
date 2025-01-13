#ifndef _FACE_OBSERVER_H_
#define _FACE_OBSERVER_H_

#include <mutex>
#include <memory>
#include "../common/image_utils.h"
#include "../common/image_observer.h"
#include "face_detector.h"
#include "face_recognizer.h"

class FaceObserver:public ImgObserver
{
private:
    std::mutex m_mutex;
    FaceDetector* m_pFaceDetector;
    FaceRecognizer* m_pFaceRecognizer;

    float computeIOU(float bbox_a[4], float bbox_b[4]);

public:
    FaceObserver(bool setDetROI = false);
    ~FaceObserver();
    void setParam(void* param) override;
    void clearParam() override { m_pFaceRecognizer->clearDataBase();}
    void process(stitching::Image* stImage) override;
};

#endif
