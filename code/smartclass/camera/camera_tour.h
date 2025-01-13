#ifndef _CAMERA_THREAD_H_
#define _CAMERA_THREAD_H_

#include "camera_base.h"
#include "../common/image_observer.h"
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

class CameraTour
{
private:
    CameraBase* m_pCamera;
    std::vector<stitching::Image>* m_pVecStImage;
    std::vector<std::pair<float,float>> m_tourTable;
    std::vector<ImgObserver*>m_vecObserver;

    float m_pan_start;
    float m_pan_step;
    int m_pan_cols;
    float m_tilt_start;
    float m_tilt_step;
    int m_tilt_rows;

    void makeTourTable();

public:
    CameraTour(CameraBase* pCamera,
                std::vector<stitching::Image>* pVecStImage,
                float pan_start, float pan_step, int pan_cols,
                float tilt_start, float tilt_step, int tilt_rows);
    void run(int round_number);
    void attachObserver(ImgObserver* observer);
    void detachObserver();
};


#endif
