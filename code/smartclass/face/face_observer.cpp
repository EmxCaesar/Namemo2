#include "face_observer.h"
#include "../common/data_path.h"
#include "face_database.h"

FaceObserver::FaceObserver(bool setDetROI)
{
    m_pFaceDetector = new FaceDetector(RETINA_ENGINE_PATH);
    m_pFaceRecognizer = new FaceRecognizer(ARCFACE_ENGINE_PATH);

    if(setDetROI){
        m_pFaceDetector->setRelatedROI(0,0,3);
    }
}

FaceObserver::~FaceObserver()
{
    delete m_pFaceDetector;
    delete m_pFaceRecognizer;
}

void FaceObserver::setParam(void* param)
{
    FaceDB* stParam = (FaceDB*)param;
    m_pFaceRecognizer->setDataBase(stParam);
}

void FaceObserver::process(stitching::Image *stImage)
{
   // std::cout << "FaceObserver process "<<std::endl;
    m_mutex.lock();
    m_pFaceDetector->process(*stImage);
    m_pFaceRecognizer->process(*stImage);
    m_mutex.unlock();
}



/*****private func****/

float FaceObserver::computeIOU(float bbox_a[4], float bbox_b[4])
{
  //x1:左上角x y1:左上角y
  //x2:右下角x y2:右下角y

  float a_x1 = bbox_a[0];
  float a_y1 = bbox_a[1];
  float a_x2 = bbox_a[2];
  float a_y2 = bbox_a[3];

  float b_x1 = bbox_b[0];
  float b_y1 = bbox_b[1];
  float b_x2 = bbox_b[2];
  float b_y2 = bbox_b[3];

  float inter_x1 = std::max(a_x1, b_x1);
  float inter_y1 = std::max(a_y1, b_y1);
  float inter_x2 = std::min(a_x2, b_x2);
  float inter_y2 = std::min(a_y2, b_y2);

  if ((inter_x1 >= inter_x2) || (inter_y1 >= inter_y2))
    return 0.0f;

  float area_a = (a_x2 - a_x1) * (a_y2 - a_y1);
  float area_b = (b_x2 - b_x1) * (b_y2 - b_y1);
  float area_inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);

  return area_inter / (area_a + area_b - area_inter);
}

