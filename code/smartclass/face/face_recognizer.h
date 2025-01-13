#ifndef _FACE_RECOG_H_
#define _FACE_RECOG_H_

#include <opencv2/opencv.hpp>
#include "../common/image_utils.h"
#include "facenn/arcface.h"
#include "face_database.h"

class FaceRecognizer
{
private:
    ArcFace* pFaceRecognizer;
    float* data;
    float* prob;   
    FaceDB faceDB;
    //float* database_buf;

    //int m_person_num;
    //int m_pose_num;

public:
    FaceRecognizer(const char* engine_path);
    ~FaceRecognizer();

    void setDataBase(FaceDB* db);
    void clearDataBase();
    void process(stitching::Image& image);
};


#endif
