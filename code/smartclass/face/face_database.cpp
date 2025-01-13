#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "face_database.h"
#include "../common/image_utils.h"
#include "facenn/retinaface.h"
#include "facenn/arcface.h"
#include "face_align.h"

#define SAVE_PIC 1
#define TXT 0
#define LOG_DETAIL 1


void face_database_make(const char* pics_folder_dir,
                        int person_num, int  pose_num, const char* pic_type)
{
    std::string path_prefix(pics_folder_dir);
    std::string pic_postfix(pic_type);

    // facenn init
    RetinaFace detector;
    ArcFace extractor;
    std::cout << "face_database_make start." << std::endl;

    // output txt
#if TXT
    std::ofstream txtStream;
    txtStream.open(
            "./face_database.txt",
            std::ios::out
    );
#endif
    // output dat
    std::ofstream datStream;
    datStream.open(
            "./face_database.dat",
            std::ios::out | std::ios::binary
    );

    // process
    for(int personCount = 0; personCount < person_num ;++personCount) // person
    {
        printf("person %d Processing...\n",personCount);
       for(int poseCount = 0; poseCount < pose_num ; ++poseCount) // pose
        {
            //read
            cv::Mat img = cv::imread( path_prefix + "/"+ std::to_string(personCount)
                            + "/" + std::to_string(poseCount) + "." + pic_postfix);
            if(!img.data){
                    std::cout << "person :" << personCount <<
                                 " pose :" << poseCount << "read img faild!" << std::endl;
            }
            //detect
            float* retina_data = new float[3 * RetinaFace::INPUT_H * RetinaFace::INPUT_W]();
            float* retina_prob = new float[RetinaFace::OUTPUT_SIZE]();
            cv::Mat re_img = RetinaFace::img_resize(img);
            //cv::imwrite("./debug/re_img_"+ std::to_string(personCount)+ std::to_string(poseCount)+".bmp",re_img );
            RetinaFace::img_preprocess(re_img,retina_data);
            detector.doInference(retina_data, retina_prob);

            std::vector<decodeplugin::Detection> decodeDetection;
            RetinaFace::nms(decodeDetection, retina_prob);
            size_t dets_size = decodeDetection.size();
            std::vector<FaceDetection> res;
            res.resize(dets_size);
            for(size_t i=0;i<dets_size;++i)
            {
                memcpy(&(res[i].bbox[0]),&(decodeDetection[i].bbox[0]),4*sizeof(float));
                res[i].class_confidence = decodeDetection[i].class_confidence;
                memcpy(&(res[i].landmark[0]),&(decodeDetection[i].landmark[0]),10*sizeof(float));
            }

            int max_region_id = 0;
            float max = 0.0;
            for(unsigned int i = 0; i< res.size();++i)// find the face with max iou
            {
                RetinaFace::get_rect_adapt_landmark(img,
                                                    RetinaFace::INPUT_W, RetinaFace::INPUT_H,
                                                    res[i].bbox, res[i].landmark);
                float temp = (res[i].bbox[2] - res[i].bbox[0])*(res[i].bbox[3] - res[i].bbox[1]);
                if(temp > max)
                {
                    max = temp;
                    max_region_id = i;
                }
            }
            cv::Mat face_aligned;
            face_align(img, face_aligned, res[max_region_id]);
#ifdef SAVE_PIC
            cv::imwrite(path_prefix + "/"+ std::to_string(personCount)
                        + "/" + std::to_string(poseCount) + "_aligned.bmp", face_aligned);
 #endif
            //extract
            float* arc_data = new float[3 * ArcFace::INPUT_H * ArcFace::INPUT_W]();
            float* arc_prob = new float[ArcFace::OUTPUT_SIZE]();
            ArcFace::img_preprocess(face_aligned, arc_data);
            extractor.doInference(arc_data, arc_prob);
            cv::Mat out(1, ArcFace::OUTPUT_SIZE, CV_32FC1, arc_prob);
            cv::Mat out_norm;
            cv::normalize(out, out_norm);
            //txt ouput
#if TXT
            memcpy(arc_prob, out_norm.data, ArcFace::OUTPUT_SIZE*sizeof(float));
            for (int dimCount = 0; dimCount < FACE_DESCRIPTOR_LENGTH; ++dimCount)
            {
                    txtStream << arc_prob[dimCount] << ' ';
            }
             txtStream << '\n';
#endif
            //dat output
            datStream.write(reinterpret_cast<char *>(out_norm.data),
             FACE_DESCRIPTOR_LENGTH*sizeof(float));
            delete [] retina_data;
            delete [] retina_prob;
            delete [] arc_data;
            delete [] arc_prob;
        } // pose end
    } // person end

#if TXT
    txtStream.close();
#endif
    datStream.close();

    printf("face_database_make done.\n");
    printf("%d students face data was saved,\n",person_num);
}

int helper(int x)
{
    if(x <= 1)
        return 0;
    else
        return x-1+helper(x-1);
}

// only for debug
void face_database_check(int person_num, int  pose_num,  float threshold)
{
    float* buffer = new float[person_num*pose_num*FACE_DESCRIPTOR_LENGTH];

    std::ifstream datStream;
    datStream.open(
        "./face_database.dat",
        std::ios_base::in | std::ios_base::binary
    );
    if (datStream)
    {
        datStream.read(reinterpret_cast<char *>(buffer),
                person_num * pose_num * FACE_DESCRIPTOR_LENGTH*sizeof(float));
    }
    else
    {
        printf("read dat file err\n");
        datStream.close();
    }
    datStream.close();

    float pos =0;
    for(int i=0;i<person_num;++i){
#ifdef LOG_DETAIL
            printf("stu : %d \n",i);
#endif
        for(int j=0;j<pose_num;++j){
            for(int k = j+1;k<pose_num;++k){
                cv::Mat vec1(1, FACE_DESCRIPTOR_LENGTH, CV_32FC1, &buffer[(i*pose_num+j)*FACE_DESCRIPTOR_LENGTH]);
                cv::Mat vec2(FACE_DESCRIPTOR_LENGTH, 1, CV_32FC1, &buffer[(i*pose_num+k)*FACE_DESCRIPTOR_LENGTH]);
                cv::Mat score = vec1 * vec2 ;
#ifdef LOG_DETAIL
                    printf("\tscore : %f \n", *(float*)score.data);
#endif
                if(*(float*)score.data <= threshold){
                    printf("\tplease check stu %d, photo between %d and %d \n", i,j,k);
                }
                pos  += *(float*)score.data;
            }
        }
    }
    pos /= person_num*helper(pose_num);
    printf(" pos : %f\n", pos);

    delete [] buffer;
}

/*void face_database_get(const char *path, float* database_buf, int person_num, int pose_num)
{
    std::ifstream datStream;
    datStream.open(path, std::ios_base::in | std::ios_base::binary);
    if (datStream)
    {
        datStream.read(reinterpret_cast<char *>(database_buf),
                person_num * pose_num * FACE_DESCRIPTOR_LENGTH * sizeof(float));
    }
    else
    {
        printf("read dat file err\n");
        datStream.close();
        return ;
    }
    datStream.close();
}*/

FaceDB& FaceDB::operator=(const FaceDB& db)
{
    classcode = db.classcode;
    person = db.person;
    pose = db.pose;
    int size = person*pose*FACE_DESCRIPTOR_LENGTH;
    data = new float[size];
    std::copy(db.data, db.data + size, data);

    return *this;
}

FaceDB::~FaceDB()
{
    if(data != nullptr){
        delete [] data;
    }
}


void FaceDB::load(const std::string &path, std::string code, int person_num, int pose_num)
{
    person = person_num;
    pose = pose_num;
    classcode = code;

    if(data != nullptr)
        delete [] data;
    data = new float[person * pose * FACE_DESCRIPTOR_LENGTH]();

    std::ifstream datStream;
    datStream.open(path, std::ios_base::in | std::ios_base::binary);
    if (datStream)
    {
        datStream.read(reinterpret_cast<char *>(data),
                person_num * pose_num * FACE_DESCRIPTOR_LENGTH * sizeof(float));
        printf("FaceDB : load dat file success\n");
    }
    else
    {
        printf("FaceDB : load dat file err\n");
        datStream.close();
        return ;
    }
    datStream.close();
}

void FaceDB::clear()
{
    if(data != nullptr){
        delete [] data;
        data = nullptr;

        classcode = "null";
        person = 0;
        pose = 0;
    }
}

