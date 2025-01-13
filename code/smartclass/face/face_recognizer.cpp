#include "face_recognizer.h"
#include "../common/chrono_timer.h"
#include "face_database.h"

#define SAVE_PIC 0
#define TIMER 0

FaceRecognizer::FaceRecognizer(const char *engine_path)
{
    pFaceRecognizer = new ArcFace(engine_path);    
}

FaceRecognizer::~FaceRecognizer()
{
    delete pFaceRecognizer;
}


void FaceRecognizer::setDataBase(FaceDB *db)
{
    if(db->empty() || db->classcode == faceDB.classcode)
        return;
    faceDB = *db;
}

void FaceRecognizer::clearDataBase()
{
    faceDB.clear();
}

void FaceRecognizer::process(stitching::Image &image)
{
    if(faceDB.empty()){
        std::cout << "FaceRecognizer: FaceDB empty!" << std::endl;
        return;
    }

    int vec_size = image.vec_faceinfo.size();
    data = new float[vec_size * 3 * ArcFace::INPUT_H * ArcFace::INPUT_W]();
    prob = new float[vec_size * ArcFace::OUTPUT_SIZE]();

    // process aligned face***************************************************8
#if TIMER
    ChronoTimer timer;
    timer.start();
#endif

#pragma omp parallel for
    for(int i = 0;i<vec_size;++i)
    {
        ArcFace::img_preprocess(image.vec_faceinfo[i].face_aligned,
                                &data[i * 3 * ArcFace::INPUT_H * ArcFace::INPUT_W]);
        // free pic data face_aligned
        image.vec_faceinfo[i].face_aligned.release();
    }

#if TIMER
    timer.end();
    std::cout<<"arc batch img_preprocess :";
    timer.print_elapse();
    timer.start();
#endif

    for(int i = 0;i<vec_size;++i)
    {
        pFaceRecognizer->doInference(&data[i * 3 * ArcFace::INPUT_H * ArcFace::INPUT_W],
                &prob[i * ArcFace::OUTPUT_SIZE]);
    }

#if TIMER
    timer.end();
    std::cout<<"arc batch inference :";
    timer.print_elapse();
    timer.start();
#endif

    // save result
#pragma omp parallel for
    for(int i = 0;i<vec_size;++i)
    {
        // save result
        cv::Mat temp(1,FACE_DESCRIPTOR_LENGTH, CV_32FC1, &prob[i *FACE_DESCRIPTOR_LENGTH]);
        cv::normalize(temp, image.vec_faceinfo[i].face_descriptor);
        // compare
        int person_id = 0;
        float max_sim = 0.0f;
        for(int personCount = 0; personCount < faceDB.person ; ++personCount)
        {
            for(int poseCount = 0; poseCount < faceDB.pose ; ++poseCount)
            {
                cv::Mat data(FACE_DESCRIPTOR_LENGTH, 1,CV_32FC1,
                             &(faceDB.data)[(personCount*faceDB.pose+poseCount)*FACE_DESCRIPTOR_LENGTH]);
                cv::Mat score = image.vec_faceinfo[i].face_descriptor * data;
                if(*(float*)score.data > max_sim)
                {
                    max_sim = *(float*)score.data;
                    person_id = personCount;
                }
            }
        }
        // save result
        image.vec_faceinfo[i].id = person_id;
        image.vec_faceinfo[i].score = max_sim;
#if SAVE_PIC
        cv::imwrite("./debug/debug_recog/id_"+std::to_string(person_id) + "_score_"+ std::to_string(max_sim) + ".bmp",
                    image.vec_faceinfo[i].face_aligned);
#endif
    }

#if TIMER
    timer.end();
    std::cout<<"arc batch compare :";
    timer.print_elapse();
#endif

    //process aligned face env********************************************

    if(image.b_use_env){
        memset(data,0,vec_size * 3 * ArcFace::INPUT_H * ArcFace::INPUT_W);
        memset(prob,0,vec_size * ArcFace::OUTPUT_SIZE);

#pragma omp parallel for
        for(int i = 0;i<vec_size;++i)
        {
            ArcFace::img_preprocess(image.vec_faceinfo[i].face_env_aligned,
                                    &data[i * 3 * ArcFace::INPUT_H * ArcFace::INPUT_W]);
            // free pic data cloth
            image.vec_faceinfo[i].face_env_aligned.release();
        }

        for(int i = 0;i<vec_size;++i)
        {
            pFaceRecognizer->doInference(&data[i * 3 * ArcFace::INPUT_H * ArcFace::INPUT_W],
                    &prob[i * ArcFace::OUTPUT_SIZE]);
        }

#pragma omp parallel for
        for(int i = 0;i<vec_size;++i)
        {
            cv::Mat temp(1,FACE_DESCRIPTOR_LENGTH, CV_32FC1, &prob[i *FACE_DESCRIPTOR_LENGTH]);
            cv::normalize(temp, image.vec_faceinfo[i].face_env_descriptor);
        }

#if TIMER
    timer.end();
    std::cout<<"arc batch face env :";
    timer.print_elapse();
#endif

    }

    delete [] data;
    delete [] prob;
}
