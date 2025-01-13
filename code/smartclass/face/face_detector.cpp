#include "face_detector.h"
#include "face_align.h"
#include "../common/chrono_timer.h"

#define SAVE_ALIGNED 0
#define SAVE_DETECT 1
#define TIMER 0

FaceDetector::FaceDetector(const char *engine_path)
{
    pFaceDetector = new RetinaFace(engine_path);
    data = new float[3 * RetinaFace::INPUT_H * RetinaFace::INPUT_W]();
	prob = new float[RetinaFace::OUTPUT_SIZE];
    pFaceDetector->doInference(data,prob);
}

FaceDetector::~FaceDetector()
{
	delete pFaceDetector;
	delete[] data;
	delete[] prob;	
}

void FaceDetector::process(stitching::Image& image)
{
    if(m_isRelatedROISet){
        image.b_use_env = true;
    }

#if TIMER
    ChronoTimer timer;
    timer.start();
#endif

    cv::Mat re_img = RetinaFace::img_resize(image.img_original);//padding

#if TIMER
    timer.end();
    std::cout<<"retina img_resize :";
    timer.print_elapse();
    timer.start();
#endif

    RetinaFace::img_preprocess(re_img,data);//bbb..ggg..rrr...

#if TIMER
    timer.end();
    std::cout<<"retina img_preprocess :";
    timer.print_elapse();
    timer.start();
#endif

    pFaceDetector->doInference(data, prob);

#if TIMER
    timer.end();
    std::cout<<"retina doInference :";
    timer.print_elapse();
    timer.start();
#endif

    std::vector<FaceDetection> res;
    FaceDetector::nms(res, prob);

#if TIMER
    timer.end();
    std::cout<<"retina nms :";
    timer.print_elapse();
    timer.start();
#endif

    image.vec_faceinfo.resize(res.size());
#if SAVE_DETECT
    cv::Mat tmp = image.img_original.clone();
      cv::imwrite(std::string("./debug/debug_original/round")+std::to_string(image.round_num)+std::string("_original")+std::to_string(image.id)+std::string(".jpg"), tmp);
#endif

#pragma omp parallel for
    for(unsigned int i =0; i<res.size();++i)
    {
        RetinaFace::get_rect_adapt_landmark(image.img_original,
                                            RetinaFace::INPUT_W, RetinaFace::INPUT_H,
                                            res[i].bbox, res[i].landmark);
        image.vec_faceinfo[i].face_detection =  res[i];// save face detect res
        face_align(image.img_original,                              // save face align res
                   image.vec_faceinfo[i].face_aligned,
                   image.vec_faceinfo[i].face_detection
                   );
        if(m_isRelatedROISet){
            face_env_align(image.img_original,
                       image.vec_faceinfo[i].face_env_aligned,
                       image.vec_faceinfo[i].face_detection.bbox,
                       m_ROIAnchorOffsetX,m_ROIAnchorOffsetY,m_ROIRatio);
        }
#if SAVE_ALIGNED
        //save aligned face
        cv::imwrite("./debug/debug_aligned/"+std::string("round")+std::to_string(image.round_num)+std::string("_img")+std::to_string(image.id)+std::string("_face")+std::to_string(i)+".jpg",image.vec_faceinfo[i].face_aligned);

        if(m_isRelatedROISet){
        cv::imwrite("./debug/debug_aligned/"+std::string("round")+std::to_string(image.round_num)+std::string("_img")+std::to_string(image.id)+std::string("_env")+std::to_string(i)+".jpg",image.vec_faceinfo[i].face_env_aligned);
        }
#endif

#if SAVE_DETECT
        //bbox
        cv::Rect r(res[i].bbox[0], res[i].bbox[1],
                res[i].bbox[2]-res[i].bbox[0], res[i].bbox[3]-res[i].bbox[1]);
        cv::rectangle(tmp, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        //class_confidence
        cv::putText(tmp, std::to_string((int)(res[i].class_confidence * 100)) + "%",
                    cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
        //landmark
        for (int j = 0; j < 10; j += 2)
        {
            cv::circle(tmp, cv::Point(res[i].landmark[j], res[i].landmark[j + 1]),
                    1, cv::Scalar(255 * (j > 2), 255 * (j > 0 && j < 8), 255 * (j < 6)), 4);
        }
#endif
    }
#if SAVE_DETECT
    cv::imwrite(std::string("./debug/debug_detect/round")+std::to_string(image.round_num)+std::string("_detect")+std::to_string(image.id)+std::string(".jpg"), tmp);
#endif

#if TIMER
    timer.end();
    std::cout<<"retina align :";
    timer.print_elapse();
#endif
}

void FaceDetector::nms(std::vector<FaceDetection>& res, float* prob)
{
    std::vector<decodeplugin::Detection> decodeDetection;
    RetinaFace::nms(decodeDetection, prob);

    // change to FaceDetection
    size_t dets_size = decodeDetection.size();
    res.resize(dets_size);
    for(size_t i=0;i<dets_size;++i)
    {
        memcpy(&(res[i].bbox[0]),&(decodeDetection[i].bbox[0]),4*sizeof(float));
        res[i].class_confidence = decodeDetection[i].class_confidence;
        memcpy(&(res[i].landmark[0]),&(decodeDetection[i].landmark[0]),10*sizeof(float));
    }
}

void FaceDetector::setRelatedROI(float roi_anchor_offset_x,
                                 float roi_anchor_offset_y, float roi_ratio)
{
    m_isRelatedROISet = true;
    m_ROIAnchorOffsetX = roi_anchor_offset_x;
    m_ROIAnchorOffsetY = roi_anchor_offset_y;
    m_ROIRatio = roi_ratio;
}
