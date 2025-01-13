#ifndef _RETINA_FACE_H_
#define _RETINA_FACE_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include "NvInfer.h"
#include "retinadecode.h"

class RetinaFace
{
public:
    // H, W must be able to  be divided by 32.
    static const int INPUT_H = 1088;
    static const int INPUT_W = 1920;
    static const int OUTPUT_SIZE = (INPUT_H / 8 * INPUT_W / 8 +INPUT_H / 16 * INPUT_W / 16
                                    + INPUT_H / 32 * INPUT_W / 32) * 2  * 15 + 1;

private:
	nvinfer1::IRuntime* m_runtime;
	nvinfer1::ICudaEngine* m_engine;
	nvinfer1::IExecutionContext* m_context;

    static inline float iou(float lbox[4], float rbox[4]);
    static inline bool cmp(const decodeplugin::Detection& a, const decodeplugin::Detection& b);

public:
    RetinaFace(const char* engine_path = "retina-r50.engine");
	~RetinaFace();

	void doInference(float* input, float* output, int batchSize = 1);

    static  cv::Mat img_resize(cv::Mat& img);
    static  void img_preprocess(cv::Mat& img, float* data);

    static void nms(std::vector<decodeplugin::Detection>& res,
                        float *output, float thresh_conf = 0.9, float thresh_nms = 0.4);
    static void get_rect_adapt_landmark(
            cv::Mat& img, int input_w, int input_h, float bbox[4], float lmk[10]);
};







#endif
