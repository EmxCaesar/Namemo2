#ifndef _ARCFACE_TENSORRT_H_
#define _ARCFACE_TENSORRT_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include "NvInfer.h"

#define FACE_DESCRIPTOR_LENGTH ArcFace::OUTPUT_SIZE

class ArcFace
{
public:
	static const int INPUT_H = 112;
	static const int INPUT_W = 112;
	static const int OUTPUT_SIZE = 512;
private:
	nvinfer1::IRuntime* m_runtime;
	nvinfer1::ICudaEngine* m_engine;
	nvinfer1::IExecutionContext* m_context;
public:
	ArcFace(const char* engine_path = "arcface-r50.engine");
	~ArcFace();

	void doInference(float* input, float* output, int batchSize = 1);

    static void img_preprocess(cv::Mat& img, float* data);
    //static void img_preprocess(std::vector<cv::Mat>& srcs, cv::Mat& dst, int& batchSize);
};

#endif
