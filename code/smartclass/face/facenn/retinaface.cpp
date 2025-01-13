#include <fstream>
#include <iostream>
#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "logging.h"
#include "retinaface.h"
#include "retinadecode.h"


#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id

using namespace nvinfer1;

static const char* INPUT_BLOB_NAME = "data";
static const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

RetinaFace::RetinaFace(const char* engine_path)
{
    //set GPU
    cudaSetDevice(DEVICE);

    //read engine
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file(engine_path, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    // runtime
    m_runtime = createInferRuntime(gLogger);
    assert(m_runtime != nullptr);
    // engine
    m_engine = m_runtime->deserializeCudaEngine(trtModelStream, size);
    assert(m_engine != nullptr);
    // context
    m_context = m_engine->createExecutionContext();
    assert(m_context != nullptr);
    delete[] trtModelStream;

    std::cout<< "RetinaFace : build context done!" <<std::endl;
}


RetinaFace::~RetinaFace(){
    // Destroy the engine
    m_context->destroy();
    m_engine->destroy();
    m_runtime->destroy();
}


void RetinaFace::doInference(float* input, float* output, int batchSize) {
    const ICudaEngine& engine = m_context->getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    m_context->enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


cv::Mat RetinaFace::img_resize(cv::Mat& img)
{
    int input_w = INPUT_W;
    int input_h = INPUT_H;

    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);

    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }

    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));

    if((img.rows == h)&&(img.cols == w)){
        img.copyTo(out(cv::Rect(x, y, img.cols, img.rows)));
        return out;
    }

    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}


void RetinaFace::img_preprocess(cv::Mat& img, float* p_data)
{
    /*
    for (int i = 0; i < INPUT_H * INPUT_W; i++)
    {
        p_data[i] = img.at<cv::Vec3b>(i)[0] - 104.0;
        p_data[i + INPUT_H * INPUT_W] = img.at<cv::Vec3b>(i)[1] - 117.0;
        p_data[i + 2 * INPUT_H * INPUT_W] = img.at<cv::Vec3b>(i)[2] - 123.0;
    }*/
    float offset[3] = {104.0, 117.0, 123.0};
#pragma omp parallel for
    for(int i=0;i<3;++i)
    {
        for (int j = 0; j< INPUT_H * INPUT_W; ++j)
        {
            p_data[j + i*INPUT_H * INPUT_W] = img.at<cv::Vec3b>(j)[i] - offset[i];
        }
    }
}


float RetinaFace::iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0], rbox[0]), //left
        std::min(lbox[2], rbox[2]), //right
        std::max(lbox[1], rbox[1]), //top
        std::min(lbox[3], rbox[3]), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) -interBoxS + 0.000001f);
}


bool RetinaFace::cmp(const decodeplugin::Detection& a, const decodeplugin::Detection& b)
{
    return a.class_confidence > b.class_confidence;
}


void RetinaFace::nms(std::vector<decodeplugin::Detection>& res, float *output,
                     float thresh_conf, float thresh_nms)
{
    std::vector<decodeplugin::Detection> dets;
    for (int i = 0; i < output[0]; i++) {
        if (output[15 * i + 1 + 4] <= thresh_conf) continue;
        float width = output[15*i+1+2] - output[15*i+1+0];
        float height = output[15*i+1+3] - output[15*i+1+1];
        if(width < 5 || height < 5) continue;

        decodeplugin::Detection det;
        memcpy(&det, &output[15 * i + 1], sizeof(decodeplugin::Detection));
        dets.push_back(det);
    }
    std::sort(dets.begin(), dets.end(), cmp);
    for (size_t m = 0; m < dets.size(); ++m) {
        auto& item = dets[m];
        res.push_back(item);
        //std::cout << item.class_confidence << " bbox " << item.bbox[0] << ", " << item.bbox[1] << ", " << item.bbox[2] << ", " << item.bbox[3] << std::endl;
        for (size_t n = m + 1; n < dets.size(); ++n) {
            if (iou(item.bbox, dets[n].bbox) > thresh_nms) {
                dets.erase(dets.begin()+n);
                --n;
            }
        }
    }
}

void RetinaFace::get_rect_adapt_landmark(cv::Mat& img, int input_w, int input_h, float bbox[4], float lmk[10]) {
    int l, r, t, b;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] / r_w;
        r = bbox[2] / r_w;
        t = (bbox[1] - (input_h - r_w * img.rows) / 2) / r_w;
        b = (bbox[3] - (input_h - r_w * img.rows) / 2) / r_w;
        bbox[0] = l;
        bbox[1] = t;
        bbox[2] = r;
        bbox[3] = b;
        for (int i = 0; i < 10; i += 2) {
            lmk[i] /= r_w;
            lmk[i + 1] = (lmk[i + 1] - (input_h - r_w * img.rows) / 2) / r_w;
        }
    } else {
        l = (bbox[0] - (input_w - r_h * img.cols) / 2) / r_h;
        r = (bbox[2] - (input_w - r_h * img.cols) / 2) / r_h;
        t = bbox[1] / r_h;
        b = bbox[3] / r_h;
        bbox[0] = l;
        bbox[1] = t;
        bbox[2] = r;
        bbox[3] = b;
        for (int i = 0; i < 10; i += 2) {
            lmk[i] = (lmk[i] - (input_w - r_h * img.cols) / 2) / r_h;
            lmk[i + 1] /= r_h;
        }
    }
   // return cv::Rect(l, t, r-l, b-t);
}
