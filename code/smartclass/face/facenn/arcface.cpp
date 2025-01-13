#include <fstream>
#include <iostream>
#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "logging.h"
#include "arcface.h"

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

ArcFace::ArcFace(const char* engine_path){
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

    std::cout<< "ArcFace : build context done!" <<std::endl;
}

ArcFace::~ArcFace(){
	// Destroy the engine
    m_context->destroy();
    m_engine->destroy();
    m_runtime->destroy();
}


void ArcFace::doInference(float* input, float* output, int batchSize) {
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

void ArcFace::img_preprocess(cv::Mat& img, float* data)
{
    // check img size
    if((img.rows != INPUT_H)||(img.cols != INPUT_W))
	{
		std::cout<<"ArcFace::preprocess #: wrong input img size!"<<std::endl;
		return;
	}
    // norm
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = ((float)img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
        data[i + INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
        data[i + 2 * INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
    }
}


