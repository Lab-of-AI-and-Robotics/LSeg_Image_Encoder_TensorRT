#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime.h"
#include "cnpy.h"
#include <vector>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

using namespace nvinfer1;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

ICudaEngine* loadEngine(const std::string& enginePath, ILogger& logger) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error opening engine file: " << enginePath << std::endl;
        return nullptr;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size);
    delete runtime;
    return engine;
}

void preprocessImage(const cv::Mat& img, float* inputBuffer, int inputWidth, int inputHeight) {
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized, floatImg;
    cv::resize(rgb, resized, cv::Size(inputWidth, inputHeight));
    resized.convertTo(floatImg, CV_32FC3, 1.0f / 255.0f);

    const float mean[3] = {0.5f, 0.5f, 0.5f};
    const float stdv[3] = {0.5f, 0.5f, 0.5f};
    int idx = 0;
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < inputHeight; i++) {
            for (int j = 0; j < inputWidth; j++) {
                float pixel = floatImg.at<cv::Vec3f>(i, j)[c];
                inputBuffer[idx++] = (pixel - mean[c]) / stdv[c];
            }
        }
    }
}

std::vector<float> run_trt_inference(ICudaEngine* engine, IExecutionContext* context,
                                      const cv::Mat& img, int size) {
    // Find I/O tensor names and sizes
    int nbIO = engine->getNbIOTensors();
    int inputIdx = -1, outputIdx = -1;
    for (int i = 0; i < nbIO; ++i) {
        const char* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        if (mode == TensorIOMode::kINPUT)  inputIdx = i;
        if (mode == TensorIOMode::kOUTPUT) outputIdx = i;
    }
    std::string inputName  = engine->getIOTensorName(inputIdx);
    std::string outputName = engine->getIOTensorName(outputIdx);

    // 1) 입력 차원 설정
    Dims4 dyn{1, 3, size, size};
    if (!context->setInputShape(inputName.c_str(), dyn)) {
        std::cerr << "[ERROR] Failed to set dynamic shape to "
                  << size << "×" << size << std::endl;
        return {};
    }

    // 2) 실제 출력 dims 구하기
    auto outDims = context->getTensorShape(outputName.c_str());
    int outputElems = 1;
    for (int d = 0; d < outDims.nbDims; ++d) {
        outputElems *= outDims.d[d];
    }

    int inputElems = 3 * size * size;
    size_t inputBytes  = static_cast<size_t>(inputElems)  * sizeof(float);
    size_t outputBytes = static_cast<size_t>(outputElems) * sizeof(float);

    // 3) 버퍼 할당 & 전처리
    std::vector<float> inputData(inputElems);
    preprocessImage(img, inputData.data(), size, size);
    void* d_input;  cudaMalloc(&d_input,  inputBytes);
    void* d_output; cudaMalloc(&d_output, outputBytes);
    cudaMemcpy(d_input, inputData.data(), inputBytes, cudaMemcpyHostToDevice);

    // 4) 바인딩, 실행, 복사
    context->setTensorAddress(inputName.c_str(),  d_input);
    context->setTensorAddress(outputName.c_str(), d_output);

    // Execute
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    // Copy output
    std::vector<float> outputData(outputElems);
    cudaMemcpy(outputData.data(), d_output, outputBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    return outputData;
}

void process_and_save_feature(const std::string& enginePath,
                              const std::string& imgPath,
                              int size,
                              const std::string& tag) {
    Logger logger;
    ICudaEngine* engine = loadEngine(enginePath, logger);
    if (!engine) {
        std::cerr << "Failed to load engine: " << enginePath << std::endl;
        return;
    }
    IExecutionContext* context = engine->createExecutionContext();

    cv::Mat img = cv::imread(imgPath);
    if (img.empty()) {
        std::cerr << "Error loading image: " << imgPath << std::endl;
        delete context;
        delete engine;
        return;
    }

    auto featureMap = run_trt_inference(engine, context, img, size);

    // process_and_save_feature() 안에서도 outputName 을 구해야 shape 을 알 수 있습니다.
    int nbIO = engine->getNbIOTensors();
    int outputIdx = -1;
    for (int i = 0; i < nbIO; ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == TensorIOMode::kOUTPUT) {
            outputIdx = i;
            break;
        }
    }
    if (outputIdx < 0) {
        std::cerr << "[ERROR] Couldn't find output tensor name\n";
        delete context;
        delete engine;
        return;
    }
    std::string outputName = engine->getIOTensorName(outputIdx);
    auto outDims = context->getTensorShape(outputName.c_str());

    std::vector<size_t> shape;
    for (int d = 0; d < outDims.nbDims; ++d) {
        shape.push_back(static_cast<size_t>(outDims.d[d]));
    }

    // Create outputs directory
    if (mkdir("outputs", 0777) && errno != EEXIST) {
        std::cerr << "Failed to create outputs directory" << std::endl;
    }
    
    std::string name = imgPath.substr(imgPath.find_last_of("/") + 1);
    name = name.substr(0, name.find_last_of("."));
    std::string outFile = "outputs/trt_vit_" + tag + "_" + std::to_string(size)
                        + "_fMap_" + name + ".npy";
    cnpy::npy_save(outFile, featureMap.data(), shape);
    std::cout << "[INFO] Saved: " << outFile << std::endl;

    delete context;
    delete engine;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <engine.trt> <image_path> <size1> [size2 ...]" << std::endl;
        return -1;
    }
    std::string enginePath = argv[1];
    std::string imgPath    = argv[2];
    std::string engineFile = enginePath.substr(enginePath.find_last_of("/") + 1);
    std::string tag = "custom";
    if (engineFile.find("ade20k") != std::string::npos) tag = "ade20k";
    if (engineFile.find("fss")    != std::string::npos) tag = "fss";

    for (int i = 3; i < argc; ++i) {
        int size = std::stoi(argv[i]);
        process_and_save_feature(enginePath, imgPath, size, tag);
    }
    return 0;
}
