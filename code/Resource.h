#pragma once

#include <mutex>
#include <atomic>
#include <opencv2/opencv.hpp>

class Resource
{
public:
    void lock(){ mtx.lock(); }
    void unlock() { mtx.unlock(); }

    static Resource* instance(){
        static Resource resrc;
        return &resrc;
    }

    void init(){
        idx = 0;
        classcode = "";
        pano = cv::Mat();
        facedata.resize(0);
        text = "";
    }

    void reset()
    {
        idx = 0;
        classcode = "";
        pano = cv::Mat();
        facedata.resize(0);
        text = "";
    }

    std::atomic_int idx;
    std::string classcode;
    cv::Mat pano;
    std::vector<int> facedata;
    std::string text;

private:
    std::mutex mtx;

    Resource() = default;
    ~Resource() = default;
};
