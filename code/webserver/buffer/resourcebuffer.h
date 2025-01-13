#ifndef _SRC_BUFFER_H_
#define _SRC_BUFFER_H_

#include <queue>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <iostream>
#include <string>

template <typename T>
class ResourceBuffer
{
public:
    void push(T&& t){
        queue.push(t);
    }

    bool get(T& t){
        if(queue.empty())
            return false;

        t = queue.front();
        queue.pop();

        return true;
    }

    virtual bool getToFile(std::string filename, std::string& path){ return false;}
    virtual bool getToFiles(std::string filename, std::vector<std::string>& paths) {return false;}
protected:
    std::queue<T> queue;

};

class StringBuffer:public ResourceBuffer<std::string>
{
public:
    bool getToFile(std::string filename, std::string& path) override
    {
        std::string s;
        bool ret = get(s);
        if(!ret)
            return false;

        std::ofstream file(filename + ".txt", std::ios::trunc);
        file << s;
        file.close();

        path = filename + ".txt";
        return true;
    }
};

class MatBuffer:public ResourceBuffer<cv::Mat>
{
public:
    bool getToFile(std::string filename, std::string& path) override
    {
        cv::Mat img;
        bool ret = get(img);
        if(!ret)
            return false;
        cv::imwrite(filename + ".jpg", img);
        path = filename + ".jpg";
        return true;
    }
};

#endif
