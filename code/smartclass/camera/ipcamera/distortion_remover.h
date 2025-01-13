#pragma once



#include <opencv2/opencv.hpp>

class DistortionRemover
{
private:
    int m_imgWidth;
    int m_imgHeight;
    cv::Mat m_map1, m_map2;

public:
    /**
     * @brief DistortionRemover构造函数
     *
     * @param calibResPath opencv生成的标定结果文件out_camera_data.yml的路径
     * @param alpha 矫正后是否产生黑边 范围0~1 0表示不生成黑边
     */
    DistortionRemover(std::string calibResPath, double alpha = 0);

    /**
     * @brief DistortionRemover构造函数
     *
     * @param cameraMatrix 3x3相机内参数矩阵
     * @param distortionCoefficients 5x1相机镜头畸变系数
     * @param imgWidth 图像宽
     * @param imgHeight 图像高
     * @param alpha 矫正后是否产生黑边 范围0~1 0表示不生成黑边
     */
    DistortionRemover(const cv::Mat& camera_matrix, const cv::Mat& distortionCoefficients,
        int imgWidth, int imgHeight,double alpha = 0);

    /**
     * @brief 函数简介
     *
     * @param srcImg 输入 
     * @param dstImg 输出
     */
    void undistort(cv::Mat& srcImg, cv::Mat& dstImg);
};

