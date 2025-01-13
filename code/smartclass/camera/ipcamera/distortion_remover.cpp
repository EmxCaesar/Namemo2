#include "distortion_remover.h"
#include <iostream>

DistortionRemover::DistortionRemover(std::string calibResPath, double alpha)
{
    cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32FC1);
    cv::Mat distortionCoefficients;

    cv::FileStorage calibResult(calibResPath, cv::FileStorage::READ);
    if (!calibResult.isOpened())
    {
        std::cout << "open " << calibResPath << "faild!\n";
        exit(-1);
    }
    std::cout <<std::endl << "DistortionRemover: "<<std::endl;
    std::cout <<"found calib file: "<<calibResPath<<std::endl;

    calibResult["image_width"] >> m_imgWidth;
    std::cout << "imgWidth: " << m_imgWidth << std::endl;
    calibResult["image_height"] >> m_imgHeight;
    std::cout << "imgHeight: " << m_imgHeight << std::endl;

    calibResult["camera_matrix"] >> cameraMatrix;
    std::cout << "cameraMatrix: "<< std::endl<<
        cameraMatrix << std::endl;

    calibResult["distortion_coefficients"] >> distortionCoefficients;
    std::cout << "distortionCoefficients: "<< std::endl <<
        distortionCoefficients << std::endl<< std::endl;

    calibResult.release();

    cv::Size imageSize(m_imgWidth, m_imgHeight);
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix,
        distortionCoefficients, imageSize, alpha, imageSize, 0);
    cv::initUndistortRectifyMap(cameraMatrix, distortionCoefficients,
        cv::Mat(), newCameraMatrix, imageSize, CV_16SC2, m_map1, m_map2);
}

DistortionRemover::DistortionRemover(const cv::Mat& cameraMatrix, 
    const cv::Mat& distortionCoefficients,
    int imgWidth, int imgHeight, double alpha):
    m_imgWidth(imgWidth), m_imgHeight(imgHeight)
{
    cv::Size imageSize(imgWidth, imgHeight);
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix,
        distortionCoefficients, imageSize, alpha, imageSize, 0);
    cv::initUndistortRectifyMap(cameraMatrix, distortionCoefficients,
        cv::Mat(), newCameraMatrix, imageSize, CV_16SC2, m_map1, m_map2);
}

void DistortionRemover::undistort(cv::Mat& srcImg, cv::Mat& dstImg)
{
    if ((srcImg.cols != m_imgWidth) || (srcImg.rows != m_imgHeight)) {
        std::cout << "the size of srcImg is incompatible!" << std::endl;
    }

    remap(srcImg, dstImg, m_map1, m_map2, cv::INTER_LINEAR);
}


