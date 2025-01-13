#ifndef _IP_CAMERA_H_
#define _IP_CAMERA_H_

#include <iostream>
#include <unistd.h>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include "onvif/stdsoap2.h"
#include "onvif/plugin/wsseapi.h"
#include "onvif/soapProxy/soapMediaBindingProxy.h"
#include "onvif/soapProxy/soapPTZBindingProxy.h"
#include "../camera_base.h"
#include "ipc_common.h"
#include "distortion_remover.h"

struct _ocp_PTZStatus
{
    float pan;
    float tilt;
    float zoom;
    tt__MoveStatus move_status_pan_tilt;
    tt__MoveStatus move_status_zoom;
};

class IPCamera:public CameraBase
{
private:
    const std::string m_username;
    const std::string m_password;
    const std::string m_hostname;

    const float m_speedPTX;
    const float m_speedPTY;
    const float m_speedZoom;

    const float m_panLimitsMin = -1;
    const float m_panLimitsMax = 1;
    const float m_tiltLimitsMin = -1;
    const float m_tiltLimitsMax = 1;
    const float m_zoomLimitsMin = 0;
    const float m_zoomLimitsMax = 1;

    std::string m_snapshotUri;
    std::string m_streamUri;

    std::string m_cameraCalibFilePath;
    std::shared_ptr<DistortionRemover> m_pDisRemover;

private:
    std::vector<std::pair<std::string, std::string>> getMediaProfiles();
    void getSnapshotUri(std::string &profileToken);
    void getStreamUri(std::string &profileToken);
    void addAuth();
    void initMediaUri();

    bool PTZStatus(_ocp_PTZStatus &ptzStatus);
    bool PTZAbsoluteMove(const float pantiltX, const float pantiltY, const float zoom,
                const float speedPTX, const float speedPTY, const float speedZoom);

public:
    IPCamera();
    IPCamera(const std::string username,
             const std::string password,
             const std::string hostname,
             const float speedPTX,
             const float speedPTY,
             const float speedZoom,
             const std::string cameraCalibFilePath);
    ~IPCamera();


    void capture(cv::Mat& img, int index) override;
    void PTZMove(const float pantiltX, const float pantiltY, const float zoom) override;
};

#endif
