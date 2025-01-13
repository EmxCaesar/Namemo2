#include "ipcamera.h"
#include "../../common/data_path.h"
#include <sstream>
#include <string>
#include <iostream>

IPCamera::IPCamera():
    m_username(USERNAME),
    m_password(PASSWORD),
    m_hostname(HOSTNAME),
    m_speedPTX(1),m_speedPTY(1),m_speedZoom(1)
{
    m_pDisRemover = std::make_shared<DistortionRemover>(
                "./data/out_camera_data.yml");
    initMediaUri();
}

IPCamera::IPCamera(const std::string username,
                   const std::string password,
                   const std::string hostname,
                   const float speedPTX,
                   const float speedPTY,
                   const float speedZoom,
                   const std::string cameraCalibFilePath):
    m_username(username),
    m_password(password),
    m_hostname(hostname),
    m_speedPTX(speedPTX),
    m_speedPTY(speedPTY),
    m_speedZoom(speedZoom),
    m_cameraCalibFilePath(cameraCalibFilePath)
{
    m_pDisRemover = std::make_shared<DistortionRemover>(
                m_cameraCalibFilePath);
    initMediaUri();
}

IPCamera::~IPCamera(){

}

std::vector<std::pair<std::string, std::string>> IPCamera::getMediaProfiles()
{
    MediaBindingProxy mediaBindingProxy;
    mediaBindingProxy.soap_endpoint = m_hostname.c_str();
    if (SOAP_OK != soap_wsse_add_UsernameTokenDigest(mediaBindingProxy.soap, nullptr, m_username.c_str(), m_password.c_str())) {
        std::cerr << "Error: soap_wsse_add_UsernameTokenDigest" << std::endl;
        report_error(mediaBindingProxy.soap);
        return std::vector<std::pair<std::string, std::string>>();
    }

    struct soap* soap = soap_new();
    _trt__GetProfiles *GetProfiles = soap_new__trt__GetProfiles(soap);
    _trt__GetProfilesResponse *GetProfilesResponse = soap_new__trt__GetProfilesResponse(soap);

    std::vector<std::pair<std::string, std::string>> profiles;
    std::cout << "Profile name" << "         " << "Profile token" << std::endl;

    while(1){
        if (SOAP_OK == mediaBindingProxy.GetProfiles(GetProfiles, *GetProfilesResponse)) {
            for (auto & Profile : GetProfilesResponse->Profiles) {
                std::cout << Profile->Name << "         " << Profile->token << std::endl;
                profiles.emplace_back(std::pair<std::string, std::string>(Profile->Name, Profile->token));
            }
            break;
        } else {
            std::cerr <<"getProfiles faild, retry in 10 seconds" << std::endl;
            sleep(10);
            //report_error(mediaBindingProxy.soap);
        }
    }


    CLEANUP_SOAP(soap);

    return profiles;
}

void IPCamera::getSnapshotUri(std::string &profileToken) {
    MediaBindingProxy mediaBindingProxy;
    mediaBindingProxy.soap_endpoint = m_hostname.c_str();
    if (SOAP_OK != soap_wsse_add_UsernameTokenDigest(mediaBindingProxy.soap, nullptr, m_username.c_str(), m_password.c_str())) {
        std::cerr << "Error: soap_wsse_add_UsernameTokenDigest" << std::endl;
        report_error(mediaBindingProxy.soap);
        return;
    }

    struct soap* soap = soap_new();
    _trt__GetSnapshotUri *GetSnapshotUri = soap_new__trt__GetSnapshotUri(soap);
    GetSnapshotUri->ProfileToken = profileToken;
    _trt__GetSnapshotUriResponse *GetSnapshotUriResponse = soap_new__trt__GetSnapshotUriResponse(soap);

    if (SOAP_OK == mediaBindingProxy.GetSnapshotUri(GetSnapshotUri, *GetSnapshotUriResponse)) {
        std::cout << "SnapshotUri: " << GetSnapshotUriResponse->MediaUri->Uri << std::endl;
        m_snapshotUri = GetSnapshotUriResponse->MediaUri->Uri;
    } else {
        std::cerr <<"Error: getSnapshotUri" << std::endl;
        report_error(mediaBindingProxy.soap);
    }

    CLEANUP_SOAP(soap);
}

void IPCamera::getStreamUri(std::string &profileToken)
{
    MediaBindingProxy mediaBindingProxy;
    mediaBindingProxy.soap_endpoint = m_hostname.c_str();
    if (SOAP_OK != soap_wsse_add_UsernameTokenDigest(mediaBindingProxy.soap, nullptr, m_username.c_str(), m_password.c_str())) {
        std::cerr << "Error: soap_wsse_add_UsernameTokenDigest" << std::endl;
        report_error(mediaBindingProxy.soap);
        return;
    }

    struct soap* soap = soap_new();
    _trt__GetStreamUri *GetStreamUri = soap_new__trt__GetStreamUri(soap);
    GetStreamUri->ProfileToken = profileToken;
    _trt__GetStreamUriResponse *GetStreamUriResponse = soap_new__trt__GetStreamUriResponse(soap);

    if (SOAP_OK == mediaBindingProxy.GetStreamUri(GetStreamUri, *GetStreamUriResponse)) {
        std::cout << "StreamUri: " << GetStreamUriResponse->MediaUri->Uri << std::endl;
        m_streamUri = GetStreamUriResponse->MediaUri->Uri;
    } else {
        std::cout <<"Error: getStreamUri" << std::endl;
        report_error(mediaBindingProxy.soap);
    }

    CLEANUP_SOAP(soap);
}

void IPCamera::addAuth()
{
    static bool flag = false;

    if (flag == true)
        return;

    int index = m_snapshotUri.find("//");
    index += 2;

    std::stringstream ss_auth;
    ss_auth << m_username << ":"<<m_password<<"@";
    std::string str_auth = ss_auth.str();
    m_snapshotUri.insert(index,str_auth);
    flag = true;

    //std::cout << "uri_with_auth : ";
    //std::cout << snapshotUri << std::endl;
}

void IPCamera::initMediaUri()
{
    std::vector<std::pair<std::string, std::string>> profiles = getMediaProfiles();

    if (!profiles.empty()) {
        std::cout << "====================== MediaBinding GetSnapshotUri ======================" << std::endl;
        getSnapshotUri(profiles[0].second);

        std::cout << "====================== MediaBinding GetStreamUri ======================" << std::endl;
        getStreamUri(profiles[0].second);
        std::cout <<std::endl;
    }else{
        std::cout << "Unable to get URI, MediaProfiles empty!\n\n";
    }

    addAuth();
}

void IPCamera::capture(cv::Mat &img, int index)
{
    if(m_snapshotUri.empty()){
        std::cout << "capture img faild! empty snapshotUri!\n";
        return;
    }

    //char img_name[50];
    //sprintf(img_name, "./download/download_%d.jpg",index);

    std::string img_name =
            std::string("./download/download_")+
            std::to_string(index)+std::string(".jpg");

    char cmd[256];
    sprintf(cmd,"wget -O %s -q '%s'",img_name.c_str(),m_snapshotUri.c_str());
    int cmd_ret = 0;
    cmd_ret = system(cmd);
    if(cmd_ret){
        std::cout << "download faild! index: "<<index<<std::endl;
    }
    //std::cout <<  img_name << " download success.\n";

    cv::Mat img_download;
    //try 3 times
    for(size_t i=0;i<3;++i){
        img_download= cv::imread(img_name, cv::IMREAD_COLOR);
        if(!img_download.data){
           std::cout << "read download jpg faild!" <<std::endl;
           sleep(1);
           continue;
        }else{
            break;
        }
    }

    m_pDisRemover->undistort(img_download, img);

}

bool IPCamera::PTZStatus(_ocp_PTZStatus &ptzStatus)
{
    std::string profileToken = PROFILETOKEN;

    PTZBindingProxy ptzBindingProxy;
    ptzBindingProxy.soap_endpoint = m_hostname.c_str();
    if (SOAP_OK != soap_wsse_add_UsernameTokenDigest(ptzBindingProxy.soap, nullptr, m_username.c_str(), m_password.c_str())) {
        std::cout << "Error: soap_wsse_add_UsernameTokenDigest" << std::endl;
        report_error(ptzBindingProxy.soap);
        return false;
    }

    struct soap* soap = soap_new();
    _tptz__GetStatus *GetStatus = soap_new__tptz__GetStatus(soap);
    _tptz__GetStatusResponse *GetStatusResponse = soap_new__tptz__GetStatusResponse(soap);

    const tt__ReferenceToken& token = profileToken;
    GetStatus->ProfileToken = token;

    bool executeResult = false;
    if (SOAP_OK == ptzBindingProxy.GetStatus(GetStatus, *GetStatusResponse)) {
        if (GetStatusResponse->PTZStatus) {
            if (GetStatusResponse->PTZStatus->Position) {
                ptzStatus.pan = GetStatusResponse->PTZStatus->Position->PanTilt->x;
                ptzStatus.tilt = GetStatusResponse->PTZStatus->Position->PanTilt->y;
                ptzStatus.zoom = GetStatusResponse->PTZStatus->Position->Zoom->x;
            } else {
                std::cout << "Error get ptz position" << std::endl;
            }
            if (GetStatusResponse->PTZStatus->MoveStatus) {
                if (GetStatusResponse->PTZStatus->MoveStatus->PanTilt) {
                    ptzStatus.move_status_pan_tilt = *(GetStatusResponse->PTZStatus->MoveStatus->PanTilt);
                } else {
                    ptzStatus.move_status_pan_tilt = tt__MoveStatus::tt__MoveStatus__UNKNOWN;
                }
                if (GetStatusResponse->PTZStatus->MoveStatus->Zoom) {
                    ptzStatus.move_status_zoom = *(GetStatusResponse->PTZStatus->MoveStatus->Zoom);
                } else {
                    ptzStatus.move_status_zoom = tt__MoveStatus::tt__MoveStatus__UNKNOWN;
                }
            } else {
                ptzStatus.move_status_pan_tilt = tt__MoveStatus::tt__MoveStatus__UNKNOWN;
                ptzStatus.move_status_zoom = tt__MoveStatus::tt__MoveStatus__UNKNOWN;
            }
            executeResult = true;
        }
    } else {
        std::cout <<"Error: getStatus" << std::endl;
        report_error(ptzBindingProxy.soap);
    }

    CLEANUP_SOAP(soap);

    return executeResult;
}

bool IPCamera::PTZAbsoluteMove(
        const float pantiltX, const float pantiltY, const float zoom,
        const float speedPTX, const float speedPTY, const float speedZoom)
{
    std::string profileToken = PROFILETOKEN;

    if (pantiltX < m_panLimitsMin || pantiltX > m_panLimitsMax || pantiltY < m_tiltLimitsMin || pantiltY > m_tiltLimitsMax
        || zoom < m_zoomLimitsMin || zoom > m_zoomLimitsMax) {
        std::cout << "Destination out of bounds" << std::endl;
        return false;
    }

    PTZBindingProxy ptzBindingProxy;
    ptzBindingProxy.soap_endpoint = m_hostname.c_str();
    if (SOAP_OK != soap_wsse_add_UsernameTokenDigest(ptzBindingProxy.soap, nullptr, m_username.c_str(), m_password.c_str())) {
        std::cout << "Error: soap_wsse_add_UsernameTokenDigest" << std::endl;
        report_error(ptzBindingProxy.soap);
        return false;
    }

    struct soap* soap = soap_new();
    _tptz__AbsoluteMove *AbsoluteMove = soap_new__tptz__AbsoluteMove(soap);
    _tptz__AbsoluteMoveResponse *AbsoluteMoveResponse = soap_new__tptz__AbsoluteMoveResponse(soap);

    const tt__ReferenceToken& token = profileToken;
    AbsoluteMove->ProfileToken = token;

    AbsoluteMove->Position = soap_new_tt__PTZVector(soap);
    AbsoluteMove->Position->PanTilt = soap_new_tt__Vector2D(soap);
    AbsoluteMove->Position->PanTilt->x = pantiltX;
    AbsoluteMove->Position->PanTilt->y = pantiltY;
    AbsoluteMove->Position->Zoom = soap_new_tt__Vector1D(soap);
    AbsoluteMove->Position->Zoom->x = zoom;
    AbsoluteMove->Speed = soap_new_tt__PTZSpeed(soap);
    AbsoluteMove->Speed->PanTilt = soap_new_tt__Vector2D(soap);
    AbsoluteMove->Speed->PanTilt->x = speedPTX;
    AbsoluteMove->Speed->PanTilt->y = speedPTY;
    AbsoluteMove->Speed->Zoom = soap_new_tt__Vector1D(soap);
    AbsoluteMove->Speed->Zoom->x = speedZoom;

    bool executeResult = false;
    if (SOAP_OK == ptzBindingProxy.AbsoluteMove(AbsoluteMove, *AbsoluteMoveResponse)) {
        executeResult = true;
    } else {
        std::cout <<"Error: absoluteMove" << std::endl;
        report_error(ptzBindingProxy.soap);
    }

    CLEANUP_SOAP(soap);

    return executeResult;
}

void IPCamera::PTZMove(const float pantiltX,
                      const float pantiltY, const float zoom)
{
    bool ret;
    ret = PTZAbsoluteMove(pantiltX, pantiltY, zoom,
            m_speedPTX, m_speedPTY, m_speedZoom);
    if(!ret)
    {
        std::cout << "Error: PTZAbsoluteMove faild!\n ";
        return;
    }
    std::cout << "Info:Moving" << std::endl;

    _ocp_PTZStatus status;
    bool getStatusResult = PTZStatus(status);
    do {
        usleep(500000);//500ms
        getStatusResult = PTZStatus(status);
        if(!getStatusResult){
            std::cout << "Error: PTZStatus went wrong..." << std::endl;
        }
    }
    while ( status.move_status_pan_tilt || status.move_status_zoom);
    //while (getStatusResult && (status.move_status_pan_tilt || status.move_status_zoom));

    std::cout << "Info: stop" << std::endl;

    if (getStatusResult) {
        std::cout << "Info: Pan:  " << status.pan << " ";
        std::cout << "Tilt: " << status.tilt << " ";
        std::cout << "Zoom: " << status.zoom << std::endl;
    }
    return ;
}
