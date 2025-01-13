#ifndef _SMART_CLASS_H_
#define _SMART_CLASS_H_

#include <iostream>
#include <pthread.h>
#include <string>
#include <atomic>
#include "common/image_utils.h"
#include "common/data_path.h"
#include "camera/ipcamera/ipcamera.h"
#include "camera/camera_tour.h"
#include "face/face_observer.h"
#include "face/face_namelist.h"
#include "stitcher/stitcher_observer.h"
#include "stitcher/stitcher.h"
#include "stitcher/stitching.h"

class SmartClass
{
public:
    static SmartClass* instance(){
        static SmartClass sc;
        return &sc;
    }
    void init();
    void run();

    void start(){ bPause = false; }
    void config(std::string code);
    void dummy();
    void pause(){ bPause = true; }
    void shutdown(){ bShutdown = true; }

private:
    SmartClass();
    ~SmartClass();

    std::vector<stitching::Image> vec_stImage;

    IPCamera* pCamera;
    stitching::Stitcher* pStitcher;

    FaceDB* pFaceDB; // for faceObserver
    NameList* pNameList; // for stitcher

    CameraTour* pCameraTour;
    ImgObserver* pFaceObserver;
    ImgObserver* pStitcherObserver;

    std::mutex mtx;

    std::atomic_bool bPause;
    std::atomic_bool bDummy;
    std::atomic_bool bShutdown;
    int pan_cols ;
    int tilt_rows;
};

void smartclass_run();
void smartclass_run_thread();


#endif
