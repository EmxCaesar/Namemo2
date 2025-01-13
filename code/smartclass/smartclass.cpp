#include "smartclass.h"
#include "../Resource.h"
#include <thread>

const char* ipcamera_hostname_prefix = "http://";
const char* ipcamera_hostname_suffix = "/onvif/device_service";
const char* namelist_prefix = "your_namelist_filename";
const char* facedb_prefix = "your_facedb_filename";

SmartClass::SmartClass():
    pCamera(nullptr),
    pStitcher(nullptr),
    pFaceDB(nullptr),
    pNameList(nullptr),
    pCameraTour(nullptr),
    pFaceObserver(nullptr),
    pStitcherObserver(nullptr),
    bPause(true),
    bDummy(true),
    bShutdown(false)
{
    vec_stImage.resize(0);
}

SmartClass::~SmartClass()
{
    delete pFaceDB;
    delete pNameList;
    delete pCamera;
    delete pStitcher;
    delete pCameraTour;
    delete pFaceObserver;
    delete pStitcherObserver;
}

void SmartClass::init()
{
    const char* ipcamera_username = "your_ipcamera_username";
    const char* ipcamera_passwd = "your_ipcamera_passwd";
    const char* ipcamera_hostname = "your_ipcamera_hostname";
    pCamera = new IPCamera(
                ipcamera_username,ipcamera_passwd,ipcamera_hostname,
                1,1,1, CAMERA_CALIB_PATH);

    float pan_start = 0.3;
    float pan_step = 0.1;
    pan_cols = 5;
    float tilt_start = 0.7;
    float tilt_step = -0.2;
    tilt_rows = 3;
    int img_num = pan_cols * tilt_rows;

    vec_stImage.resize(img_num);
    pCameraTour = new CameraTour(pCamera, &vec_stImage,
                              pan_start, pan_step, pan_cols,
                             tilt_start,tilt_step,tilt_rows );    

    pStitcher = new stitching::Stitcher(
                           1.0, 0.5,true,1, 0.6f,"voronoi",
                           cv::detail::Blender::NO,"surf", "ray","cylindrical");

    pFaceObserver =new FaceObserver();
    pStitcherObserver = new stitching::StitcherObserver(pStitcher);
    pCameraTour->attachObserver(pFaceObserver);
    pCameraTour->attachObserver(pStitcherObserver);

    pNameList = new NameList();
    pFaceDB = new FaceDB();
}


void SmartClass::run()
{
    int round_count = 0;
    while(true)
    {
        if(bShutdown){
            break;
        }

        if(bPause || bDummy){
            usleep(500000);
        }else{
            round_count++;

            pStitcher->namelist_feed(pNameList);
            pFaceObserver->setParam((void*)(pFaceDB));

            pCameraTour->run(round_count);

            cv::Mat pano;
            std::vector<FaceBox> stitch_box;
            std::cout << "pan cols : " << pan_cols << std::endl;
            stitching::stitching(vec_stImage, *(pStitcher), pano, stitch_box,
                                 pan_cols, pFaceDB->person );

            std::cout << "stitch_box length: "<< stitch_box.size()<<std::endl;
            cv::imwrite("pano"+ std::to_string(round_count)+".jpg",pano);

            Resource* src = Resource::instance();
            src->lock();
            src->idx = round_count;
            src->pano = pano;
            std::vector<int> facedata;
            for(unsigned int i=0;i<stitch_box.size();++i)
            {
                facedata.push_back(static_cast<int>(stitch_box[i].x1));
                facedata.push_back(static_cast<int>(stitch_box[i].y1));
                facedata.push_back(static_cast<int>(stitch_box[i].x2));
                facedata.push_back(static_cast<int>(stitch_box[i].y2));
                facedata.push_back(stitch_box[i].id);
                facedata.push_back(static_cast<int>(stitch_box[i].score*100));
            }
            src->facedata = facedata;
            src->unlock();

            sleep(30);
        }
    }
}


void SmartClass::config(std::string code)
{
    std::string namelist_path = std::string(namelist_prefix) +
            std::string("/namelist_") + code + std::string(".txt");
    std::string facedb_path = std::string(facedb_prefix) +
            std::string("/facedb_") + code + std::string(".dat");

    pFaceDB->clear();
    pNameList->clear();
    assert(pFaceDB->empty());
    assert(pNameList->empty());
    Resource::instance()->reset();

    pNameList->load(namelist_path, code);
    pFaceDB->load(facedb_path,code , pNameList->length(), FACE_DATABASE_POSE_NUM);
    //pStitcher->namelist_feed(pNameList);
    //pFaceObserver->setParam((void*)pFaceDB);

    bDummy = false;
    bPause = true;
}

void SmartClass::dummy()
{
    bDummy = true;
    bPause = true;
}

void smartclass_run()
{
    SmartClass* sc  = SmartClass::instance();
    sc->run();
}

void smartclass_run_thread()
{
    std::thread(smartclass_run).detach();
}
