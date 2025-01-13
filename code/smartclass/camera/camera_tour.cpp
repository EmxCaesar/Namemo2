#include "camera_tour.h"
#include "pthread.h"
#include <thread>
#include <functional>

#define SAVE_PIC 0

CameraTour::CameraTour(CameraBase* pCamera,
                           std::vector<stitching::Image> *vecMatImg,
                           float pan_start, float pan_step, int pan_cols,
                           float tilt_start, float tilt_step, int tilt_rows):
    m_pCamera(pCamera),
    m_pVecStImage(vecMatImg),
    m_pan_start(pan_start),
    m_pan_step(pan_step),
    m_pan_cols(pan_cols),
    m_tilt_start(tilt_start),
    m_tilt_step(tilt_step),
    m_tilt_rows(tilt_rows)
{
    makeTourTable();
}

void CameraTour::makeTourTable()
{
    int tour_points_size = m_pan_cols * m_tilt_rows;
    std::vector<std::pair<float,float>> tour_table(tour_points_size);
    for(int i=0;i<tour_points_size;++i)
    {
        int row = std::floor(i / m_pan_cols);
        int col = i % m_pan_cols;

        float pan;
        float tilt;
        //pan
        if(row%2 == 0){
            pan = m_pan_start + col*m_pan_step;
        }else{
            pan = m_pan_start + (m_pan_cols-1 - col)*m_pan_step;
        }
        //tilt
        tilt = m_tilt_start + row*m_tilt_step;
        std::pair<float,float> pt_pair(pan, tilt);
        tour_table[i] = pt_pair;
        std::cout << "index: "<< i <<" pan: "<<pan<<" tilt: "<<tilt<<std::endl;
    }
    m_tourTable = tour_table;
}

void CameraTour::run(int round_number)
{
    size_t observer_size = m_vecObserver.size();
    size_t image_size = m_pVecStImage->size();
    size_t thread_size = observer_size*image_size;
    std::vector<std::thread*> thread_arr(thread_size);

    for(unsigned int i = 0;i<m_pVecStImage->size();i++)
    {
        // move and capture
        m_pCamera->PTZMove(m_tourTable[i].first,m_tourTable[i].second,1);
        cv::Mat img;
        m_pCamera->capture(img, i );

        //save
        (*m_pVecStImage)[i] = stitching::Image(img, i, round_number);
#if SAVE_PIC
        std::string img_name = std::string("./download/img_")+
                std::to_string(i)+std::string(".jpg");
        cv::imwrite(img_name,img);
#endif

        //inform
        for(size_t j =0;j<observer_size;j++)
        {
            thread_arr[j*image_size+i] = new std::thread(&ImgObserver::process,
                                            m_vecObserver[j], &(*m_pVecStImage)[i]);
        }
    }

    for(unsigned int i=0;i<thread_arr.size();++i){
        thread_arr[i]->join();
        delete thread_arr[i];
    }
}

void CameraTour::attachObserver(ImgObserver* observer)
{
    m_vecObserver.push_back(observer);
}

void CameraTour::detachObserver()
{
    m_vecObserver.resize(0);
}

