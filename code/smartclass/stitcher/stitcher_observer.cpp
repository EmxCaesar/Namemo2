#include "stitcher_observer.h"

namespace stitching
{

StitcherObserver::StitcherObserver(Stitcher* stitcher)
    :m_stitcher(stitcher)
{

}

void StitcherObserver::process(Image *stImage)
{
    //std::cout << "StitcherObserver process" << std::endl;
    m_mutex.lock();
    m_stitcher->extract_feature(*stImage);
    m_mutex.unlock();
    //std::cout<< "Image #" << stImage->id << " find features " <<
               // (*stImage).feature.keypoints.size()<<std::endl;
}

}
