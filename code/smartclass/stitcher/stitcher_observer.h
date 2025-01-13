#ifndef _STITCHER_OBSERVER_H_
#define _STITCHER_OBSERVER_H_

#include "../common/image_observer.h"
#include "stitcher.h"
#include "../common/image_utils.h"

namespace stitching
{

class StitcherObserver:public ImgObserver
{
private:
    std::mutex m_mutex;
    Stitcher* m_stitcher;

public:
    StitcherObserver(Stitcher* stitcher);
    void process(stitching::Image* stImage) override;
};

}
#endif
