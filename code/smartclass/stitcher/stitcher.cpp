#include <iostream>
#include <boost/timer.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include "stitcher.h"
#include "../common/chrono_timer.h"

#define DEBUG_PIC 0
#define DEBUG_LOG 0

namespace stitching
{

static ChronoTimer timer;

Stitcher::Stitcher(
         float work_megapix, float seam_megapix,
        bool try_cuda, float conf_thresh, float match_conf,
         std::string seam_find_type, int blend_type,
         std::string features_type, std::string ba_cost_func, std::string warp_type)
{
    // set megapix
    _work_megapix = work_megapix;
    _seam_megapix = seam_megapix;
    _blend_type = blend_type;

    // create feature finder
    if (features_type == "orb")
    {
        _finder = cv::ORB::create();
    }
#ifdef HAVE_OPENCV_XFEATURES2D
    else if (features_type == "surf")
    {
        _finder = cv::xfeatures2d::SURF::create();
    }
#endif
    else if (features_type == "cuda_surf")
    {
        // cuda surf create in feature_images()
        _finder = nullptr;
    }
    else if (features_type == "sift")
    {
        _finder = cv::SIFT::create();
    }
    else
    {
        std::cout << "Unknown 2D features type: '" << features_type << "'.\n";
        return;
    }

    // create matcher estimater
    _matcher = cv::makePtr<cv::detail::BestOf2NearestMatcher>(try_cuda, match_conf);
    _estimator = cv::makePtr<cv::detail::HomographyBasedEstimator>();

    // BA setting
    if (ba_cost_func == "reproj")
        _adjuster = cv::makePtr<cv::detail::BundleAdjusterReproj>();
    else if (ba_cost_func == "ray")
        _adjuster = cv::makePtr<cv::detail::BundleAdjusterRay>();
    else
    {
        std::cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
        return;
    }
    _conf_thresh = conf_thresh;
    _adjuster->setConfThresh(_conf_thresh);
    std::string ba_refine_mask="xxxxx";
    cv::Mat_<uchar> refine_mask = cv::Mat::zeros(3, 3, CV_8U);
    if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
    if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
    if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
    if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
    if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
    _adjuster->setRefinementMask(refine_mask);

    // create warper
#ifdef HAVE_OPENCV_CUDAWARPING
    if (try_cuda && cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        if (warp_type == "plane")
            _warper_creator = cv::makePtr<cv::PlaneWarperGpu>();
        else if (warp_type == "cylindrical")
            _warper_creator = cv::makePtr<cv::CylindricalWarperGpu>();
        else if (warp_type == "spherical")
            _warper_creator = cv::makePtr<cv::SphericalWarperGpu>();
    }
    else
#endif
    {
        if (warp_type == "plane")
            _warper_creator = cv::makePtr<cv::PlaneWarper>();
        else if (warp_type == "affine")
            _warper_creator = cv::makePtr<cv::AffineWarper>();
        else if (warp_type == "cylindrical")
            _warper_creator = cv::makePtr<cv::CylindricalWarper>();
        else if (warp_type == "spherical")
            _warper_creator = cv::makePtr<cv::SphericalWarper>();
        else if (warp_type == "fisheye")
            _warper_creator = cv::makePtr<cv::FisheyeWarper>();
        else if (warp_type == "stereographic")
            _warper_creator = cv::makePtr<cv::StereographicWarper>();
        else if (warp_type == "compressedPlaneA2B1")
            _warper_creator = cv::makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlaneA1.5B1")
            _warper_creator = cv::makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA2B1")
            _warper_creator = cv::makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA1.5B1")
            _warper_creator = cv::makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniA2B1")
            _warper_creator = cv::makePtr<cv::PaniniWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniA1.5B1")
            _warper_creator = cv::makePtr<cv::PaniniWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniPortraitA2B1")
            _warper_creator = cv::makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniPortraitA1.5B1")
            _warper_creator = cv::makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "mercator")
            _warper_creator = cv::makePtr<cv::MercatorWarper>();
        else if (warp_type == "transverseMercator")
            _warper_creator = cv::makePtr<cv::TransverseMercatorWarper>();
    }
    if (!_warper_creator)
    {
        std::cout << "Can't create the following warper '" << warp_type << "'\n";
        return;
    }

    // create compensator
    _compensator = cv::detail::ExposureCompensator::createDefault(
                cv::detail::ExposureCompensator::GAIN);

    // create seam finder
    if (seam_find_type == "no")
        _seam_finder = cv::makePtr<cv::detail::NoSeamFinder>();
    else if (seam_find_type == "voronoi")
        _seam_finder =cv::makePtr<cv::detail::VoronoiSeamFinder>();
    else if (seam_find_type == "gc_color")
    {
        _seam_finder = cv::makePtr<cv::detail::GraphCutSeamFinder>(
                    cv::detail::GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (seam_find_type == "gc_colorgrad")
    {
        _seam_finder = cv::makePtr<cv::detail::GraphCutSeamFinder>(
                    cv::detail::GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
    else if (seam_find_type == "dp_color")
        _seam_finder = cv::makePtr<cv::detail::DpSeamFinder>(
                    cv::detail::DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")
        _seam_finder = cv::makePtr<cv::detail::DpSeamFinder>(
                    cv::detail::DpSeamFinder::COLOR_GRAD);
    if (!_seam_finder)
    {
        std::cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
        return ;
    }
}// constucter end

// feed pointer of vec_stImage to stitcher
void Stitcher::images_feed(std::vector<stitching::Image>* p_vec_stImage)
{
    _p_vec_stImage = p_vec_stImage;
}

void Stitcher::images_clear()
{
    _p_vec_stImage = nullptr;
}

// extract feature of stImage
void Stitcher::extract_feature(stitching::Image& stImage)
{
    // resize to work scale
    float work_scale;
    if(!_work_scale_set)
    {
        work_scale = std::max(0.25, std::sqrt(_work_megapix * 1e6 / stImage.img_original.size().area()));
        _work_scale = work_scale;
        _work_scale_set = true;
    }
    else
    {
        work_scale = _work_scale;
    }
    cv::Mat work_img;
    cv::resize(stImage.img_original, work_img, cv::Size(), work_scale, work_scale);
    //cv::imwrite(std::to_string(stImage.id)+"_work_img.bmp",work_img);

    // extract features in workscale
    cv::detail::computeImageFeatures(_finder, work_img, stImage.feature);
    stImage.feature.img_idx = stImage.id;

    cv::Mat out;
    cv::drawKeypoints(work_img, stImage.feature.keypoints,  out);
    cv::imwrite(std::string("./debug/points/points_") + std::to_string(stImage.id) +std::string(".jpg"), out);

    // resize to seam scale and save it in img_seam
    float seam_scale;
    if (!_seam_scale_set)
    {
        seam_scale = std::max(0.1, std::sqrt(_seam_megapix * 1e6 / stImage.img_original.size().area()));
        _seam_work_aspect = seam_scale / work_scale;
        _seam_scale = seam_scale;
        _seam_scale_set = true;
    }
    else
    {
        seam_scale = _seam_scale;
    }
    cv::resize(stImage.img_original, stImage.img_seam, cv::Size(), seam_scale, seam_scale);
    //cv::imwrite(std::to_string(stImage.id)+"_seam_img.bmp",stImage.img_seam);
    stImage.img_original.release();
}

// extract features of vec_stImage
void Stitcher::images_feature()
{
    timer.start();
#pragma omp parallel for
    for(size_t i =0; i<(*_p_vec_stImage).size();++i){
        extract_feature((*_p_vec_stImage)[i]);
        std::cout<< "Image #" << i << " find features " <<
                    (*_p_vec_stImage)[i].feature.keypoints.size()<<std::endl;
    }
    timer.end_print("feature");
}

// extract features of vec_stImage with cuda_surf
void Stitcher::images_feature_cuda()
{
    timer.start();

    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    cv::cuda::SURF_CUDA surf_cuda_finder;

    for(size_t i =0; i<(*_p_vec_stImage).size();++i){
        // resize to work scale
        float work_scale;
        if(!_work_scale_set)
        {
            work_scale = std::max(0.25, std::sqrt(_work_megapix * 1e6 / (*_p_vec_stImage)[i].img_original.size().area()));
            _work_scale = work_scale;
            _work_scale_set = true;
        }
        else
        {
            work_scale = _work_scale;
        }
        cv::Mat work_img;
        cv::resize((*_p_vec_stImage)[i].img_original, work_img, cv::Size(), work_scale, work_scale);

        // extract feature with cuda_surf
        cv::cuda::GpuMat imgGPU;
        cv::cuda::GpuMat keypointsGPU;
        cv::cuda::GpuMat descriptorsGPU;

        cv::Mat work_img_gray;
        cv::cvtColor(work_img,work_img_gray,CV_BGR2GRAY);
        imgGPU.upload(work_img_gray);
        CV_Assert(!imgGPU.empty());

        surf_cuda_finder(imgGPU, cv::cuda::GpuMat(), keypointsGPU, descriptorsGPU);
        surf_cuda_finder.downloadKeypoints(keypointsGPU,
                                           (*_p_vec_stImage)[i].feature.keypoints);
        //std::cout<<"row: "<<descriptorsGPU.rows<<"   col:  "<< descriptorsGPU.cols<<std::endl;
        //std::vector<float> vec_descriptors;
        descriptorsGPU.download((*_p_vec_stImage)[i].feature.descriptors);
        //std::cout<<"vec size: "<<vec_descriptors.size()<<std::endl;

        //cv::Mat descriptorsMat(vec_descriptors);
        //descriptorsMat.reshape(0,descriptorsGPU.rows);
        //descriptorsMat.convertTo((*_p_vec_stImage)[i].feature.descriptors,CV_32F);

        // resize to seam scale and save it in img_seam
        float seam_scale;
        if (!_seam_scale_set)
        {
            seam_scale = std::max(0.1, std::sqrt(_seam_megapix * 1e6 / (*_p_vec_stImage)[i].img_original.size().area()));
            _seam_work_aspect = seam_scale / work_scale;
            _seam_scale = seam_scale;
            _seam_scale_set = true;
        }
        else
        {
            seam_scale = _seam_scale;
        }
        cv::resize((*_p_vec_stImage)[i].img_original, (*_p_vec_stImage)[i].img_seam, cv::Size(),
                   seam_scale, seam_scale);
        //print
        std::cout<< "Image #" << i << " find features " <<
                    (*_p_vec_stImage)[i].feature.keypoints.size()<<std::endl;
    }
    timer.end_print("feature");
}

// match two stImage with feature and save result in pairwise_match
void Stitcher::match_image_pair(stitching::Image& stImage_fir, stitching::Image& stImage_sec)
{
    int fir_idx = stImage_fir.id;
    int sec_idx = stImage_sec.id;
    if (stImage_fir.feature.keypoints.size() < 20 || stImage_sec.feature.keypoints.size() < 20)
    {
        std::cout << "don't have enough key points in picture" <<
                     fir_idx << "and" << sec_idx << "\n";
        return;
    }

    std::vector<cv::detail::ImageFeatures> features;
    features.push_back(stImage_fir.feature);
    features.push_back(stImage_sec.feature);
    std::vector<cv::detail::MatchesInfo> pairwise_matches;

    // two kind of operater= match method is different, use vector is better
    (*_matcher)(features, pairwise_matches);

    pairwise_matches[1].src_img_idx = fir_idx;
    pairwise_matches[1].dst_img_idx = sec_idx;
    pairwise_matches[2].src_img_idx = sec_idx;
    pairwise_matches[2].dst_img_idx = fir_idx;

    stImage_fir.pairwise_match[sec_idx] = pairwise_matches[1];//match_info1;
    stImage_sec.pairwise_match[fir_idx] = pairwise_matches[2];//match_info2;
}

// use four kind of match mode to match stImages in vec_stImages
// mode = 0 default                 best performace
// mode = 1 linear                    fastest
// mode = 2 dense                   add up-down match
// mode = 3 very dense          add 2-step up-down match
void Stitcher::Images_match(int mode, int col)
{
    // do not use multithread in this func, it has already use cuda

    timer.start();

    int num_images = (*_p_vec_stImage).size();
    for(int i =0; i<num_images;++i){
        (*_p_vec_stImage)[i].pairwise_match.resize(num_images);
    }

    if(mode==0)// default
    {
        std::vector<cv::detail::ImageFeatures> features(num_images);
        std::vector<cv::detail::MatchesInfo> pairwise_matches(num_images * num_images);

        for(int i =0; i<num_images;++i){
            features[i] = (*_p_vec_stImage)[i].feature;
        }

        (*_matcher)(features,pairwise_matches);

        for(int i =0; i<num_images;++i){
            std::copy(pairwise_matches.begin()+i*num_images,
                      pairwise_matches.begin()+(i+1)*num_images,
                      (*_p_vec_stImage)[i].pairwise_match.begin());
        }
    }

    if(mode == 1)// linear
    {
        for(int i=1;i<num_images;++i)
        {
            match_image_pair((*_p_vec_stImage)[i-1], (*_p_vec_stImage)[i]);
            if(i>=2)
                match_image_pair((*_p_vec_stImage)[i-2], (*_p_vec_stImage)[i]);
        }
    }

    if(mode == 2)// dense
    {
        for(int i=1;i<num_images;++i)
        {
            match_image_pair((*_p_vec_stImage)[i-1], (*_p_vec_stImage)[i]);
            if(i>=2)
                match_image_pair((*_p_vec_stImage)[i-2], (*_p_vec_stImage)[i]);
            if((i>col)&&(i%col!=0)){
                int up_img_idx = i-(i%col)*2-1;
                match_image_pair((*_p_vec_stImage)[up_img_idx],(*_p_vec_stImage)[i]);
            }
        }
    }

    if(mode == 3) // ultra
    {
        for(int i=1;i<num_images;++i)
        {
            match_image_pair((*_p_vec_stImage)[i-1], (*_p_vec_stImage)[i]);
            if(i>=2)
                match_image_pair((*_p_vec_stImage)[i-2], (*_p_vec_stImage)[i]);
            if((i>col)&&(i%col!=0)){
                int up_img_idx = i-(i%col)*2-1;
                match_image_pair((*_p_vec_stImage)[up_img_idx],(*_p_vec_stImage)[i]);
            }
            if(i>=(2*col))
                match_image_pair((*_p_vec_stImage)[i-2*col], (*_p_vec_stImage)[i]);
        }
    }

    // release
    for(unsigned int  i = 0;i<(*_p_vec_stImage).size();++i){
        (*_p_vec_stImage)[i].feature.descriptors.release();
    }

    timer.end_print("match");
}

// filter stImages don't have faces, leave the biggest component of stImages
void Stitcher::images_filter(std::vector<int>& reserve_idx, bool filter_face)
{
    timer.start();

    if ((*_p_vec_stImage).empty()) return;

    //save images idx in vector if it has face
    std::vector<int> have_face_idx;
    if(filter_face)
    {
        for (size_t i = 0; i < (*_p_vec_stImage).size(); i++)
        {
            //pushback idx if it has face
            if ((*_p_vec_stImage)[i].vec_faceinfo.size() > 0)
            {
                (*_p_vec_stImage)[i].have_face = true;
                have_face_idx.push_back(i);
            }
        }
    }else
    {
        // pushback all
        for (size_t i = 0; i < (*_p_vec_stImage).size(); i++)
        {
            if ((*_p_vec_stImage)[i].vec_faceinfo.size() > 0)
            {
                (*_p_vec_stImage)[i].have_face = true;
            }
            have_face_idx.push_back(i);
        }
    }

    // copy features and pairwise_matches to a vector
    std::vector<cv::detail::ImageFeatures> features(have_face_idx.size());
    std::vector<cv::detail::MatchesInfo> pairwise_matches(
                have_face_idx.size() * have_face_idx.size());
    for (size_t i = 0; i < have_face_idx.size(); i++)
    {
        int img_idx = have_face_idx[i];
        features[i] = (*_p_vec_stImage)[img_idx].feature;
        std::vector<cv::detail::MatchesInfo> sub_pairwise_matches;
        for (size_t j = 0; j < have_face_idx.size(); j++)
        {
            sub_pairwise_matches.push_back(
                        (*_p_vec_stImage)[img_idx].pairwise_match[have_face_idx[j]]);
        }
        std::copy(sub_pairwise_matches.begin(), sub_pairwise_matches.end(),
            pairwise_matches.begin() + have_face_idx.size() * i);
    }

    //filter images according to conf_thresh
    std::vector<int> leaveBiggest_idx;
    try
    {
        leaveBiggest_idx = cv::detail::leaveBiggestComponent(features, pairwise_matches, _conf_thresh);
    }
    catch (...)
    {
        std::cout << "leaveBiggestComponent exception\n";
        return;
    }

    // save reserved index in vector reserve_idx
    for (size_t i = 0; i < leaveBiggest_idx.size(); i++)
    {
        reserve_idx.push_back(have_face_idx[leaveBiggest_idx[i]]);
    }

    // find images filtered, and set is_filter to true
    for (size_t i = 0; i < (*_p_vec_stImage).size(); i++)
    {
        if (std::find(reserve_idx.begin(), reserve_idx.end(), i) == reserve_idx.end())
        {
            (*_p_vec_stImage)[i].is_filter = true;
        }
    }

    // remove pairwise_match of filtered images
    for (size_t i = 0; i < (*_p_vec_stImage).size(); i++)
    {
        if ((*_p_vec_stImage)[i].is_filter) continue;
        std::vector<cv::detail::MatchesInfo> sub_pairwise_matches;
        for (size_t j = 0; j < reserve_idx.size(); j++)
        {
            sub_pairwise_matches.push_back((*_p_vec_stImage)[i].pairwise_match[reserve_idx[j]]);
        }
        (*_p_vec_stImage)[i].pairwise_match = sub_pairwise_matches;
    }
    std::cout << "reserved_idx size : " << reserve_idx.size() << std::endl;
    timer.end_print("filter");
}

// estimate camera params and use BA to optimize them
void Stitcher::images_estimate_cameras(std::vector<int>& reserve_idx, int col,
                                  std::vector<cv::detail::CameraParams>& cameras)
{
    //copy features and pairwise_matches of reserved images into vector
    int num_reserve = reserve_idx.size();
    std::vector<cv::detail::ImageFeatures> features(num_reserve);
    std::vector<cv::detail::MatchesInfo> pairwise_matches(num_reserve * num_reserve);
    int image_idx = 0;
    for (size_t i = 0; i < (*_p_vec_stImage).size(); i++)
    {
        if ((*_p_vec_stImage)[i].is_filter) continue;

        features[image_idx] = (*_p_vec_stImage)[i].feature;
        std::copy((*_p_vec_stImage)[i].pairwise_match.begin(), (*_p_vec_stImage)[i].pairwise_match.end(),
            pairwise_matches.begin() + num_reserve * image_idx);
        image_idx++;
    }

    // estimate camera params
    timer.start();
    (*_estimator)(features, pairwise_matches, cameras);
    timer.end_print("estimator");

    // change data type
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        cv::Mat R, t;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].t.convertTo(t, CV_32F);
        cameras[i].R = R;
        cameras[i].t = t;
    }

    // camera params bundle adjust
    timer.start();
    (*_adjuster)(features, pairwise_matches, cameras);
    timer.end_print("bundle adjust");

    // find a middle idx in reserved_idx
    int middle_idx = std::ceil(col/2);
    std::cout << "middle idx : " << middle_idx << std::endl;
    /*while(std::find(reserve_idx.begin(),reserve_idx.end(),middle_idx)==reserve_idx.end())
    {
        //can't find it in reserved_idx
        middle_idx+=col;
        if(middle_idx > (*_p_vec_stImage).size()){// out of range
            middle_idx = reserve_idx[std::ceil(col/2)];
            break;
        }
    }*/

#if DEBUG_LOG
    std::cout << "debug: 1" <<std::endl;
#endif

    // set middle camera as world axis     
     cv::Mat Rwc1 = cameras[middle_idx].R;
     cv::Mat twc1 = cameras[middle_idx].t;

     for (size_t i = 0; i < cameras.size(); i++)
     {
         cv::Mat Rc2 = cameras[i].R;
         cv::Mat tc2 = cameras[i].t;
         cv::Mat new_R(Rc2.rows, Rc2.cols, Rc2.type());
         cv::Mat new_t(tc2.rows, tc2.cols, tc2.type());
         change_world_coord(Rwc1, twc1, Rc2, tc2, new_R, new_t);
         cameras[i].R = new_R;
         cameras[i].t = new_t;
     }
#if DEBUG_LOG
    std::cout << "debug: 2" <<std::endl;
#endif
     for (size_t i = 0; i < cameras.size(); i++)
     {
         (*_p_vec_stImage)[reserve_idx[i]].camera = cameras[i];
     }

     std::cout << "set middle camera as world axis done."<<std::endl;
     // print info
     /*
     for (size_t i = 0; i < cameras.size(); ++i)
     {
         std::cout << "Camera #" << reserve_idx[i] << ":\nK:\n"
                   << cameras[i].K() << "\nR:\n" << cameras[i].R << std::endl;
     }*/
}

// use warper to warp img,mask and face_box
void Stitcher::images_warp(std::vector<cv::detail::CameraParams> &cameras,  cv::Point& pa_top_left)
{
    timer.start();

    // Find median focal length to calcu warped_image_scale
    std::vector<double> focals;
    for (unsigned int i = 0; i < cameras.size(); ++i)
    {
        focals.push_back(cameras[i].focal);
    }
    std::sort(focals.begin(), focals.end());
#if DEBUG_LOG
    std::cout << "debug: 3" <<std::endl;
#endif
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(
                    (focals[focals.size() / 2 - 1] +focals[focals.size() / 2]) /2   );

#if DEBUG_LOG
    std::cout << "debug: Find median focal length to calcu warped_image_scale" << std::endl;
#endif

    // Preapre images masks, the size of mask equals to img_seam
    for (unsigned int i = 0; i < (*_p_vec_stImage).size(); ++i)
    {
        if ((*_p_vec_stImage)[i].is_filter) continue;
        (*_p_vec_stImage)[i].mask.create((*_p_vec_stImage)[i].img_seam.size(), CV_8U);
        (*_p_vec_stImage)[i].mask.setTo(cv::Scalar::all(255));
    }

#if DEBUG_LOG
    std::cout << "debug: Preapre images masks, the size of mask equals to img_seam" << std::endl;
#endif

    // create warper
    _warper = _warper_creator->create(static_cast<float>(
                                          warped_image_scale * _seam_work_aspect));

    // warp
    for (unsigned int i = 0; i < (*_p_vec_stImage).size(); ++i)
    {
        if ((*_p_vec_stImage)[i].is_filter) continue;

        cv::Mat_<float> K;// new K in seamscale, previous K is in workscale
        (*_p_vec_stImage)[i].camera.K().convertTo(K, CV_32F);
        float swa = (float)_seam_work_aspect;
        K(0, 0) *= swa; K(0, 2) *= swa;
        K(1, 1) *= swa; K(1, 2) *= swa;

        (*_p_vec_stImage)[i].corner = _warper->warp((*_p_vec_stImage)[i].img_seam, K,
                                             (*_p_vec_stImage)[i].camera.R,cv::INTER_LINEAR, cv::BORDER_REFLECT,
                                             (*_p_vec_stImage)[i].img_warp);
#if DEBUG_LOG
        std::cout<< "debug: warp img_seam " << i<<std::endl;
#endif

#if DEBUG_PIC
        cv::imwrite("./debug_stitcher/img_warp/"+
                    std::to_string(i)+".jpg", (*_p_vec_stImage)[i].img_warp);
#endif
        (*_p_vec_stImage)[i].size = (*_p_vec_stImage)[i].img_warp.size();

        (*_p_vec_stImage)[i].vec_facebox.resize((*_p_vec_stImage)[i].vec_faceinfo.size());
        (*_p_vec_stImage)[i].vec_warpfacebox.resize((*_p_vec_stImage)[i].vec_faceinfo.size());

        /*warp image point*/
        for (size_t f = 0; f < (*_p_vec_stImage)[i].vec_facebox.size(); f++)
        {
            FaceBox& src_box = (*_p_vec_stImage)[i].vec_facebox[f];

            // copy data from vec_faceinfo to vec_facebox
            src_box.x1 = (*_p_vec_stImage)[i].vec_faceinfo[f].face_detection.bbox[0];
            src_box.y1 = (*_p_vec_stImage)[i].vec_faceinfo[f].face_detection.bbox[1];
            src_box.x2 = (*_p_vec_stImage)[i].vec_faceinfo[f].face_detection.bbox[2];
            src_box.y2 = (*_p_vec_stImage)[i].vec_faceinfo[f].face_detection.bbox[3];
            src_box.id = (*_p_vec_stImage)[i].vec_faceinfo[f].id;
            src_box.score = (*_p_vec_stImage)[i].vec_faceinfo[f].score;

            // warp point
            cv::Point2f warp_lt_point = _warper->warpPoint(
                        cv::Point2f(src_box.x1 * _seam_scale, src_box.y1 * _seam_scale),
                        K, (*_p_vec_stImage)[i].camera.R);
            cv::Point2f wrap_rb_point = _warper->warpPoint(
                        cv::Point2f(src_box.x2 * _seam_scale, src_box.y2 * _seam_scale),
                        K, (*_p_vec_stImage)[i].camera.R);

            // save data
            FaceBox dst_box;
            float x1 = std::min(warp_lt_point.x, wrap_rb_point.x);
            float x2 = std::max(warp_lt_point.x, wrap_rb_point.x);
            float y1 = std::min(warp_lt_point.y, wrap_rb_point.y);
            float y2 = std::max(warp_lt_point.y, wrap_rb_point.y);
            ////////////////////////////////////
            dst_box.x1 = x1; dst_box.y1 = y1;
            dst_box.x2 = x2; dst_box.y2 = y2;
            dst_box.id = src_box.id;
            dst_box.score = src_box.score;
            (*_p_vec_stImage)[i].vec_warpfacebox[f] = dst_box;
        }
#if DEBUG_LOG
        std::cout << "debug: warp point "<< i<<std::endl;
#endif
        _warper->warp((*_p_vec_stImage)[i].mask, K,
            (*_p_vec_stImage)[i].camera.R, cv::INTER_NEAREST,
            cv::BORDER_CONSTANT, (*_p_vec_stImage)[i].mask_warp);
#if DEBUG_LOG
        std::cout <<"debug: warp mask "<< i<<std::endl;
#endif

#if DEBUG_PIC
        cv::imwrite("./debug_stitcher/mask_warp/"+
                    std::to_string(i)+".jpg", (*_p_vec_stImage)[i].mask_warp);
#endif

        (*_p_vec_stImage)[i].mask.release();
        (*_p_vec_stImage)[i].img_seam.release();
    }

    for (size_t i = 0; i < (*_p_vec_stImage).size(); i++)
    {
        cv::Point& top_left = (*_p_vec_stImage)[i].corner;
        pa_top_left.x = std::min(top_left.x, pa_top_left.x);
        pa_top_left.y = std::min(top_left.y, pa_top_left.y);
    }

    timer.end_print("warp images");
}

// exposure compensate
void Stitcher::images_exposure_compensate()
{
    timer.start();

    // prepare data
    std::vector<cv::Point> corners;
    std::vector<cv::Mat> images_warped;
    std::vector<cv::Mat> masks_warped;
    for (size_t i = 0; i < (*_p_vec_stImage).size(); i++)
    {
        if ((*_p_vec_stImage)[i].is_filter) continue;
        corners.push_back((*_p_vec_stImage)[i].corner);
        images_warped.push_back((*_p_vec_stImage)[i].img_warp);
        masks_warped.push_back((*_p_vec_stImage)[i].mask_warp);
    }

    // convert to UMat
    std::vector<cv::UMat> images_warped_u, masks_warped_u;
    images_warped_u.resize(images_warped.size());
    masks_warped_u.resize(masks_warped.size());
    for (size_t i = 0; i < images_warped.size(); i++)
    {
        images_warped_u[i] = images_warped[i].clone().getUMat(cv::ACCESS_RW);
        masks_warped_u[i] = masks_warped[i].clone().getUMat(cv::ACCESS_RW);
    }

    // setting compensator
    _compensator->feed(corners, images_warped_u, masks_warped_u);

    // apply
    int idx = 0;
    for (size_t i = 0; i < (*_p_vec_stImage).size(); i++)
    {
        if ((*_p_vec_stImage)[i].is_filter) continue;
        _compensator->apply(idx, (*_p_vec_stImage)[i].corner, (*_p_vec_stImage)[i].img_warp,
            (*_p_vec_stImage)[i].mask_warp);
#if DEBUG_PIC
        cv::imwrite("./debug_stitcher/img_warp_expo/"+
                    std::to_string(i)+".jpg", (*_p_vec_stImage)[i].img_warp);
#endif
        idx++;
    }

    timer.end_print("exposure compensate");
}

// find seam, result is masks
void Stitcher::images_find_seam()
{
    timer.start();

    for (size_t i = 0; i < (*_p_vec_stImage).size(); i++)
    {
        if ((*_p_vec_stImage)[i].is_filter) continue;
        (*_p_vec_stImage)[i].img_warp.convertTo((*_p_vec_stImage)[i].img_warped_f, CV_32F);
        (*_p_vec_stImage)[i].mask_warp.copyTo((*_p_vec_stImage)[i].mask_seam);
        (*_p_vec_stImage)[i].img_warp.release();
    }

    // copy data to vector
    std::vector<cv::Mat> images_warped_f, masks_seam;
    std::vector<cv::Point> corners;
    for (size_t i = 0; i < (*_p_vec_stImage).size(); i++)
    {
        if ((*_p_vec_stImage)[i].is_filter) continue;
        images_warped_f.push_back((*_p_vec_stImage)[i].img_warped_f);
        masks_seam.push_back((*_p_vec_stImage)[i].mask_seam);
        corners.push_back((*_p_vec_stImage)[i].corner);
    }

    // convert to UMat
    std::vector<cv::UMat> images_warped_fu, masks_seam_u;
    images_warped_fu.resize(images_warped_f.size());
    masks_seam_u.resize(images_warped_f.size());
    for (size_t i = 0; i < images_warped_f.size(); i++)
    {
        images_warped_fu[i] = images_warped_f[i].clone().getUMat(cv::ACCESS_RW);
        masks_seam_u[i] = masks_seam[i].clone().getUMat(cv::ACCESS_RW);
    }

    // find seam
    _seam_finder->find(images_warped_fu, corners, masks_seam_u);

    // save data
    for (size_t i = 0; i < images_warped_f.size(); i++)
    {
        masks_seam[i] = masks_seam_u[i].getMat(cv::ACCESS_RW).clone();
    }
    images_warped_fu.clear();
    masks_seam_u.clear();

    std::vector<int> reserve_index;
    for (size_t i = 0; i < (*_p_vec_stImage).size(); i++)
    {
        if ((*_p_vec_stImage)[i].is_filter) continue;
        reserve_index.push_back(i);
    }
    for (size_t i = 0; i < reserve_index.size(); i++)
    {
        (*_p_vec_stImage)[reserve_index[i]].corner = corners[i];
        (*_p_vec_stImage)[reserve_index[i]].mask_seam = masks_seam[i];
#if DEBUG_PIC
        cv::imwrite("./debug_stitcher/mask_seam/"+
                    std::to_string(reserve_index[i])+".jpg",
                    (*_p_vec_stImage)[reserve_index[i]].mask_seam);
#endif
    }

    timer.end_print("find seam");
}

// leave the most possible box in reserved_face_box, each index represent a person
void Stitcher::images_filter_facebox(std::vector<std::vector<WarpFaceBox>>& reserved_face_box)
{
    timer.start();

    // save all warped face into reserved_face_box,
    // index  of reserved_face_box represent face_id
    for (size_t i = 0; i < (*_p_vec_stImage).size(); i++)
    {
        if ((*_p_vec_stImage)[i].is_filter) continue;
        for (size_t idx = 0; idx < (*_p_vec_stImage)[i].vec_warpfacebox.size(); idx++)
        {

            WarpFaceBox warp_face;
            warp_face.img_id = i;
            warp_face.aft_warp = (*_p_vec_stImage)[i].vec_warpfacebox[idx];
            int face_id = (*_p_vec_stImage)[i].vec_warpfacebox[idx].id;

            cv::Mat& warp_seam_image = (*_p_vec_stImage)[i].mask_seam;
            cv::Point top_left = (*_p_vec_stImage)[i].corner;
            int top = std::max(0, (int)std::ceil(warp_face.aft_warp.y1 - top_left.y));
            int bottom = std::min(warp_seam_image.rows - 1,
                                  (int)std::ceil(warp_face.aft_warp.y2 - top_left.y));
            int left = std::max(0, (int)std::ceil(warp_face.aft_warp.x1 - top_left.x));
            int right = std::min(warp_seam_image.cols - 1,
                                 (int)std::ceil(warp_face.aft_warp.x2 - top_left.x));
            //std::cout << "top" << top << "bottom" << bottom
                      //<< "left" << left << "right" <<right <<std::endl;
            cv::Mat roi1 = warp_seam_image(cv::Range(top, bottom), cv::Range(left, right));

            int nozeros_num = cv::countNonZero(roi1);
            warp_face.nonzero_num = nozeros_num;
            //std::cout << face_id << std::endl;
            reserved_face_box[face_id].push_back(warp_face);
        }
    }

    // save the biggest zero_num and max score in  reserved_face_box[r]
    for (size_t r = 0; r < reserved_face_box.size(); r++)
    {
        // sort every person's face_box with score
        std::sort(reserved_face_box[r].begin(), reserved_face_box[r].end(),
                  [](WarpFaceBox face1, WarpFaceBox face2)
            {
                return face2.aft_warp.score < face1.aft_warp.score;
            });
        if (reserved_face_box[r].empty()) continue;

        // pushback max score
        std::vector<WarpFaceBox> filter_boxes_vec;
        filter_boxes_vec.push_back({ reserved_face_box[r][0] });

        // pushback ,in distance
        WarpFaceBox& warp_face = reserved_face_box[r][0];
        float max_score = warp_face.aft_warp.score;
        float face_width = warp_face.aft_warp.x2 - warp_face.aft_warp.x1;
        float face_height = warp_face.aft_warp.y2 - warp_face.aft_warp.y1;
        float thread = std::max(face_width, face_height) * 1.5;
        for (size_t c = 1; c < reserved_face_box[r].size(); c++)
        {
            if (find_min_distance(filter_boxes_vec, reserved_face_box[r][c]) < thread)
            {
                filter_boxes_vec.push_back(reserved_face_box[r][c]);
            }
        }

        // sort with nonzero_num
        std::sort(filter_boxes_vec.begin(), filter_boxes_vec.end(),
                  [](WarpFaceBox face1, WarpFaceBox face2)
            {
                return face2.nonzero_num < face1.nonzero_num;
            });

        // save result in reserved_face_box
        filter_boxes_vec[0].aft_warp.score = max_score;
        if (max_score > 0.4)
        {
            reserved_face_box[r] = { filter_boxes_vec[0] };// {} assignment
        }
        else
        {
            reserved_face_box[r].clear();
        }
    }

    // clear
    for (size_t i = 0; i < (*_p_vec_stImage).size(); i++)
    {
        if ((*_p_vec_stImage)[i].is_filter) continue;
        (*_p_vec_stImage)[i].vec_warpfacebox.clear();
    }

    timer.end_print("filter facebox");
}

//optim_seam
void Stitcher::images_optim_seam(std::vector<std::vector<WarpFaceBox>>& reserved_face_box)
{
    timer.start();

    for (size_t r = 0; r < reserved_face_box.size(); r++)
    {
        FaceBox box;
        box.x1 = std::numeric_limits<float>::max();
        box.y1 = std::numeric_limits<float>::max();
        box.x2 = -std::numeric_limits<float>::max();
        box.y2 = -std::numeric_limits<float>::max();
        if (reserved_face_box[r].empty()) continue;
        for (size_t c = 0; c < reserved_face_box[r].size(); c++)
        {
            box.x1 = std::min(box.x1, reserved_face_box[r][c].aft_warp.x1);
            box.y1 = std::min(box.y1, reserved_face_box[r][c].aft_warp.y1);
            box.x2 = std::max(box.x2, reserved_face_box[r][c].aft_warp.x2);
            box.y2 = std::max(box.y2, reserved_face_box[r][c].aft_warp.y2);
        }
        int img1_id = reserved_face_box[r][0].img_id;
        cv::Mat& warp_seam_image1 = (*_p_vec_stImage)[img1_id].mask_seam;

        cv::Point top_left1 = (*_p_vec_stImage)[img1_id].corner;
        FaceBox scaleBox = expand_box(box, 1.3);
        scaleBox.id = reserved_face_box[r][0].aft_warp.id;
        scaleBox.score = reserved_face_box[r][0].aft_warp.score;
        (*_p_vec_stImage)[img1_id].vec_warpfacebox.push_back(scaleBox);

        int top = std::max(0, (int)std::ceil(scaleBox.y1 - top_left1.y));
        int bottom = std::min(warp_seam_image1.rows - 1, (int)std::ceil(scaleBox.y2 - top_left1.y));
        int left = std::max(0, (int)std::ceil(scaleBox.x1 - top_left1.x));
        int right = std::min(warp_seam_image1.cols - 1, (int)std::ceil(scaleBox.x2 - top_left1.x));
        /*得到真正的ROI*/
        scaleBox.y1 = top_left1.y + top;
        scaleBox.y2 = top_left1.y + bottom;
        scaleBox.x1 = top_left1.x + left;
        scaleBox.x2 = top_left1.x + right;

        cv::Mat roi1 = warp_seam_image1(cv::Range(top, bottom), cv::Range(left, right));
        int nozeros_num = cv::countNonZero(roi1);
        int rect_area = roi1.cols * roi1.rows;

        if (rect_area == nozeros_num) continue;
        cv::Mat allOneImg = cv::Mat::ones(roi1.rows, roi1.cols, roi1.type()) * 255;

        allOneImg.copyTo(roi1);


        for (size_t c = 1; c < reserved_face_box[r].size(); c++)
        {
            int img2_id = reserved_face_box[r][c].img_id;
            cv::Mat& warp_seam_image2 = (*_p_vec_stImage)[img2_id].mask_seam;
            cv::Point top_left2 = (*_p_vec_stImage)[img2_id].corner;
            top = std::max(0, (int)std::ceil(scaleBox.y1 - top_left2.y));
            bottom = std::min(warp_seam_image2.rows - 1, (int)std::ceil(scaleBox.y2 - top_left2.y));
            left = std::max(0, (int)std::ceil(scaleBox.x1 - top_left2.x));
            right = std::min(warp_seam_image2.cols - 1, (int)std::ceil(scaleBox.x2 - top_left2.x));
            cv::Mat roi2 = warp_seam_image2(cv::Range(top, bottom), cv::Range(left, right));
            cv::Mat allZeroImg = cv::Mat::zeros(roi2.rows, roi2.cols, roi2.type());
            allZeroImg.copyTo(roi2);
        }
        for (size_t c = 0; c < (*_p_vec_stImage).size(); c++)
        {
            if ((*_p_vec_stImage)[c].is_filter) continue;
            int img2_id = c;
            if (img1_id == img2_id) continue;
            cv::Mat& warp_seam_image2 = (*_p_vec_stImage)[img2_id].mask_seam;
            cv::Point top_left2 = (*_p_vec_stImage)[img2_id].corner;
            top = std::max(0, (int)std::ceil(scaleBox.y1 - top_left2.y));
            bottom = std::min(warp_seam_image2.rows - 1, (int)std::ceil(scaleBox.y2 - top_left2.y));
            left = std::max(0, (int)std::ceil(scaleBox.x1 - top_left2.x));
            right = std::min(warp_seam_image2.cols - 1, (int)std::ceil(scaleBox.x2 - top_left2.x));
            if (left >= right || top >= bottom) continue;
            cv::Mat roi2 = warp_seam_image2(cv::Range(top, bottom), cv::Range(left, right));
            cv::Mat allZeroImg = cv::Mat::zeros(roi2.rows, roi2.cols, roi2.type());
            allZeroImg.copyTo(roi2);
        }
    }
    timer.end_print("optim seam");
}

// blend images and copy face_box into panoroma
void Stitcher::images_blend(cv::Point& top_left, cv::Mat& result_pano,
                            std::vector<FaceBox>& stitch_box)
{
    timer.start();

    cv::Ptr<cv::detail::Blender> blender = cv::detail::Blender::createDefault(
                _blend_type, false);
    if (_blend_type == cv::detail::Blender::MULTI_BAND)
    {
        cv::detail::MultiBandBlender* mb = dynamic_cast<cv::detail::MultiBandBlender*>(
                    static_cast<cv::detail::Blender*>(blender));
        mb->setNumBands(3);
        std::cout << "Multi-band blender, number of bands: "
                  << mb->numBands() << std::endl;
    }
    else if (_blend_type == cv::detail::Blender::FEATHER)
    {
        cv::detail::FeatherBlender* fb = dynamic_cast<cv::detail::FeatherBlender*>(
                    static_cast<cv::detail::Blender*>(blender));
        fb->setSharpness(0.01);
        std::cout << "Feather blender, sharpness: " << fb->sharpness() << std::endl;
    }
    std::vector<cv::Point> corners;
    std::vector<cv::Size> sizes;
    for (size_t i = 0; i < (*_p_vec_stImage).size(); i++)
    {
        if ((*_p_vec_stImage)[i].is_filter) continue;
        corners.push_back((*_p_vec_stImage)[i].corner);
        sizes.push_back((*_p_vec_stImage)[i].size);
    }
    blender->prepare(corners, sizes);

    for (size_t i = 0; i < (*_p_vec_stImage).size(); i++)    //遍历所有图像
    {
        if ((*_p_vec_stImage)[i].is_filter) continue;
        cv::Mat image_warped_s;
        (*_p_vec_stImage)[i].img_warped_f.convertTo(image_warped_s, CV_16S);    //改变数据类型
         blender->feed(image_warped_s, (*_p_vec_stImage)[i].mask_seam,
                       (*_p_vec_stImage)[i].corner);    //初始化数据
    }

    cv::Mat result, result_mask;
    //完成融合操作，得到全景图像result和它的掩码result_mask
    blender->blend(result, result_mask);
    //std::cout << result_mask.channels() << std::endl;

    //转换人脸
    stitch_box.resize(0);
    for (size_t i = 0; i < (*_p_vec_stImage).size(); i++)
    {
        for (size_t c = 0; c < (*_p_vec_stImage)[i].vec_warpfacebox.size(); c++)
        {
            FaceBox _stitch_box;
            FaceBox& face_info = (*_p_vec_stImage)[i].vec_warpfacebox[c];
            int x1 = std::max(0, int(face_info.x1 - top_left.x));
            int x2 = std::min(int(face_info.x2 - top_left.x), result.cols - 1);
            int y1 = std::max(int(face_info.y1 - top_left.y), 0);
            int y2 = std::min(int(face_info.y2 - top_left.y), result.rows - 1);
            float score = face_info.score;
            face_info.x1 = x1; face_info.x2 = x2; face_info.y1 = y1; face_info.y2 = y2;

            if (score < 0.4){
                cv::rectangle(result, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
                cv::line(result,cv::Point(x1, y1),cv::Point(x2, y1),cv::Scalar(0,255,255),2);
                face_info.id = NameInfo::m_unkonw_id;
            }

            else if (score >= 0.4 && score < 0.45){
                cv::rectangle(result, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 225, 225), 2);
            }

            else if (score >= 0.45)
            {
                cv::rectangle(result, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
            }

            draw_will_marker(result,cv::Point(x1, y1),cv::Point(x2, y2),face_info.id);

            //保存位置信息
            _stitch_box.x1 = x1;
            _stitch_box.x2 = x2;
            _stitch_box.y1 = y1;
            _stitch_box.y2 = y2;
            _stitch_box.id = face_info.id;
            _stitch_box.score = score;
            stitch_box.push_back(_stitch_box);
        }
    }
    cv::Mat grid_map = generate_gridmap(result.size(), 10);
    cv::Mat inv_mask;
    cv::bitwise_not(result_mask, inv_mask);

    result.convertTo(result_pano, grid_map.type());
    cv::add(grid_map, result_pano, result_pano, inv_mask);

    timer.end_print("blend images");
}

void Stitcher::namelist_feed(NameList *namelist)
{
    if(namelist->empty() || namelist->classcode == _namelist.classcode)
        return;

    _namelist = *namelist;
}

void Stitcher::namelist_clear()
{
    _namelist.clear();
}

bool Stitcher::namelist_empty()
{
    return _namelist.empty();
}

//----------------auxiliary function---------------

void Stitcher::change_world_coord(const cv::Mat& Rw, const cv::Mat& tw,
    const cv::Mat& Rc2, const cv::Mat& tc2,
    cv::Mat& new_R, cv::Mat& new_t)
{
    new_R = Rw.t() * Rc2;
    new_t = Rw.t() * (tc2 - tw);
}

float Stitcher::compute_distance(const FaceBox& bbox1, const FaceBox& bbox2)
{
    float x1_c = (bbox1.x1 + bbox1.x2) / 2;
    float y1_c = (bbox1.y1 + bbox1.y2) / 2;
    float x2_c = (bbox2.x1 + bbox2.x2) / 2;
    float y2_c = (bbox2.y1 + bbox2.y2) / 2;
    return std::sqrt((x2_c - x1_c) * (x2_c - x1_c) + (y2_c - y1_c) * (y2_c - y1_c));
}

float Stitcher::find_min_distance(const std::vector<WarpFaceBox>& bboxes,
                                const WarpFaceBox& bbox)
{
    if (bboxes.empty()) return 0;
    float min_distance = std::numeric_limits<float>::max();
    for (size_t i = 0; i < bboxes.size(); i++)
    {
        float distance = compute_distance(bboxes[i].aft_warp, bbox.aft_warp);
        min_distance = std::min(min_distance, distance);
    }
    return min_distance;
}

FaceBox Stitcher::expand_box(FaceBox& box, float scale)
{
    float x1 = box.x1;
    float x2 = box.x2;
    float y1 = box.y1;
    float y2 = box.y2;
    float roi_height = y2 - y1 + 1;
    float roi_width = x2 - x1 + 1;
    float x_c = (x1 + x2) / 2;
    float y_c = (y1 + y2) / 2;
    roi_height = roi_height * scale;
    roi_width = roi_width * scale;

    FaceBox scaleBox;
    scaleBox.x1 = x_c - roi_width / 2;
    scaleBox.x2 = x_c + roi_width / 2;
    scaleBox.y1 = y_c - roi_height / 2;
    scaleBox.y2 = y_c + roi_height / 2;
    return scaleBox;
}

cv::Mat Stitcher::generate_gridmap(const cv::Size& size, int stride)
{
    cv::Mat frame = cv::Mat::ones(size, CV_8U) * 128;
    cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR); //CV_GRAY2BGR
    int nc = frame.channels();

    int nWidthOfROI = stride;

    for (int j = 0; j < frame.rows; j++)
    {
        uchar* data = frame.ptr<uchar>(j);
        for (int i = 0; i < frame.cols * nc; i += nc)
        {
            if ((i / nc / nWidthOfROI + j / nWidthOfROI) % 2)
            {
                // bgr
                data[i / nc * nc + 0] = 204;
                data[i / nc * nc + 1] = 204;
                data[i / nc * nc + 2] = 204;
            }
        }
    }
    return frame;
}

void Stitcher::draw_will_marker(cv::Mat& dst,
                                cv::Point leftup, cv::Point rightbottom, int id)
{
    if(id == NameInfo::m_unkonw_id){
        //cv::drawMarker(dst,leftup,cv::Scalar(0,0,255),1,40,2);
        cv::line(dst,leftup,cv::Point(rightbottom.x,leftup.y),cv::Scalar(0,0,255),2);
        return;
    }

    if(id > _namelist.length()){
        std::cout<< "draw_will_marker id out of range" << std::endl;
        return;
    }

    if((_namelist.data)[id].m_will == -1){// red
        //cv::drawMarker(dst,leftup,cv::Scalar(0,0,255),1,40,2);
        cv::line(dst,leftup,cv::Point(rightbottom.x,leftup.y),cv::Scalar(20,40,240),2);
        return;
    }

    if((_namelist.data)[id].m_will == 0){// yellow
        //cv::drawMarker(dst,leftup,cv::Scalar(0,255,255),5,40,2);
        cv::line(dst,leftup,cv::Point(rightbottom.x,leftup.y),cv::Scalar(0,255,255),2);
        return;
    }

    if((_namelist.data)[id].m_will == 1){// green
        //cv::drawMarker(dst,leftup,cv::Scalar(255,0,0),4,40,2);
        cv::line(dst,leftup,cv::Point(rightbottom.x,leftup.y),cv::Scalar(20,200,20),2);
        return;
    }
}



}//namespace
