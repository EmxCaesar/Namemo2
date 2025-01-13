#ifndef _STITCHER_H_
#define _STITCHER_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <mutex>
#include <fstream>
#include "../common/image_utils.h"
#include "../face/face_namelist.h"

namespace stitching
{

class Stitcher
{
private:
    cv::Ptr<cv::Feature2D> _finder;
    cv::Ptr<cv::detail::BestOf2NearestMatcher> _matcher;
    cv::Ptr<cv::detail::HomographyBasedEstimator> _estimator;
    cv::Ptr<cv::detail::BundleAdjusterBase> _adjuster;
    cv::Ptr<cv::WarperCreator> _warper_creator;
    cv::Ptr<cv::detail::RotationWarper> _warper;
    cv::Ptr<cv::detail::SeamFinder> _seam_finder;
    cv::Ptr<cv::detail::ExposureCompensator> _compensator;
    bool _work_scale_set=false;
    bool _seam_scale_set=false;

    std::vector<stitching::Image>* _p_vec_stImage;
    //NameInfo* _p_namelist;
    NameList _namelist;
    //int _namelist_length;

    void change_world_coord(const cv::Mat& Rw, const cv::Mat& tw,
        const cv::Mat& Rc2, const cv::Mat& tc2,
        cv::Mat& new_R, cv::Mat& new_t);
    void match_image_pair(stitching::Image& stImage_fir, stitching::Image& stImage_sec);
    float compute_distance(const FaceBox& bbox1, const FaceBox& bbox2);
    float find_min_distance(const std::vector<WarpFaceBox>& bboxes,
                                    const WarpFaceBox& bbox);
    FaceBox expand_box(FaceBox& box, float scale);
    cv::Mat generate_gridmap(const cv::Size& size, int stride);
    void draw_will_marker(cv::Mat& dst, cv::Point leftup, cv::Point rightbottom,int id);

public:
    float _work_scale;
    float _seam_scale;
    float _work_megapix;
    float _seam_megapix;
    float _seam_work_aspect;
    float _conf_thresh;
    int _blend_type;

public:
    Stitcher(float work_megapix = 0.6, float seam_megapix = 0.3,
            bool try_gpu=true, float conf_thresh = 1, float match_conf=0.3f,
           std::string seam_find_type="voronoi", int blend_type=cv::detail::Blender::MULTI_BAND,
            std::string features_type="orb", std::string ba_cost_func="ray",
             std::string warp_type="cylindrical"
            );

    void extract_feature(stitching::Image& stImage);

    void images_feed(std::vector<stitching::Image>* p_vec_stImage);
    void images_clear();

    void images_feature();
    void images_feature_cuda();
    void Images_match(int mode=0, int col = 0);
    void images_filter(std::vector<int>& reserve_idx, bool filter_face = false);
    void images_estimate_cameras(std::vector<int>& reserve_idx, int col,
                            std::vector<cv::detail::CameraParams> &cameras);
    void images_warp(std::vector<cv::detail::CameraParams> &cameras, cv::Point &pa_top_left);
    void images_exposure_compensate();
    void images_find_seam();
    void images_filter_facebox(std::vector<std::vector<WarpFaceBox> > &reserved_face_box);
    void images_optim_seam(std::vector<std::vector<WarpFaceBox>>& reserved_face_box);
    void images_blend(cv::Point& top_left, cv::Mat& result_pano,
                      std::vector<FaceBox>& stitch_box);

    void namelist_feed(NameList* namelist);
    void namelist_clear();
    bool namelist_empty();
};

}//namespace

#endif
