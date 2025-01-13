#include "face_align.h"

//dst 5 point for 112x112
static float default5Points_arr[5][2] = {
        {30.2946f+8.0f, 51.6963f},
        {65.5318f+8.0f, 51.5014f},
        {48.0252f+8.0f, 71.7366f},
        {33.5493f+8.0f, 92.3655f},
        {62.7299f+8.0f, 92.2041f}
};
static cv::Mat default5Points(5,2,CV_32FC1);

//src 5 point
static float detect5Points_arr[5][2];
static cv::Mat detect5Points(5,2,CV_32FC1);


//cpp implement of similarTransform
namespace FaceAlign{
        cv::Mat meanAxis0(const cv::Mat &src)
        {
                int num = src.rows;
                int dim = src.cols;

                // x1 y1
                // x2 y2

                cv::Mat output(1,dim,CV_32F);
                for(int i = 0 ; i <  dim; i ++)
                {
                    float sum = 0 ;
                    for(int j = 0 ; j < num ; j++)
                    {
                        sum+=src.at<float>(j,i);
                    }
                    output.at<float>(0,i) = sum/num;
                }

                return output;
        }

        cv::Mat elementwiseMinus(const cv::Mat &A,const cv::Mat &B)
        {
                cv::Mat output(A.rows,A.cols,A.type());

                assert(B.cols == A.cols);
                if(B.cols == A.cols)
                {
                    for(int i = 0 ; i <  A.rows; i ++)
                    {
                        for(int j = 0 ; j < B.cols; j++)
                        {
                            output.at<float>(i,j) = A.at<float>(i,j) - B.at<float>(0,j);
                        }
                    }
                }
                return output;
        }

        cv::Mat varAxis0(const cv::Mat &src)
        {
        cv::Mat temp_ = elementwiseMinus(src,meanAxis0(src));
                cv::multiply(temp_ ,temp_ ,temp_ );
                return meanAxis0(temp_);

        }

        int MatrixRank(cv::Mat M)
        {
        cv::Mat w, u, vt;
        cv::SVD::compute(M, w, u, vt);
        cv::Mat1b nonZeroSingularValues = w > 0.0001;
                int rank = countNonZero(nonZeroSingularValues);
                return rank;
        }

        //    References
        //    ----------
        //    .. [1] "Least-squares estimation of transformation parameters between two
        //    point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
        //
        //    """
        //
        //    Anthor:Jack Yu

        cv::Mat similarTransform(cv::Mat src,cv::Mat dst) {
                int num = src.rows;
                int dim = src.cols;
                cv::Mat src_mean = meanAxis0(src);
                cv::Mat dst_mean = meanAxis0(dst);
                cv::Mat src_demean = elementwiseMinus(src, src_mean);
                cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);
                cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
                cv::Mat d(dim, 1, CV_32F);
                d.setTo(1.0f);
                if (cv::determinant(A) < 0) {
                    d.at<float>(dim - 1, 0) = -1;

                }
                cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
                cv::Mat U, S, V;
                cv::SVD::compute(A, S,U, V);

                // the SVD function in opencv differ from scipy .
                int rank = MatrixRank(A);
                if (rank == 0) {
                    assert(rank == 0);

                } else if (rank == dim - 1) {
                    if (cv::determinant(U) * cv::determinant(V) > 0) {
                        T.rowRange(0, dim).colRange(0, dim) = U * V;
                    } else {
                        int s = d.at<float>(dim - 1, 0) = -1;
                        d.at<float>(dim - 1, 0) = -1;

                        T.rowRange(0, dim).colRange(0, dim) = U * V;
                        cv::Mat diag_ = cv::Mat::diag(d);
                        cv::Mat twp = diag_*V; //np.dot(np.diag(d), V.T)
                cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
                cv::Mat C = B.diag(0);
                        T.rowRange(0, dim).colRange(0, dim) = U* twp;
                        d.at<float>(dim - 1, 0) = s;
                    }
                }
                else{
                    cv::Mat diag_ = cv::Mat::diag(d);
                    cv::Mat twp = diag_*V.t(); //np.dot(np.diag(d), V.T)
                    cv::Mat res = U* twp; // U
                    T.rowRange(0, dim).colRange(0, dim) = -U.t()* twp;
                }
                cv::Mat var_ = varAxis0(src_demean);
                float val = cv::sum(var_).val[0];
                cv::Mat res;
                cv::multiply(d,S,res);
                float scale =  1.0/val*cv::sum(res).val[0];
                T.rowRange(0, dim).colRange(0, dim) = - T.rowRange(0, dim).colRange(0, dim).t();
                cv::Mat  temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
                cv::Mat  temp2 = src_mean.t(); //src_mean.T
                cv::Mat  temp3 = temp1*temp2; // np.dot(T[:dim, :dim], src_mean.T)
                cv::Mat temp4 = scale*temp3;
                T.rowRange(0, dim).colRange(dim, dim+1)=  -(temp4 - dst_mean.t()) ;
                T.rowRange(0, dim).colRange(0, dim) *= scale;
                return T;
        }

}//namespace end

void face_align(cv::Mat& src, cv::Mat& aligned, FaceDetection& face_detection)
{
    //aim 5 points init
    memcpy(default5Points.data, default5Points_arr, 2 * 5 * sizeof(float));

    //padding
    int x_width = face_detection.bbox[2]-face_detection.bbox[0];
    int y_height = face_detection.bbox[3]-face_detection.bbox[1];
    int x_padding = x_width*0.8;
    int y_padding = y_height*0.8;
    int x1_new = std::max(face_detection.bbox[0]-x_padding, float(0));
    int y1_new = std::max(face_detection.bbox[1]-y_padding, float(0));
    int x2_new = std::min(face_detection.bbox[2]+x_padding, float(src.cols));
    int y2_new = std::min(face_detection.bbox[3]+y_padding, float(src.rows));
    cv::Mat rectImg = src(cv::Rect(x1_new, y1_new,
                                   x2_new-x1_new,
                                   y2_new-y1_new)).clone();
    if(src.total()==0)
        printf("warning: face_align rectImg is empty");

    //src 5 point
    for(int j=0;j<5;++j){
        detect5Points_arr[j][0] = face_detection.landmark[2*j] - x1_new;
        detect5Points_arr[j][1] = face_detection.landmark[2*j+1] - y1_new;
    }
    memcpy(detect5Points.data, detect5Points_arr, 2 * 5 * sizeof(float));

    //face align
    cv::Mat M = FaceAlign::similarTransform(detect5Points, default5Points);
    //std::cout<< "trans :"<< M <<std::endl;
    cv::warpPerspective(rectImg, aligned, M, cv::Size(112, 112));

}

void face_env_align(cv::Mat& src, cv::Mat& aligned, float bbox[4],
  float offset_x, float offset_y, float ratio)
{
    float scaled_ratio = ratio;
    float reduced_ratio = 0.9;

  // get bbox
  float x1 = bbox[0];
  float y1 = bbox[1];
  float x2 = bbox[2];
  float y2 = bbox[3];

  // length of the square roi area
  float width = x2 - x1;
  float height = y2 - y1;
  float length = std::max(width, height);
  length *= scaled_ratio;
  while(length > src.rows || length > src.cols) {
      length*=reduced_ratio;
      std::cout << "ratio is too big, reduce "<<reduced_ratio << std::endl;
  }

  // anchor and leftup point of the roi
  float anchor_x = (x1 + x2) / 2 + offset_x * width / 2;
  float anchor_y = (y1 + y2) / 2 + offset_y * height / 2;
  float leftup_x = anchor_x - length / 2;
  float leftup_y = anchor_y - length / 2;

  if (leftup_x <= 0)
    leftup_x = 0;
  if (leftup_y <= 0)
    leftup_y = 0;
  if (leftup_x + length >= src.cols)
    leftup_x = src.cols - length;
  if (leftup_y + length >= src.rows)
    leftup_y = src.rows - length;

  // get roi and resize
  cv::Rect rect_roi(leftup_x, leftup_y, length, length);
  cv::Mat square_roi = src(rect_roi).clone();

  cv::resize(square_roi, aligned, cv::Size(112, 112));
}
