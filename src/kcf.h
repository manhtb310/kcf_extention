#ifndef KCF_HEADER_6565467831231
#define KCF_HEADER_6565467831231

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include "fhog.hpp"

#include "complexmat.hpp"
#ifdef CUFFT
#include "cuda_error_check.hpp"
#include <cuda_runtime.h>
#endif

#include "cnfeat.hpp"
#ifdef FFTW
#include "fft_fftw.h"
#define FFT Fftw
#elif defined(CUFFT)
#include "fft_cufft.h"
#define FFT cuFFT
#else
#include "fft_opencv.h"
#define FFT FftOpencv
#endif
#include "pragmas.h"

class Kcf_Tracker_Private;
struct ThreadCtx;

struct BBox_c
{
    double cx, cy, w, h, a;

    inline cv::Point2d center() const { return cv::Point2d(cx, cy); }

    inline void scale(double factor)
    {
        cx *= factor;
        cy *= factor;
        w  *= factor;
        h  *= factor;
    }

    inline cv::Rect get_rect()
    {
        return cv::Rect(int(cx-w/2.), int(cy-h/2.), int(w), int(h));
    }

};

class KCF_Tracker
{
    friend ThreadCtx;
    friend Kcf_Tracker_Private;
public:
    bool m_debug {false};
    enum class vd {NONE, PATCH, RESPONSE} m_visual_debug {vd::NONE};
    constexpr static bool m_use_scale {true};
    constexpr static bool m_use_color {false};
    constexpr static bool m_use_subpixel_localization {true};
    constexpr static bool m_use_subgrid_scale {true};
    constexpr static bool m_use_subgrid_angle {true};
    constexpr static bool m_use_cnfeat {false};
    constexpr static bool m_use_linearkernel {false};
    const int p_cell_size = 4;            //4 for hog (= bin_size)

    /*
    padding             ... extra area surrounding the target           (1.5)
    kernel_sigma        ... gaussian kernel bandwidth                   (0.5)
    lambda              ... regularization                              (1e-4)
    interp_factor       ... linear interpolation factor for adaptation  (0.02)
    output_sigma_factor ... spatial bandwidth (proportional to target)  (0.1)
    cell_size           ... hog cell size                               (4)
    */
    KCF_Tracker(double padding, double kernel_sigma, double lambda, double interp_factor, double output_sigma_factor, int cell_size);
    KCF_Tracker();
    ~KCF_Tracker();

    // Init/re-init methods
    void init(cv::Mat & img, const cv::Rect & bbox, int fit_size_x = -1, int fit_size_y = -1);
    void setTrackerPose(BBox_c & bbox, cv::Mat & img, int fit_size_x = -1, int fit_size_y = -1);
    void updateTrackerPosition(BBox_c & bbox);

    // frame-to-frame object tracking
    void track(cv::Mat & img);
    BBox_c getBBox();
    double getFilterResponse() const; // Measure of tracking accuracy

private:
    FFT &fft;

    // Initial pose of tracked object in internal image coordinates
    // (scaled by p_downscale_factor if p_resize_image)
    BBox_c p_init_pose;

    // Information to calculate current pose of the tracked object
    cv::Point2d p_current_center;
    double p_current_scale = 1.;
    double p_current_angle = 0.;

    double max_response = -1.;

    bool p_resize_image = false;

    constexpr static double p_downscale_factor = 0.5;
    constexpr static double p_floating_error = 0.0001;

    const double p_padding = 1.5;
    const double p_output_sigma_factor = 0.1;
    double p_output_sigma;
    const double p_kernel_sigma = 0.5;    //def = 0.5
    const double p_lambda = 1e-4;         //regularization in learning step
    const double p_interp_factor = 0.02;  //def = 0.02, linear interpolation factor for adaptation
    cv::Size p_windows_size;              // size of the patch to find the tracked object in
    cv::Size fit_size;                    // size to which rescale the patch for better FFT performance

    constexpr static uint p_num_scales = m_use_scale ? 5 : 1;
    constexpr static double p_scale_step = 1.03;
    double p_min_max_scale[2];
    std::vector<double> p_scales;

    constexpr static uint p_num_angles = 3;
    constexpr static int p_angle_step = 10;
    std::vector<double> p_angles;

    constexpr static int p_num_of_feats = 31 + (m_use_color ? 3 : 0) + (m_use_cnfeat ? 10 : 0);
    cv::Size feature_size;

    std::unique_ptr<Kcf_Tracker_Private> d;

    class Model {
        cv::Size feature_size;
        uint height, width, n_feats;
    public:
        ComplexMat yf {height, width, 1};
        ComplexMat model_alphaf {height, width, 1};
        ComplexMat model_alphaf_num {height, width, 1};
        ComplexMat model_alphaf_den {height, width, 1};
        ComplexMat model_xf {height, width, n_feats};
        ComplexMat xf {height, width, n_feats};
        ComplexMat kf {height, width, 1};

        // Temporary variables for trainig
        MatScaleFeats patch_feats{1, n_feats, feature_size};
        MatScaleFeats temp{1, n_feats, feature_size};
        
        // Type. feature and weight
        std::string kernelType;
        std::string featureType;
        float weight;


        Model(cv::Size feature_size, uint _n_feats)
            : feature_size(feature_size)
            , height(Fft::freq_size(feature_size).height)
            , width(Fft::freq_size(feature_size).width)
            , n_feats(_n_feats) {}
    };
    std::unique_ptr<Model> model;

    std::unique_ptr<Model> filterPool[(int)4];
    std::unique_ptr<Model> multiFilters[(int)4];

    class GaussianCorrelation {
      public:
        GaussianCorrelation(uint num_scales, uint num_feats, cv::Size size)
            : xf_sqr_norm(num_scales)
            , xyf(Fft::freq_size(size), num_feats, num_scales)
            , ifft_res(num_scales, size)
            , k(num_scales, size)
        {}
        void operator()(ComplexMat &result, const ComplexMat &xf, const ComplexMat &yf, double sigma, bool auto_correlation, const KCF_Tracker &kcf, bool gaussian, bool hog);

      private:
        DynMem xf_sqr_norm;
        DynMem yf_sqr_norm{1};
        ComplexMat xyf;
        MatScales ifft_res;
        MatScales k;
    };

    //helping functions
    void scale_track(ThreadCtx &vars, cv::Mat &input_rgb, cv::Mat &input_gray);
    cv::Mat get_subwindow(const cv::Mat &input, int cx, int cy, int size_x, int size_y, double angle) const;
    cv::Mat gaussian_shaped_labels(double sigma, int dim1, int dim2);
    std::unique_ptr<GaussianCorrelation> gaussian_correlation;
    cv::Mat circshift(const cv::Mat &patch, int x_rot, int y_rot) const;
    cv::Mat cosine_window_function(int dim1, int dim2);
    cv::Mat get_features(cv::Mat &input_rgb, cv::Mat &input_gray, cv::Mat *dbg_patch, int cx, int cy, int size_x, int size_y, double scale, double angle) const;
    cv::Point2f sub_pixel_peak(cv::Point &max_loc, cv::Mat &response) const;
    cv::Mat get_features_v2(cv::Mat &input_rgb, cv::Mat &input_gray, cv::Mat *dbg_patch, int cx, int cy, double scale, double angle, cv::Mat &feaGray) const;

    double sub_grid_scale(uint index);
    void resizeImgs(cv::Mat &input_rgb, cv::Mat &input_gray);
    void train(cv::Mat input_rgb, cv::Mat input_gray, double interp_factor);
    double findMaxReponse(uint &max_idx, cv::Point2d &new_location) const;
    double sub_grid_angle(uint max_index);
    void trainV2(MatScaleFeats &feaHog, MatScales &feaGray, double interp_factor);

};

#endif //KCF_HEADER_6565467831231
