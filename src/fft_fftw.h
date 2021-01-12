#ifndef FFT_FFTW_H
#define FFT_FFTW_H

#include "fft.h"

#ifndef CUFFTW
  #include <fftw3.h>
#else
  #include <cufftw.h>
#endif //CUFFTW

class Fftw : public Fft
{
  public:
    Fftw();
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales);
    void set_window(const MatDynMem &window);
    void forward(const MatScales &real_input, ComplexMat &complex_result);
    void forward_window(MatScaleFeats &patch_feats_in, ComplexMat &complex_result, MatScaleFeats &tmp);
    void inverse(ComplexMat &complex_input, MatScales &real_result);
    ~Fftw();

protected:
    fftwf_plan create_plan_fwd(uint howmany) const;
    fftwf_plan create_plan_inv(uint howmany) const;

private:
    cv::Mat m_window;
    fftwf_plan plan_f = 0, plan_fw = 0, plan_i_1ch = 0;
#ifdef BIG_BATCH
    fftwf_plan plan_f_all_scales = 0, plan_fw_all_scales = 0, plan_i_all_scales = 0;
#endif
};

#endif // FFT_FFTW_H
