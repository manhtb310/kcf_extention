
#include "fft.h"
#include <cassert>
#include "debug.h"

void Fft::init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales)
{
    m_width = width;
    m_height = height;
    m_num_of_feats = num_of_feats;
#ifdef BIG_BATCH
    m_num_of_scales = num_of_scales;
#else
    (void)num_of_scales;
#endif
}

void Fft::set_window(const MatDynMem &window)
{
    assert(window.dims == 2);
    assert(window.size().width == int(m_width));
    assert(window.size().height == int(m_height));
    (void)window;
}

void Fft::forward(const MatScales &real_input, ComplexMat &complex_result)
{
    TRACE("");
    DEBUG_PRINT(real_input);
    assert(real_input.dims == 3);
#ifdef BIG_BATCH
    assert(real_input.size[0] == 1 || real_input.size[0] == int(m_num_of_scales));
#else
    assert(real_input.size[0] == 1);
#endif
    assert(real_input.size[1] == int(m_height));
    assert(real_input.size[2] == int(m_width));

    assert(int(complex_result.cols) == freq_size(cv::Size(m_width, m_height)).width);
    assert(int(complex_result.rows) == freq_size(cv::Size(m_width, m_height)).height);
    assert(complex_result.channels() == uint(real_input.size[0]));

    (void)real_input;
    (void)complex_result;
}

void Fft::forward_window(MatScaleFeats &patch_feats, ComplexMat &complex_result, MatScaleFeats &tmp)
{
        assert(patch_feats.dims == 4);
#ifdef BIG_BATCH
        assert(patch_feats.size[0] == 1 || patch_feats.size[0] ==  int(m_num_of_scales));
#else
        assert(patch_feats.size[0] == 1);
#endif
        assert(patch_feats.size[1] == int(m_num_of_feats));
        assert(patch_feats.size[2] == int(m_height));
        assert(patch_feats.size[3] == int(m_width));

        assert(tmp.dims == patch_feats.dims);
        assert(tmp.size[0] == patch_feats.size[0]);
        assert(tmp.size[1] == patch_feats.size[1]);
        assert(tmp.size[2] == patch_feats.size[2]);
        assert(tmp.size[3] == patch_feats.size[3]);

        assert(int(complex_result.cols) == freq_size(cv::Size(m_width, m_height)).width);
        assert(int(complex_result.rows) == freq_size(cv::Size(m_width, m_height)).height);
        assert(complex_result.channels() == uint(patch_feats.size[0] * patch_feats.size[1]));

        (void)patch_feats;
        (void)complex_result;
        (void)tmp;
}

void Fft::inverse(ComplexMat &complex_input, MatScales &real_result)
{
    TRACE("");
    DEBUG_PRINT(complex_input);
    assert(real_result.dims == 3);
#ifdef BIG_BATCH
        assert(real_result.size[0] == 1 || real_result.size[0] ==  int(m_num_of_scales));
#else
        assert(real_result.size[0] == 1);
#endif
    assert(real_result.size[1] == int(m_height));
    assert(real_result.size[2] == int(m_width));

    assert(int(complex_input.cols) == freq_size(cv::Size(m_width, m_height)).width);
    assert(int(complex_input.rows) == freq_size(cv::Size(m_width, m_height)).height);
    assert(complex_input.channels() == uint(real_result.size[0]));

    (void)complex_input;
    (void)real_result;
}
