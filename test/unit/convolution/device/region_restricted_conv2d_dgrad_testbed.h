#pragma once

#include <vector>

#include <fstream>
#include <iostream>
#include <sstream>

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/tensor_view.h"
#include "cutlass/util/host_reorder.h"

#include "testbed.h"

namespace test {
namespace convolution {
namespace device {

template <typename Convolution>
struct RRConvDgradTestbed {
    using ElementAccumulator = typename Convolution::ElementAccumulator;
    using ElementCompute = typename Convolution::ConvolutionKernel::Epilogue::
            OutputOp::ElementCompute;

    /// Initialization
    cutlass::Distribution::Kind init_diff;
    cutlass::Distribution::Kind init_filter;
    cutlass::Distribution::Kind init_mask_input;
    cutlass::Distribution::Kind init_mask_output;
    cutlass::Distribution::Kind init_bias;
    cutlass::Distribution::Kind init_z;
    uint64_t seed;

    cutlass::HostTensor<typename Convolution::ElementSrc,
                        typename Convolution::LayoutSrc>
            tensor_diff;
    cutlass::HostTensor<typename Convolution::ElementFilter,
                        typename Convolution::LayoutFilter>
            tensor_filter;
    cutlass::HostTensor<typename Convolution::ElementMaskInput,
                        typename Convolution::LayoutMaskInput>
            tensor_mask_input;
    cutlass::HostTensor<typename Convolution::ElementMaskOutput,
                        typename Convolution::LayoutMaskOutput>
            tensor_mask_output;
    cutlass::HostTensor<typename Convolution::ElementDst,
                        typename Convolution::LayoutDst>
            tensor_z;
    cutlass::HostTensor<typename Convolution::ElementDst,
                        typename Convolution::LayoutDst>
            tensor_grad;
    cutlass::HostTensor<typename Convolution::ElementBias,
                        typename Convolution::LayoutBias>
            tensor_bias;
    cutlass::HostTensor<typename Convolution::ElementDst,
                        typename Convolution::LayoutDst>
            reference_grad;
    //
    // Methods
    //
    RRConvDgradTestbed(cutlass::Distribution::Kind init_diff_ =
                               cutlass::Distribution::Constant,
                       cutlass::Distribution::Kind init_filter_ =
                               cutlass::Distribution::Constant,
                       cutlass::Distribution::Kind init_mask_input_ =
                               cutlass::Distribution::Constant,
                       cutlass::Distribution::Kind init_mask_output_ =
                               cutlass::Distribution::Constant,
                       cutlass::Distribution::Kind init_bias_ =
                               cutlass::Distribution::Constant,
                       cutlass::Distribution::Kind init_z_ =
                               cutlass::Distribution::Constant,
                       uint64_t seed_ = 2080)
            : init_diff(init_diff_),
              init_filter(init_filter_),
              init_mask_input(init_mask_input_),
              init_mask_output(init_mask_output_),
              init_bias(init_bias_),
              init_z(init_z_),
              seed(seed_) {}

    /// Helper to initialize a tensor view
    template <typename Element, typename Layout>
    bool initialize_tensor(cutlass::TensorView<Element, Layout> view,
                           cutlass::Distribution::Kind dist_kind,
                           uint64_t seed) {
        if (dist_kind == cutlass::Distribution::Uniform) {
            double scope_max = 3, scope_min = 1;

            cutlass::reference::host::TensorFillRandomUniform(
                    view, seed, scope_max, scope_min, 0);
        } else if (dist_kind == cutlass::Distribution::Identity) {
            cutlass::reference::host::TensorFillIdentity(view);
        } else if (dist_kind == cutlass::Distribution::Gaussian) {
            cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0,
                                                               0.5);
        } else if (dist_kind == cutlass::Distribution::Sequential) {
            cutlass::reference::host::BlockFillSequential(view.data(),
                                                          view.capacity());
        } else if (dist_kind == cutlass::Distribution::Constant) {
            cutlass::reference::host::TensorFill(view, Element(1));
        } else {
            // TODO: Implement the rest
            EXPECT_TRUE(false) << "Not implemented";
            return false;
        }

        return true;
    }

    /// Initializes data structures
    void initialize(cutlass::conv::Conv2dProblemSize conv_param,
                    bool verify = true) {
        if_constexpr<is_depthwise_convolution<Convolution>()>([&](auto _) {
            auto&& conv_param_ = _(conv_param);
            ASSERT_EQ(conv_param_.K, conv_param_.C);
        });

        //
        // Allocate the CONVOLUTION workspace
        //

        tensor_diff.resize(typename Convolution::LayoutSrc::TensorCoord{
                conv_param.N, conv_param.P, conv_param.Q, conv_param.K});
        tensor_grad.resize(typename Convolution::LayoutDst::TensorCoord{
                conv_param.N, conv_param.H, conv_param.W, conv_param.C});
        reference_grad.resize(typename Convolution::LayoutDst::TensorCoord{
                conv_param.N, conv_param.H, conv_param.W, conv_param.C});
        tensor_mask_input.resize(typename Convolution::LayoutDst::TensorCoord{
                conv_param.N, conv_param.H, conv_param.W, 1});
        tensor_mask_output.resize(typename Convolution::LayoutSrc::TensorCoord{
                conv_param.N, conv_param.P, conv_param.Q, 1});
        tensor_bias.resize(typename Convolution::LayoutBias::TensorCoord{
                1, 1, 1, conv_param.C});
        tensor_z.resize(typename Convolution::LayoutDst::TensorCoord{
                conv_param.N, conv_param.H, conv_param.W, conv_param.C});
        if_constexpr<is_depthwise_convolution<Convolution>()>(
                [&](auto _) {
                    auto&& conv_param_ = _(conv_param);
                    ASSERT_EQ(conv_param_.K, conv_param_.C);
                    tensor_filter.resize(
                            typename Convolution::LayoutFilter::TensorCoord{
                                    conv_param_.K, conv_param_.R, conv_param_.S,
                                    1});
                },
                [&](auto _) {
                    auto&& conv_param_ = _(conv_param);
                    tensor_filter.resize(
                            typename Convolution::LayoutFilter::TensorCoord{
                                    conv_param_.K, conv_param_.R, conv_param_.S,
                                    conv_param_.C});
                });

        EXPECT_TRUE(initialize_tensor(tensor_filter.host_view(), init_filter,
                                      seed + 2019));
        EXPECT_TRUE(initialize_tensor(tensor_diff.host_view(), init_diff,
                                      seed + 2018));
        EXPECT_TRUE(initialize_tensor(tensor_mask_input.host_view(),
                                      init_mask_input, seed + 2019));
        EXPECT_TRUE(initialize_tensor(tensor_mask_output.host_view(),
                                      init_mask_output, seed + 2018));
        EXPECT_TRUE(initialize_tensor(tensor_bias.host_view(), init_bias,
                                      seed + 2017));
        EXPECT_TRUE(
                initialize_tensor(tensor_z.host_view(), init_z, seed + 2016));
        // cutlass::reference::host::TensorCopy(reference_grad.host_view(),
        //                                      tensor_z.host_view());
        tensor_diff.sync_device();
        tensor_filter.sync_device();
        tensor_mask_input.sync_device();
        tensor_mask_output.sync_device();
        tensor_bias.sync_device();
        tensor_z.sync_device();
    }

    /// Compares computed reference with device reference and outputs to a file
    /// if incorrect
    bool compare_reference() {
        tensor_grad.sync_host();

        EXPECT_GT(
                cutlass::reference::host::TensorNorm(tensor_filter.host_view()),
                0);
        EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_diff.host_view()),
                  0);
        EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_bias.host_view()),
                  0);

        if (tensor_grad.size() > 1)
            EXPECT_GT(cutlass::reference::host::TensorNorm(
                              tensor_grad.host_view()),
                      0);

        if (reference_grad.size() > 1)
            EXPECT_GT(cutlass::reference::host::TensorNorm(
                              reference_grad.host_view()),
                      0);

        bool passed = cutlass::reference::host::TensorEquals(
                reference_grad.host_view(), tensor_grad.host_view());

        if (!passed) {
            std::stringstream fname_ref;

            fname_ref << "error_Conv2d_Dgrad_device_reference_"
                      << Convolution::ThreadblockShape::kM << "x"
                      << Convolution::ThreadblockShape::kN << "x"
                      << Convolution::ThreadblockShape::kK << "_"
                      << Convolution::WarpShape::kM << "x"
                      << Convolution::WarpShape::kN << "x"
                      << Convolution::WarpShape::kK << ".txt";

            std::ofstream file_ref(fname_ref.str());

            file_ref << "\nReference output=\n" << reference_grad.host_view();

            std::stringstream fname_comp;

            fname_comp << "error_Conv2d_Dgrad_device_computed_"
                       << Convolution::ThreadblockShape::kM << "x"
                       << Convolution::ThreadblockShape::kN << "x"
                       << Convolution::ThreadblockShape::kK << "_"
                       << Convolution::WarpShape::kM << "x"
                       << Convolution::WarpShape::kN << "x"
                       << Convolution::WarpShape::kK << ".txt";

            std::ofstream file_comp(fname_comp.str());

            file_comp << "filter=\n" << tensor_filter.host_view();
            file_comp << "\ndiff=\n" << tensor_diff.host_view();
            file_comp << "\nrin=\n" << tensor_mask_input.host_view();
            file_comp << "\nrout=\n" << tensor_mask_output.host_view();
            file_comp << "\nComputed =\n" << tensor_grad.host_view();
        }

        EXPECT_TRUE(passed);

        return passed;
    }

    /// Verifies the result is a GEMM
    bool verify(cutlass::conv::Conv2dProblemSize conv_param,
                ElementCompute alpha, ElementCompute beta) {
        //
        // Verify
        //

        // FIXME: add beta, gamma ...
        cutlass::reference::host::RegionRestrictedConvolution2dDgrad<
                Convolution::kConvolutionType, typename Convolution::ElementSrc,
                typename Convolution::LayoutSrc,
                typename Convolution::ElementFilter,
                typename Convolution::LayoutFilter,
                typename Convolution::ElementMaskInput,
                typename Convolution::LayoutMaskInput,
                typename Convolution::ElementMaskOutput,
                typename Convolution::LayoutMaskOutput,
                typename Convolution::ElementDst,
                typename Convolution::LayoutDst, ElementCompute,
                ElementAccumulator, typename Convolution::Operator>
                reference_convolution;

        reference_convolution(conv_param, alpha, tensor_diff.host_ref(),
                              tensor_filter.host_ref(),
                              tensor_mask_input.host_ref(),
                              tensor_mask_output.host_ref(),
                              reference_grad.host_ref(), ElementAccumulator(0));

        return compare_reference();
    }

    /// Executes one test
    bool run(cutlass::conv::Conv2dProblemSize conv_param,
             ElementCompute alpha = ElementCompute(1),
             ElementCompute beta = ElementCompute(0),
             ElementCompute gamma = ElementCompute(0)) {
        this->initialize(conv_param);

        //
        // Initialize the CONVOLUTION operator
        //

        // FIXME: last arguments needs compatiable with BiasAddLinearCombination
        typename Convolution::Arguments arguments{
                conv_param,
                tensor_diff.device_ref(),
                tensor_filter.device_ref(),
                tensor_mask_input.device_ref(),
                tensor_mask_output.device_ref(),
                tensor_bias.device_ref(),
                tensor_z.device_ref(),
                tensor_grad.device_ref(),
                {alpha, beta, gamma}};

        Convolution conv_op;

        size_t workspace_size = Convolution::get_workspace_size(arguments);

        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        cutlass::Status status = conv_op.initialize(arguments, workspace.get());

        EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

        //
        // Run the CONVOLUTION
        //

        status = conv_op();

        EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

        //
        // Verify
        //

        bool passed = this->verify(conv_param, alpha, beta);

        if (!passed) {
            std::cout << "Error with alpha = " << alpha << "\n"
                      << conv_param << std::endl;
        }

        return passed;
    }

    bool perf(cutlass::conv::Conv2dProblemSize conv_param,
              ElementCompute alpha = ElementCompute(1), int iterations = 1,
              bool verify = false, ElementCompute beta = ElementCompute(0),
              ElementCompute gamma = ElementCompute(0)) {
        this->initialize(conv_param, verify);

        //
        // Initialize the CONVOLUTION operator
        //

        typename Convolution::Arguments arguments{
                conv_param,
                tensor_diff.device_ref(),
                tensor_filter.device_ref(),
                tensor_mask_input.device_ref(),
                tensor_mask_output.device_ref(),
                tensor_bias.device_ref(),
                tensor_z.device_ref(),
                tensor_grad.device_ref(),
                {alpha, beta, gamma}};

        Convolution conv_op;

        size_t workspace_size = Convolution::get_workspace_size(arguments);

        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        cutlass::Status status = conv_op.initialize(arguments, workspace.get());

        EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

        //
        // Run the CONVOLUTION
        //

        status = conv_op();
        status = conv_op();

        TimerGPU timer;
        for (int i = 0; i < iterations; ++i) {
            status = conv_op();
        }
        EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);
        float time_ms = timer.read() / static_cast<float>(iterations);
        float ops = 2.f * static_cast<float>(
                                  static_cast<int64_t>(conv_param.N) *
                                  conv_param.K * conv_param.P * conv_param.Q *
                                  conv_param.R * conv_param.S * conv_param.C);
        if_constexpr<is_depthwise_convolution<Convolution>()>([&](auto _) {
            auto&& conv_param_ = _(conv_param);
            ops /= static_cast<float>(conv_param_.C);
        });

        std::cout << conv_param << "Time = " << time_ms << "ms"
                  << "\n"
                  << "Performance = " << ops / (time_ms * 1e9) << "Tops"
                  << std::endl;

        bool passed = true;
        if (verify) {
            //
            // Verify
            //

            passed = this->verify(conv_param, alpha, beta);

            if (!passed) {
                std::cout << "Error with alpha = " << alpha << "\n"
                          << std::endl;
            }
        }
        return passed;
    }
};

template <typename Convolution>
bool BenchRegionRestrictedDepthwiseConv2dDgrad(int batch = 64,
                                               int iterations = 1000,
                                               bool do_verify = false) {
    bool passed = true;
    double problem_alpha[] = {1.0};

    RRConvDgradTestbed<Convolution> testbed;
    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;
    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;

    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    for (int fh : {3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29}) {
        for (int ih : {32}) {
            for (int group : {384}) {
                int ph = fh >> 1;
                int sh = 1;
                int iw = ih;
                int oh = (ih + (ph << 1) - fh) / sh + 1;
                int ow = (iw + (ph << 1) - fh) / sh + 1;
                if (oh <= 0 || ow <= 0) {
                    continue;
                }
                args.emplace_back(ConvolutionParameter{
                        batch /*N*/, ih /*H*/, iw /*W*/, group /*C*/,
                        group /*K*/, fh /*R*/, fh /*S*/, oh /*P*/, ow /*Q*/, ph,
                        ph, sh, sh, 1, 1, mode, 1 /*split_k_slices*/, group});
            }
        }
    }

    auto bench = [&args, &mode](int n, int h, int w, int c, int k, int r,
                                int s) -> void {
        int ph = r >> 1;
        int stride = 1;
        int oh = (h + (ph << 1) - r) / stride + 1;
        int ow = (w + (ph << 1) - s) / stride + 1;
        if (oh > 0 && ow > 0) {
            args.emplace_back(ConvolutionParameter{n, h, w, c, k, r, s, oh, ow,
                                                   ph, ph, stride, stride, 1, 1,
                                                   mode, 1, k});
        }
    };
    bench(64, 56, 56, 128, 128, 7, 7);
    bench(64, 56, 56, 128, 128, 27, 27);

    bench(64, 28, 28, 256, 256, 7, 7);
    bench(64, 28, 28, 256, 256, 27, 27);

    bench(64, 14, 14, 512, 512, 7, 7);
    bench(64, 14, 14, 512, 512, 27, 27);

    bench(64, 7, 7, 1024, 1024, 7, 7);
    bench(64, 7, 7, 1024, 1024, 27, 27);

    bool verify = do_verify;
    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            passed =
                    testbed.perf(arg, cutlass::from_real<ElementCompute>(alpha),
                                 iterations, verify);
            if (!passed)
                return false;
        }
    }
    return passed;
}

template <typename Convolution>
bool TestRegionRestrictedDepthwiseConv2dDgrad() {
    bool passed = true;
    double problem_alpha[] = {1.0};

    RRConvDgradTestbed<Convolution> testbed;

    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    // TEST
    for (int n : {1, 2, 4}) {
        for (int g : {1, 3, 5}) {
            for (int ih : {32}) {
                for (int fh :
                     {3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}) {
                    for (int ph : {static_cast<int>(fh / 2), 0}) {
                        for (int sh : {1, 2}) {
                            int oh = (ih + 2 * ph - fh) / sh + 1;
                            int ow = (ih + 2 * ph - fh) / sh + 1;
                            if (!(oh > 0 && ow > 0)) {
                                printf("unexpected spatial size of convolution "
                                       "testcase, skip it\n");
                                continue;
                            }
                            args.emplace_back(ConvolutionParameter{
                                    n, ih, ih, g, g, fh, fh, oh, ow, ph, ph, sh,
                                    sh, 1, 1, mode, 1, g});
                        }
                    }
                }
            }
        }
    }

    for (auto iter_arg = args.begin(); iter_arg != args.end(); iter_arg++) {
        for (auto alpha : problem_alpha) {
            auto arg = *iter_arg;
            passed =
                    testbed.run(arg, cutlass::from_real<ElementCompute>(alpha));
            if (!passed) {
                return false;
            }
        }
    }
    return passed;
}
}  // namespace device

}  // namespace convolution
}  // namespace test
