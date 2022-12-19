#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/convolution/device/convolution.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "region_restricted_conv2d_fprop_testbed.h"

#define RUN_DEPTHWISE_CONVOLUTION(stage, dt)                                 \
    do {                                                                     \
        using ElementOutput = float;                                         \
        using ElementAccumulator = float;                                    \
        using ElementCompute = float;                                        \
        using ElementBias = float;                                           \
        using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;          \
                                                                             \
        using Convolution =                                                  \
                cutlass::conv::device::RegionRestrictedConvolutionFprop<     \
                        float, cutlass::layout::TensorNCHW, float,           \
                        cutlass::layout::TensorNCHW, dt,                     \
                        cutlass::layout::TensorNCHW, dt,                     \
                        cutlass::layout::TensorNCHW, ElementOutput,          \
                        cutlass::layout::TensorNCHW, float,                  \
                        cutlass::layout::TensorNCHW, float,                  \
                        cutlass::conv::ConvType::kDepthwiseConvolution,      \
                        cutlass::arch::OpClassSimt, cutlass::arch::Sm50,     \
                        ThreadBlockShape, WarpShape, InstructionShape,       \
                        cutlass::epilogue::thread::BiasAddLinearCombination< \
                                ElementOutput, 1, ElementAccumulator,        \
                                ElementBias, ElementCompute>,                \
                        cutlass::conv::threadblock::                         \
                                DepthwiseConvolutionFpropThreadblockSwizzle, \
                        1, 1, 1, 1, 1,                                       \
                        cutlass::conv::SpecialOptimizeDesc::NONE,            \
                        cutlass::arch::OpMultiplyAdd,                        \
                        cutlass::conv::ImplicitGemmMode::GEMM_TN>;           \
                                                                             \
        EXPECT_TRUE(test::convolution::device::                              \
                            BenchRegionRestrictedDepthwiseConv2dFprop<       \
                                    Convolution>(32, 1000));                \
    } while (0)

TEST(SM50_Device_Region_Resticted_Depthwise_Conv2d_Fprop_f32_f32_nchw_simt_op_perf,
     128x64x8_32x64x8) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 64, 8>;
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
    RUN_DEPTHWISE_CONVOLUTION(1, int32_t);
}

TEST(SM50_DeviceRegion_Resticted__Depthwise_Conv2dFprop_f32_f32_NCHW_simt_op_perf,
     32x128x8_32x64x8) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<32, 128, 8>;
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
    RUN_DEPTHWISE_CONVOLUTION(1, int32_t);
}

TEST(SM50_DeviceRegion_Resticted__Depthwise_Conv2dFprop_f32_f32_NCHW_simt_op_perf,
     128x32x8_64x32x8) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 8>;
    RUN_DEPTHWISE_CONVOLUTION(1, int32_t);
}

TEST(SM50_DeviceRegion_Resticted__Depthwise_Conv2dFprop_f32_f32_NCHW_simt_op_perf,
     32x64x8_32x64x8) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<32, 64, 8>;
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
    RUN_DEPTHWISE_CONVOLUTION(1, int32_t);
}

TEST(SM50_DeviceRegion_Resticted__Depthwise_Conv2dFprop_f32_f32_NCHW_simt_op_perf,
     64x32x8_64x32x8) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<64, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 8>;
    RUN_DEPTHWISE_CONVOLUTION(1, int32_t);
}

TEST(SM50_DeviceRegion_Resticted__Depthwise_Conv2dFprop_f32_f32_NCHW_simt_op_perf,
     32x32x8_32x32x8) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<32, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, 8>;
    RUN_DEPTHWISE_CONVOLUTION(1, int32_t);
}
