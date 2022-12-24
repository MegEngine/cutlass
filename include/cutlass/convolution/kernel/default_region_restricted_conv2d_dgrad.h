
#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/convolution/kernel/implicit_batched_gemm_region_restricted_dwconv2d_dgrad.h"

#include "cutlass/convolution/threadblock/implicit_region_restricted_mma_core.h"
#include "cutlass/convolution/threadblock/implicit_region_restricted_mma_core_simt.h"
#include "cutlass/convolution/threadblock/implicit_mma_core.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_simt.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_sm70.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_sm75.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_sm80.h"

#include "cutlass/convolution/threadblock/dwconv2d_tile_iterator_tn.h"

#include "cutlass/convolution/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/threadblock/dwconv2d_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/dwconv2d_direct_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/dwconv2d_direct_epilogue_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/dwconv2d_direct_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/dwconv2d_epilogue_volta_tensor_op.h"
#include "cutlass/convolution/threadblock/dwconv2d_tile_iterator_tn_filter_dgrad_precomp.h"

#include "cutlass/epilogue/thread/bias_add_linear_combination_clamp.h"
#include "cutlass/epilogue/thread/bias_add_linear_combination_relu_clamp.h"
#include "cutlass/epilogue/thread/bias_add_linear_combination_hswish_clamp.h"
#include "cutlass/epilogue/thread/linear_combination.h"
namespace cutlass {
namespace conv {
namespace kernel {

template <
        /// Element type for Src Tensor operand
        typename ElementSrc,
        /// Layout type for Src Tensor operand
        typename LayoutSrc,
        /// Element type for Filter Tensor operand
        typename ElementFilter,
        /// Layout type for Filter Tensor operand
        typename LayoutFilter,
        /// Element type for mask input
        typename ElementMaskInput,
        /// Layout type for mask input
        typename LayoutMaskInput,
        /// Element type for mask output
        typename ElementMaskOutput,
        /// Layout type for mask output
        typename LayoutMaskOutput,
        /// Element type for Dst Tensor operands
        typename ElementDst,
        /// Layout type for Dst Tensor operands
        typename LayoutDst,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// MathOperatorTag class tag
        typename OperatorClass,
        /// Tag indicating architecture to tune for
        typename ArchTag,
        /// Threadblock-level tile size (concept: GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: GemmShape)
        typename WarpShape,
        /// Instruction-level tile size (concept: GemmShape)
        typename InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Operation performed by conv
        typename MathOperatorTag,
        /// Access granularity of Src Tensor in units of elements
        int AlignmentFilter,
        /// Access granularity of Filter Tensor in units of elements
        int AlignmentDiff,
        /// Access granularity of Rin Tensor in units of elements
        int kAlignmentMaskInput,
        /// Access granularity of Rout Tensor in units of elements
        int kAlignmentMaskOutput,
        /// Implicit Gemm Mode
        ImplicitGemmMode GemmMode,
        /// convolution type
        ConvType ConvolutionType = ConvType::kConvolution>
struct DefaultRegionRestrictedConvolution2dDgrad;

template <
        /// Element type for mask input
        typename ElementMaskInput,
        /// Layout type for mask input
        typename LayoutMaskInput,
        /// Element type for mask output
        typename ElementMaskOutput,
        /// Layout type for mask output
        typename LayoutMaskOutput,
        /// Element type for Dst Tensor operands
        typename ElementDst,
        /// Layout type for Dst Tensor operands
        typename LayoutDst,
        /// Tag indicating architecture to tune for
        typename ArchTag,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Threadblock-level tile size (concept: GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: GemmShape)
        typename WarpShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Operation performed by conv
        typename MathOperatorTag,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter,
        /// Access granularity of Rin Tensor in units of elements
        int kAlignmentMaskInput,
        /// Access granularity of Rout Tensor in units of elements
        int kAlignmentMaskOutput>
struct DefaultRegionRestrictedConvolution2dDgrad<
        float, layout::TensorNCHW, float, layout::TensorNCHW, ElementMaskInput,
        LayoutMaskInput, ElementMaskOutput, LayoutMaskOutput, ElementDst,
        LayoutDst, ElementAccumulator, arch::OpClassSimt, ArchTag,
        ThreadblockShape, WarpShape, gemm::GemmShape<1, 1, 1>, EpilogueOutputOp,
        ThreadblockSwizzle, Stages, MathOperatorTag, kAlignmentSrc,
        kAlignmentFilter, kAlignmentMaskInput, kAlignmentMaskOutput,
        ImplicitGemmMode::GEMM_TN, ConvType::kDepthwiseConvolution> {
    using InstructionShape = gemm::GemmShape<1, 1, 1>;
    using ElementSrc = float;
    using ElementFilter = float;
    using LayoutSrc = layout::TensorNCHW;
    using LayoutFilter = layout::TensorNCHW;
    using OperatorClass = arch::OpClassSimt;
    static const int kStages = Stages;
    // FIXME: using?
    static const ImplicitGemmMode kGemmMode = ImplicitGemmMode::GEMM_TN;
    static const ConvType kConvolutionType = ConvType::kDepthwiseConvolution;

    // Define the MmaCore components
    using MmaCore = typename cutlass::conv::threadblock::
            DefaultRegionRestrictedDgradMmaCore<
                    ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
                    LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
                    kAlignmentFilter, ElementMaskInput, LayoutMaskInput,
                    kAlignmentMaskInput, ElementMaskOutput, LayoutMaskOutput,
                    kAlignmentMaskOutput, ElementAccumulator, LayoutDst,
                    OperatorClass, Stages, MathOperatorTag, true, kGemmMode>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc = cutlass::conv::threadblock::Dwconv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
            ElementSrc, LayoutSrc, typename MmaCore::IteratorThreadMapSrc,
            MmaCore::IteratorThreadMapSrc::kElementsPerAccess>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter =
            cutlass::conv::threadblock::Dwconv2dTileFilterIteratorDgradPrecomp<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kN>,
                    ElementFilter, LayoutFilter,
                    typename MmaCore::IteratorThreadMapFilter,
                    MmaCore::IteratorThreadMapFilter::kElementsPerAccess>;

    // Define iterators over tiles from the mask input
    using IteratorMaskInput = cutlass::conv::threadblock::Dwconv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kN>,
            ElementMaskInput, LayoutMaskInput,
            typename MmaCore::IteratorThreadMapMaskInput,
            MmaCore::IteratorThreadMapMaskInput::kElementsPerAccess>;

    using IteratorMaskOutput = cutlass::conv::threadblock::Dwconv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
            ElementMaskOutput, LayoutMaskOutput,
            typename MmaCore::IteratorThreadMapMaskOutput,
            MmaCore::IteratorThreadMapMaskOutput::kElementsPerAccess>;

    using MmaPipelineSingleStage = cutlass::conv::threadblock::
            RegionRestrictedDgradMmaNtPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, IteratorMaskInput,
                    typename MmaCore::SmemIteratorMaskInput, IteratorMaskOutput,
                    typename MmaCore::SmemIteratorMaskOutput,
                    ElementAccumulator, LayoutDst, typename MmaCore::MmaPolicy>;

    using Mma = MmaPipelineSingleStage;

    static_assert(kStages == 1, "Two stage is not supported");

    // Define the epilogue
    using LayoutBias = LayoutDst;
    using Epilogue =
            typename cutlass::epilogue::threadblock::Dwconv2dEpilogueSimt<
                    ThreadblockShape, LayoutDst, LayoutBias,
                    typename Mma::Operator,  // MmaWarpSimt
                    EpilogueOutputOp, 1>::Epilogue;

    /// Define the kernel-level conv operator.
    using Kernel = cutlass::conv::kernel::
            ImplicitBatchedGemmRegionRestrictedDepthwiseConvolution2dDgrad<
                    Mma, Epilogue, ThreadblockSwizzle>;
};

}  // namespace kernel
}  // namespace conv
}  // namespace cutlass
