/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
modification, are permitted
 * provided that the following conditions are met:
namespace conv {
 *     * Redistributions of source code must retain the above copyright notice,
this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
notice, this list of
 *       conditions and the following disclaimer in the documentation and/or
other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
contributors may be used
 *       to endorse or promote products derived from this software without
specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA
CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/**
 * \file include/cutlass/convolution/kernel/default_conv2d_fprop.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
/*! \file
    \brief
    Default kernel-level implicit GEMM conv definitions combine
   threadblock-scoped matrix multiply-add with the appropriate
   threadblock-scoped epilogue.
*/

#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/convolution/kernel/implicit_gemm_nt_convolution.h"
#include "cutlass/convolution/kernel/implicit_gemm_nt_precomp_convolution.h"
#include "cutlass/convolution/kernel/implicit_gemm_tn_precomp_convolution.h"

#include "cutlass/convolution/threadblock/conv2d_tile_iterator_nt.h"
#include "cutlass/convolution/threadblock/conv2d_tile_iterator_nt_src_fprop_precomp.h"
#include "cutlass/convolution/threadblock/conv2d_tile_iterator_tn_fprop_nhwc_precomp.h"

#include "cutlass/convolution/threadblock/implicit_mma_core.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_simt.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_sm75.h"

#include "cutlass/convolution/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/threadblock/convolution_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/convolution_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"

#include "cutlass/epilogue/thread/bias_add_linear_combination_relu_clamp.h"
#include "cutlass/epilogue/thread/bias_add_linear_combination_hswish_clamp.h"
#include "cutlass/epilogue/thread/linear_combination.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ElementSrc,
          /// Layout type for Src Tensor operand
          typename LayoutSrc,
          /// Element type for Filter Tensor operand
          typename ElementFilter,
          /// Layout type for Filter Tensor operand
          typename LayoutFilter,
          /// Element type for Dst and Z Tensor operands
          typename ElementDst,
          /// Layout type for Dst and Z Tensor operands
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
          int AlignmentSrc,
          /// Access granularity of Filter Tensor in units of elements
          int AlignmentFilter,
          /// whether use special optimization for conv 1x1
          bool NeedLoadFromConstMem = true,
          /// Implicit Gemm Mode
          ImplicitGemmMode GemmMode = ImplicitGemmMode::GEMM_NT,
          /// use reorder filter K to avoid shared load
          bool WithoutSharedLoad = false>
struct DefaultConvolution2dFprop;

/// Partial specialization for SIMT DP4A
template <
        /// Element type for Dst and Z Tensor operands
        typename ElementDst,
        /// Layout type for Dst and Z Tensor operand
        typename LayoutDst,
        /// Tag indicating architecture to tune for
        typename ArchTag,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
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
        int kAlignmentFilter>
struct DefaultConvolution2dFprop<
        int8_t, layout::TensorCxRSKx<4>, int8_t, layout::TensorCxRSKx<4>,
        ElementDst, LayoutDst, ElementAccumulator, arch::OpClassSimt, ArchTag,
        ThreadblockShape, WarpShape, gemm::GemmShape<1, 1, 4>, EpilogueOutputOp,
        ThreadblockSwizzle, Stages, MathOperatorTag, kAlignmentSrc,
        kAlignmentFilter> {
    using InstructionShape = gemm::GemmShape<1, 1, 4>;
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorCxRSKx<4>;
    using LayoutFilter = layout::TensorCxRSKx<4>;
    using OperatorClass = arch::OpClassSimt;
    static const int kStages = Stages;

    // Define the MmaCore components
    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass,
            Stages, MathOperatorTag, true>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc = cutlass::conv::threadblock::Conv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementSrc, LayoutSrc, 4, typename MmaCore::IteratorThreadMapSrc,
            MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
            cutlass::conv::threadblock::TileMap<
                    LayoutSrc,
                    cutlass::conv::threadblock::TileMapType::kRow2C_Col2N>>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter = cutlass::conv::threadblock::Conv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
            ElementFilter, LayoutFilter, 4,
            typename MmaCore::IteratorThreadMapFilter,
            MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
            cutlass::conv::threadblock::TileMap<
                    LayoutFilter,
                    cutlass::conv::threadblock::TileMapType::kRow2C_Col2N>>;

    using MmaPipelineSingleStage = cutlass::conv::threadblock::MmaNtSingleStage<
            typename MmaCore::Shape, IteratorSrc,
            typename MmaCore::SmemIteratorSrc, IteratorFilter,
            typename MmaCore::SmemIteratorFilter, ElementAccumulator, LayoutDst,
            typename MmaCore::MmaPolicy>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPipelineTwoStages = cutlass::conv::threadblock::MmaNtPipelined<
            typename MmaCore::Shape, IteratorSrc,
            typename MmaCore::SmemIteratorSrc, IteratorFilter,
            typename MmaCore::SmemIteratorFilter, ElementAccumulator, LayoutDst,
            typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    static int const kEpilogueElementsPerAccess = 4;

    /// Define the epilogue
    using Epilogue =
            typename cutlass::epilogue::threadblock::ConvolutionEpilogueSimt<
                    ThreadblockShape, LayoutDst, LayoutDst,
                    typename Mma::Operator, EpilogueOutputOp,
                    kEpilogueElementsPerAccess>::Epilogue;

    /// Define the kernel-level conv operator.
    using Kernel = cutlass::conv::kernel::ImplicitGemmNtConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kFprop>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for NCHW4 layout

template <typename LayoutDst,
          /// Element type for Dst and Z Tensor operands
          typename ElementDst,
          /// Tag indicating architecture to tune for
          typename ArchTag,
          /// Element type for internal accumulation
          typename ElementAccumulator,
          /// Threadblock-level tile size (concept: gemm::GemmShape)
          typename ThreadblockShape,
          /// Warp-level tile size (concept: gemm::GemmShape)
          typename WarpShape,
          /// Epilogue output operator
          typename EpilogueOutputOp,
          /// Threadblock-level swizzling operator
          typename ThreadblockSwizzle,
          /// Number of stages used in the pipelined mainloop
          int Stages,
          /// Operation performed by GEMM
          typename MathOperatorTag,
          /// Access granularity of Src Tensor in units of elements
          int kAlignmentSrc,
          /// Access granularity of Filter Tensor in units of elements
          int kAlignmentFilter,
          /// whether use special optimization for conv 1x1
          bool NeedLoadFromConstMem>
struct DefaultConvolution2dFprop<
        int8_t, layout::TensorNCxHWx<4>, int8_t, layout::TensorCxRSKx<4>,
        ElementDst, LayoutDst, ElementAccumulator, arch::OpClassSimt, ArchTag,
        ThreadblockShape, WarpShape, gemm::GemmShape<1, 1, 4>, EpilogueOutputOp,
        ThreadblockSwizzle, Stages, MathOperatorTag, kAlignmentSrc,
        kAlignmentFilter, NeedLoadFromConstMem> {
    using InstructionShape = gemm::GemmShape<1, 1, 4>;
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<4>;
    using LayoutFilter = layout::TensorCxRSKx<4>;
    static const int kStages = Stages;

    using OperatorClass = arch::OpClassSimt;
    /// Define the threadblock-scoped matrix multiply-accumulate
    // Define the MmaCore components
    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass,
            Stages, MathOperatorTag, true>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc =
            cutlass::conv::threadblock::Conv2dTileSrcIteratorFpropPrecomp<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kN>,
                    ElementSrc, LayoutSrc,
                    typename MmaCore::IteratorThreadMapSrc,
                    MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
                    cutlass::conv::threadblock::TileMap<
                            LayoutSrc, cutlass::conv::threadblock::TileMapType::
                                               kRow2C_Col2NHW>,
                    NeedLoadFromConstMem>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter = cutlass::conv::threadblock::Conv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
            ElementFilter, LayoutFilter, 4,
            typename MmaCore::IteratorThreadMapFilter,
            MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
            cutlass::conv::threadblock::TileMap<
                    LayoutFilter,
                    cutlass::conv::threadblock::TileMapType::kRow2CHW_Col2N>>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaNtPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaNtPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    static int const kEpilogueElementsPerAccess =
            cutlass::platform::is_same<ElementDst, float>::value
                    ? 1
                    : (cutlass::platform::is_same<ElementDst,
                                                  cutlass::int4b_t>::value ||
                                       cutlass::platform::is_same<
                                               ElementDst,
                                               cutlass::uint4b_t>::value
                               ? 8
                               : 4);

    /// Define the epilogue
    using Epilogue =
            typename cutlass::epilogue::threadblock::ConvolutionEpilogueSimt<
                    ThreadblockShape, LayoutDst, LayoutDst,
                    typename Mma::Operator, EpilogueOutputOp,
                    kEpilogueElementsPerAccess>::Epilogue;

    /// Define the kernel-level conv operator.
    using Kernel = cutlass::conv::kernel::ImplicitGemmNtPrecompConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kFprop>;
};
////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for CHWN4 layout

template <  /// Layout type for Dst and Z Tensor operand
        typename LayoutDst,
        /// Element type for Dst and Z Tensor operands
        typename ElementDst,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Instruction-level tile size (concept: gemm::GemmShape)
        typename InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Operation performed by GEMM
        typename MathOperatorTag,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter>
struct DefaultConvolution2dFprop<
        int8_t, layout::TensorCxRSKx<4>, int8_t, layout::TensorCxRSKx<16>,
        ElementDst, LayoutDst, ElementAccumulator, arch::OpClassTensorOp,
        arch::Sm75, ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp, ThreadblockSwizzle, 2, MathOperatorTag, kAlignmentSrc,
        kAlignmentFilter> {
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorCxRSKx<4>;
    using LayoutFilter = layout::TensorCxRSKx<16>;
    using OperatorClass = arch::OpClassTensorOp;

    // Define the MmaCore components
    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass, 2,
            MathOperatorTag, true>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc = cutlass::conv::threadblock::Conv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementSrc, LayoutSrc, 4, typename MmaCore::IteratorThreadMapSrc,
            MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
            cutlass::conv::threadblock::TileMap<
                    LayoutSrc,
                    cutlass::conv::threadblock::TileMapType::kRow2C_Col2NHW>>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter = cutlass::conv::threadblock::Conv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
            ElementFilter, LayoutFilter, 16,
            typename MmaCore::IteratorThreadMapFilter,
            MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
            cutlass::conv::threadblock::TileMap<
                    LayoutFilter,
                    cutlass::conv::threadblock::TileMapType::kRow2CHW_Col2N>>;

    // Define the threadblock-scoped pipelined matrix multiply
    using Mma = typename cutlass::conv::threadblock::MmaNtPipelined<
            typename MmaCore::Shape, IteratorSrc,
            typename MmaCore::SmemIteratorSrc, IteratorFilter,
            typename MmaCore::SmemIteratorFilter, ElementAccumulator, LayoutDst,
            typename MmaCore::MmaPolicy>;

    static int const kEpilogueElementsPerAccess = 4;

    /// Define the epilogue
    using Epilogue =
            typename cutlass::epilogue::threadblock::ConvolutionEpilogueSimt<
                    ThreadblockShape, LayoutDst, LayoutDst,
                    typename Mma::Operator, EpilogueOutputOp,
                    kEpilogueElementsPerAccess>::Epilogue;

    /// Define the kernel-level conv operator.
    using Kernel = cutlass::conv::kernel::ImplicitGemmNtConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kFprop>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for NCHW32 layout

template <  /// Layout type for Dst and Z Tensor operand
        typename LayoutDst_,
        /// Element type for Dst and Z Tensor operands
        typename ElementDst,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Instruction-level tile size (concept: gemm::GemmShape)
        typename InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Interleaving quantity
        int Interleaved,
        /// Operation performed by GEMM
        typename MathOperatorTag,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter, bool NeedLoadFromConstMem>
struct DefaultConvolution2dFprop<
        int8_t, layout::TensorNCxHWx<Interleaved>, int8_t,
        layout::TensorCxRSKx<Interleaved>, ElementDst, LayoutDst_,
        ElementAccumulator, arch::OpClassTensorOp, arch::Sm75, ThreadblockShape,
        WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
        Stages, MathOperatorTag, kAlignmentSrc, kAlignmentFilter,
        NeedLoadFromConstMem> {
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<Interleaved>;
    using LayoutFilter = layout::TensorCxRSKx<Interleaved>;
    using LayoutDst = layout::TensorNCxHWx<Interleaved>;
    using OperatorClass = arch::OpClassTensorOp;
    static const int kStages = Stages;

    static_assert(kAlignmentSrc == 128 / sizeof_bits<ElementSrc>::value,
                  "Alignment must match thread data map's vector length");

    static_assert(kAlignmentFilter == 128 / sizeof_bits<ElementFilter>::value,
                  "Alignment must match thread data map's vector length");

    // Define the MmaCore components
    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass,
            Stages, MathOperatorTag, true>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc =
            cutlass::conv::threadblock::Conv2dTileSrcIteratorFpropPrecomp<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kN>,
                    ElementSrc, LayoutSrc,
                    typename MmaCore::IteratorThreadMapSrc,
                    MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
                    cutlass::conv::threadblock::TileMap<
                            LayoutSrc, cutlass::conv::threadblock::TileMapType::
                                               kRow2C_Col2NHW>,
                    NeedLoadFromConstMem>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter = cutlass::conv::threadblock::Conv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
            ElementFilter, LayoutFilter, Interleaved,
            typename MmaCore::IteratorThreadMapFilter,
            MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
            cutlass::conv::threadblock::TileMap<
                    LayoutFilter,
                    cutlass::conv::threadblock::TileMapType::kRow2CHW_Col2N>>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaNtPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaNtPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    /// 64 bit store
    static int const kEpilogueElementsPerAccess =
            64 / sizeof_bits<ElementDst>::value;

    /// Define the epilogue
    using Epilogue = typename cutlass::epilogue::threadblock::
            ConvolutionEpilogueTensorOp<ThreadblockShape, LayoutDst, LayoutDst,
                                        typename Mma::Operator,
                                        EpilogueOutputOp,
                                        kEpilogueElementsPerAccess>::Epilogue;

    /// Define the kernel-level conv operator.
    using Kernel = cutlass::conv::kernel::ImplicitGemmNtPrecompConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kFprop>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for NCHW32 layout
/// Use GEMM NT and WithoutSharedLoad
template <  /// Layout type for Dst and Z Tensor operand
        typename LayoutDst_,
        /// Element type for Dst and Z Tensor operands
        typename ElementDst,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Instruction-level tile size (concept: gemm::GemmShape)
        typename InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Interleaving quantity
        int Interleaved,
        /// Operation performed by GEMM
        typename MathOperatorTag,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter, bool NeedLoadFromConstMem>
struct DefaultConvolution2dFprop<
        int8_t, layout::TensorNCxHWx<Interleaved>, int8_t,
        layout::TensorCxRSKx<Interleaved>, ElementDst, LayoutDst_,
        ElementAccumulator, arch::OpClassTensorOp, arch::Sm75, ThreadblockShape,
        WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
        Stages, MathOperatorTag, kAlignmentSrc, kAlignmentFilter,
        NeedLoadFromConstMem, ImplicitGemmMode::GEMM_TN, true> {
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<Interleaved>;
    using LayoutFilter = layout::TensorCxRSKx<Interleaved>;
    using LayoutDst = layout::TensorNCxHWx<Interleaved>;
    using OperatorClass = arch::OpClassTensorOp;
    static const int kStages = Stages;

    static_assert(kAlignmentSrc == 128 / sizeof_bits<ElementSrc>::value,
                  "Alignment must match thread data map's vector length");

    static_assert(kAlignmentFilter == 128 / sizeof_bits<ElementFilter>::value,
                  "Alignment must match thread data map's vector length");

    // Define the MmaCore components
    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass,
            Stages, MathOperatorTag, true, ImplicitGemmMode::GEMM_TN>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc =
            cutlass::conv::threadblock::Conv2dTileSrcIteratorFpropPrecomp<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kM>,
                    ElementSrc, LayoutSrc,
                    typename MmaCore::IteratorThreadMapSrc,
                    MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
                    cutlass::conv::threadblock::TileMap<
                            LayoutSrc, cutlass::conv::threadblock::TileMapType::
                                               kRow2C_Col2NHW>,
                    NeedLoadFromConstMem, ImplicitGemmMode::GEMM_TN>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter = cutlass::conv::threadblock::Conv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementFilter, LayoutFilter, Interleaved,
            typename MmaCore::IteratorThreadMapFilter,
            MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
            cutlass::conv::threadblock::TileMap<
                    LayoutFilter,
                    cutlass::conv::threadblock::TileMapType::kRow2CHW_Col2N>,
            ImplicitGemmMode::GEMM_TN>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaTnPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaTnPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    /// 64 bit store
    static int const kEpilogueElementsPerAccess =
            64 / sizeof_bits<ElementDst>::value;

    /// Define the epilogue
    using Epilogue = typename cutlass::epilogue::threadblock::
            ConvolutionEpilogueTensorOp<
                    ThreadblockShape, LayoutDst, LayoutDst,
                    typename Mma::Operator, EpilogueOutputOp,
                    kEpilogueElementsPerAccess, true>::Epilogue;

    /// Define the kernel-level conv operator.
    using Kernel = cutlass::conv::kernel::ImplicitGemmTnPrecompConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kFprop>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for int4b_t and NCHW64 layout
/// ElementFilter must be int4b_t
/// ElementSrc can be int4b_t or uint4b_t

template <
        /// ElementSrc is int4b_t or uint4b_t
        bool Signed,
        /// Element type for Dst and Z Tensor operands
        typename ElementDst,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Instruction-level tile size (concept: gemm::GemmShape)
        typename InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Interleaving quantity
        int Interleaved,
        /// Operation performed by GEMM
        typename MathOperatorTag,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter, bool NeedLoadFromConstMem>
struct DefaultConvolution2dFprop<
        integer_subbyte<4, Signed>, layout::TensorNCxHWx<Interleaved>, int4b_t,
        layout::TensorCxRSKx<Interleaved>, ElementDst,
        layout::TensorNCxHWx<Interleaved>, ElementAccumulator,
        arch::OpClassTensorOp, arch::Sm75, ThreadblockShape, WarpShape,
        InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, Stages,
        MathOperatorTag, kAlignmentSrc, kAlignmentFilter,
        NeedLoadFromConstMem> {
    static_assert(Interleaved == 64, "Interleaved must be divisible by 64");

    using ElementSrc = integer_subbyte<4, Signed>;
    using ElementFilter = int4b_t;
    using LayoutSrc = layout::TensorNCxHWx<Interleaved>;
    using LayoutFilter = layout::TensorCxRSKx<Interleaved>;
    using LayoutDst = layout::TensorNCxHWx<Interleaved>;
    using OperatorClass = arch::OpClassTensorOp;
    static const int kStages = Stages;

    static_assert(kAlignmentSrc == 128 / sizeof_bits<ElementSrc>::value,
                  "Alignment must match thread data map's vector length");

    static_assert(kAlignmentFilter == 128 / sizeof_bits<ElementFilter>::value,
                  "Alignment must match thread data map's vector length");

    // Define the MmaCore components
    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass,
            Stages, MathOperatorTag, true>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc =
            cutlass::conv::threadblock::Conv2dTileSrcIteratorFpropPrecomp<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kN>,
                    ElementSrc, LayoutSrc,
                    typename MmaCore::IteratorThreadMapSrc,
                    MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
                    cutlass::conv::threadblock::TileMap<
                            LayoutSrc, cutlass::conv::threadblock::TileMapType::
                                               kRow2C_Col2NHW>,
                    NeedLoadFromConstMem>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter = cutlass::conv::threadblock::Conv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
            ElementFilter, LayoutFilter, Interleaved,
            typename MmaCore::IteratorThreadMapFilter,
            MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
            cutlass::conv::threadblock::TileMap<
                    LayoutFilter,
                    cutlass::conv::threadblock::TileMapType::kRow2CHW_Col2N>>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaNtPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaNtPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    /// 64 bit store
    static int const kEpilogueElementsPerAccess =
            64 / sizeof_bits<ElementDst>::value;

    /// Define the epilogue
    using Epilogue = typename cutlass::epilogue::threadblock::
            ConvolutionEpilogueTensorOp<ThreadblockShape, LayoutDst, LayoutDst,
                                        typename Mma::Operator,
                                        EpilogueOutputOp,
                                        kEpilogueElementsPerAccess>::Epilogue;

    /// Define the kernel-level conv operator.
    using Kernel = cutlass::conv::kernel::ImplicitGemmNtPrecompConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kFprop>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for int4b_t and NCHW64 layout
/// ElementFilter must be int4b_t
/// ElementSrc can be int4b_t or uint4b_t
/// Use GEMM NT and WithoutSharedLoad

template <
        /// ElementSrc is int4b_t or uint4b_t
        bool Signed,
        /// Element type for Dst and Z Tensor operands
        typename ElementDst,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Instruction-level tile size (concept: gemm::GemmShape)
        typename InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Interleaving quantity
        int Interleaved,
        /// Operation performed by GEMM
        typename MathOperatorTag,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter, bool NeedLoadFromConstMem>
struct DefaultConvolution2dFprop<
        integer_subbyte<4, Signed>, layout::TensorNCxHWx<Interleaved>, int4b_t,
        layout::TensorCxRSKx<Interleaved>, ElementDst,
        layout::TensorNCxHWx<Interleaved>, ElementAccumulator,
        arch::OpClassTensorOp, arch::Sm75, ThreadblockShape, WarpShape,
        InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, Stages,
        MathOperatorTag, kAlignmentSrc, kAlignmentFilter, NeedLoadFromConstMem,
        ImplicitGemmMode::GEMM_TN, true> {
    static_assert(Interleaved == 64, "Interleaved must be divisible by 64");

    using ElementSrc = integer_subbyte<4, Signed>;
    using ElementFilter = int4b_t;
    using LayoutSrc = layout::TensorNCxHWx<Interleaved>;
    using LayoutFilter = layout::TensorCxRSKx<Interleaved>;
    using LayoutDst = layout::TensorNCxHWx<Interleaved>;
    using OperatorClass = arch::OpClassTensorOp;
    static const int kStages = Stages;

    static_assert(kAlignmentSrc == 128 / sizeof_bits<ElementSrc>::value,
                  "Alignment must match thread data map's vector length");

    static_assert(kAlignmentFilter == 128 / sizeof_bits<ElementFilter>::value,
                  "Alignment must match thread data map's vector length");

    // Define the MmaCore components
    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass,
            Stages, MathOperatorTag, true, ImplicitGemmMode::GEMM_TN>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc =
            cutlass::conv::threadblock::Conv2dTileSrcIteratorFpropPrecomp<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kM>,
                    ElementSrc, LayoutSrc,
                    typename MmaCore::IteratorThreadMapSrc,
                    MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
                    cutlass::conv::threadblock::TileMap<
                            LayoutSrc, cutlass::conv::threadblock::TileMapType::
                                               kRow2C_Col2NHW>,
                    NeedLoadFromConstMem, ImplicitGemmMode::GEMM_TN>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter = cutlass::conv::threadblock::Conv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementFilter, LayoutFilter, Interleaved,
            typename MmaCore::IteratorThreadMapFilter,
            MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
            cutlass::conv::threadblock::TileMap<
                    LayoutFilter,
                    cutlass::conv::threadblock::TileMapType::kRow2CHW_Col2N>,
            ImplicitGemmMode::GEMM_TN>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaTnPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaTnPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    /// 64 bit store
    static int const kEpilogueElementsPerAccess =
            64 / sizeof_bits<ElementDst>::value;

    /// Define the epilogue
    using Epilogue = typename cutlass::epilogue::threadblock::
            ConvolutionEpilogueTensorOp<
                    ThreadblockShape, LayoutDst, LayoutDst,
                    typename Mma::Operator, EpilogueOutputOp,
                    kEpilogueElementsPerAccess, true>::Epilogue;

    /// Define the kernel-level conv operator.
    using Kernel = cutlass::conv::kernel::ImplicitGemmTnPrecompConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kFprop>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for NCHW32 layout

template <  /// Element type for Dst and Z Tensor operands
        typename ElementDst,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Instruction-level tile size (concept: gemm::GemmShape)
        typename InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Interleaving quantity
        int Interleaved,
        /// Operation performed by GEMM
        typename MathOperatorTag,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter, bool NeedLoadFromConstMem>
struct DefaultConvolution2dFprop<
        int8_t, layout::TensorNCxHWx<Interleaved>, int8_t,
        layout::TensorCxRSKx<Interleaved>, ElementDst, layout::TensorNCxHWx<4>,
        ElementAccumulator, arch::OpClassTensorOp, arch::Sm75, ThreadblockShape,
        WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
        Stages, MathOperatorTag, kAlignmentSrc, kAlignmentFilter,
        NeedLoadFromConstMem> {
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<Interleaved>;
    using LayoutFilter = layout::TensorCxRSKx<Interleaved>;
    using LayoutDst = layout::TensorNCxHWx<4>;

    using OperatorClass = arch::OpClassTensorOp;
    static const int kStages = Stages;

    static_assert(kAlignmentSrc == 128 / sizeof_bits<ElementSrc>::value,
                  "Alignment must match thread data map's vector length");

    static_assert(kAlignmentFilter == 128 / sizeof_bits<ElementFilter>::value,
                  "Alignment must match thread data map's vector length");

    // Define the MmaCore components
    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass,
            Stages, MathOperatorTag, true>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc =
            cutlass::conv::threadblock::Conv2dTileSrcIteratorFpropPrecomp<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kN>,
                    ElementSrc, LayoutSrc,
                    typename MmaCore::IteratorThreadMapSrc,
                    MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
                    cutlass::conv::threadblock::TileMap<
                            LayoutSrc, cutlass::conv::threadblock::TileMapType::
                                               kRow2C_Col2NHW>,
                    NeedLoadFromConstMem>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter = cutlass::conv::threadblock::Conv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
            ElementFilter, LayoutFilter, 32,
            typename MmaCore::IteratorThreadMapFilter,
            MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
            cutlass::conv::threadblock::TileMap<
                    LayoutFilter,
                    cutlass::conv::threadblock::TileMapType::kRow2CHW_Col2N>>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaNtPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaNtPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    /// 32 bit store
    static int const kEpilogueElementsPerAccess =
            32 / sizeof_bits<ElementDst>::value;

    /// Define the epilogue
    using Epilogue = typename cutlass::epilogue::threadblock::
            ConvolutionEpilogueTensorOp<ThreadblockShape, LayoutDst, LayoutDst,
                                        typename Mma::Operator,
                                        EpilogueOutputOp,
                                        kEpilogueElementsPerAccess>::Epilogue;

    /// Define the kernel-level conv operator.
    using Kernel = cutlass::conv::kernel::ImplicitGemmNtPrecompConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kFprop>;
};

template <
        /// ElementSrc is int4b_t or uint4b_t
        bool Signed,
        /// Element type for Dst and Z Tensor operands
        typename ElementDst,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Instruction-level tile size (concept: gemm::GemmShape)
        typename InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Operation performed by GEMM
        typename MathOperatorTag,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter,
        /// whether use special optimization for conv 1x1
        bool NeedLoadFromConstMem,
        /// use reorder filter K to avoid shared load
        bool WithoutSharedLoad>
struct DefaultConvolution2dFprop<
        integer_subbyte<4, Signed>, layout::TensorNHWC, int4b_t,
        layout::TensorNCxHWx<kAlignmentFilter>, ElementDst, layout::TensorNHWC,
        ElementAccumulator, arch::OpClassTensorOp, arch::Sm75, ThreadblockShape,
        WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
        Stages, MathOperatorTag, kAlignmentSrc, kAlignmentFilter,
        NeedLoadFromConstMem, ImplicitGemmMode::GEMM_TN, WithoutSharedLoad> {
    using ElementSrc = integer_subbyte<4, Signed>;
    using ElementFilter = int4b_t;
    using LayoutSrc = layout::TensorNHWC;
    using LayoutFilter = layout::TensorNCxHWx<kAlignmentFilter>;
    using LayoutDst = layout::TensorNHWC;
    using OperatorClass = arch::OpClassTensorOp;
    static const int kStages = Stages;

    static_assert(kAlignmentSrc == kAlignmentFilter,
                  "kAlignmentSrc and kAlignmentFilter must be the same");

    static_assert(kAlignmentSrc % 8 == 0,
                  "Alignment must match thread data map's vector length");

    static_assert(kAlignmentFilter % 8 == 0,
                  "Alignment must match thread data map's vector length");

    // Define the MmaCore components
    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass,
            Stages, MathOperatorTag, false, ImplicitGemmMode::GEMM_TN>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc =
            cutlass::conv::threadblock::Conv2dTileSrcIteratorFpropPrecompNHWC<
                    cutlass::MatrixShape<ThreadblockShape::kM,
                                         ThreadblockShape::kK>,
                    ElementSrc, LayoutSrc,
                    typename MmaCore::IteratorThreadMapSrc, kAlignmentSrc,
                    NeedLoadFromConstMem>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter =
            cutlass::conv::threadblock::Conv2dTileFilterIteratorFpropKCxRSx<
                    cutlass::MatrixShape<ThreadblockShape::kK,
                                         ThreadblockShape::kN>,
                    ElementFilter, LayoutFilter,
                    typename MmaCore::IteratorThreadMapFilter,
                    kAlignmentFilter>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaTnPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaTnPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    using Epilogue = typename cutlass::epilogue::threadblock::
            ConvolutionEpilogueTensorOp<
                    ThreadblockShape, LayoutDst, LayoutDst,
                    typename Mma::Operator, EpilogueOutputOp,
                    EpilogueOutputOp::kCount, WithoutSharedLoad>::Epilogue;

    /// Define the kernel-level conv operator.
    using Kernel = cutlass::conv::kernel::ImplicitGemmTnPrecompConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kFprop>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace conv
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
