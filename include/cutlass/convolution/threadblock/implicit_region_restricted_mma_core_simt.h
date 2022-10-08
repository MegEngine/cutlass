/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *notice, this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
   \brief Defines basic properties needed by CTA-level GEMMs assuming
  expectations about data layout of the global memory fragments, data types,
  and internal tile sizes.

     Partial specializations for threadblock::Mma operations targeting simt
  instructions.
*/

/**
 * \file include/cutlass/convolution/threadblock/implicit_mma_core_simt.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear_2dthreadtile.h"
#include "cutlass/convolution/threadblock/regular_tile_iterator_transposed.h"

#include "cutlass/convolution/threadblock/implicit_mma_core.h"
#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "cutlass/layout/tensor.h"

#include "cutlass/convolution/threadblock/dwconv2d_tile_iterator_tn_filter_fprop_precomp.h"
#include <cutlass/convolution/threadblock/implicit_mma_core_simt.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///
///   A: layout::TensorNCHW
///   B: layout::TensorNCHW
///   Operator: simt class
///
/// This uses the default warp-level operator given tile sizes
template <
        /// Shape of threadblock-scoped matrix multiply operator (concept:
        /// GemmShape)
        typename Shape_,
        /// Shape of warp-level matrix multiply operator (concept: GemmShape)
        typename WarpShape_,
        /// Element data type of Src Tensor operand
        typename ElementSrc_,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Element data type of Filter Tensor operand
        typename ElementFilter_,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter,
        /// Element data type of mask input
        typename ElementMaskInput_,
        /// Layout of mask input
        typename LayoutMaskInput_,
        /// Access granularity of mask input in units of elements
        int kAlignmentMaskInput,
        /// Element data type of mask output
        typename ElementMaskOutput_,
        /// Layout of mask output
        typename LayoutMaskOutput_,
        /// Access granularity of mask output in units of elements
        int kAlignmentMaskOutput,
        /// Data type of accumulator
        typename ElementDst_,
        /// Layout of accumulator
        typename LayoutDst_,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Operation performed by Convolution
        typename Operator_>
struct DefaultRegionRestrictedMmaCore<
        Shape_, WarpShape_, gemm::GemmShape<1, 1, 1>, ElementSrc_,
        layout::TensorNCHW, kAlignmentSrc, ElementFilter_, layout::TensorNCHW,
        kAlignmentFilter, ElementMaskInput_, LayoutMaskInput_,
        kAlignmentMaskInput, ElementMaskOutput_, LayoutMaskOutput_,
        kAlignmentMaskOutput, ElementDst_, LayoutDst_, arch::OpClassSimt,
        Stages, Operator_, true, ImplicitGemmMode::GEMM_NT> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<1, 1, 1>;
    using ElementSrc = ElementSrc_;
    using LayoutSrc = layout::TensorNCHW;
    using ElementFilter = ElementFilter_;
    using LayoutFilter = layout::TensorNCHW;
    using ElementMaskInput = ElementMaskInput_;
    using LayoutMaskInput = LayoutMaskInput_;
    using ElementMaskOutput = ElementMaskOutput_;
    using LayoutMaskOutput = LayoutMaskOutput_;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassSimt;
    static int const PartitionsK = Shape::kK / WarpShape::kK;

    /// Default Operator
    using Operator = Operator_;

    /// Number of warps present
    using WarpCount = gemm::GemmShape<Shape::kM / WarpShape::kM,
                                      Shape::kN / WarpShape::kN, PartitionsK>;

    // Divisility requirements
    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    /// Number of threads per warp
    static int const kWarpSize = gemm::warp::WarpSize<arch::OpClassSimt>::value;

    /// Number of threads total
    static int const kThreads = WarpCount::kCount * kWarpSize;

    //
    // Shared memory layouts
    //

    using SmemLayoutSrc = layout::RowMajor;
    using SmemLayoutFilter = layout::ColumnMajor;
    using SmemLayoutMaskInput = SmemLayoutSrc;
    using SmemLayoutMaskOutput = SmemLayoutFilter;

    //
    // Iterators to write to shared memory
    //

    /// ThreadMap of iterator Src
    using IteratorThreadMapSrc = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            kAlignmentSrc>;

    using SmemThreadMapSrc = IteratorThreadMapSrc;

    /// Shared memory iterator to Src Tensor operand
    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementSrc, SmemLayoutSrc, 0,
            SmemThreadMapSrc>;

    /// Policy of iterator Filter
    using IteratorThreadMapFilter = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
            kAlignmentFilter>;

    using SmemThreadMapFilter = IteratorThreadMapFilter;

    /// Shared memory iterator to Filter Tensor operand
    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementFilter, SmemLayoutFilter,
            1, SmemThreadMapFilter>;

    //
    // Warp-level matrix multiply operator
    //

    // Define the warp-level op
    static const int WarpNumThreadsM =
            detail::simt_get_warp_threads_m<WarpShape>();
    static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) &&
                          !(WarpShape::kN % WarpNumThreadsN),
                  "WarpShape must be divisible by ThreadTile shape.");
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsSrc = 128 / sizeof_bits<ElementSrc>::value;
    static const int numElementsFilter =
            128 / sizeof_bits<ElementFilter>::value;
    static const int LaneM = cutlass::const_min(numElementsFilter, ThreadTileM);
    static const int LaneN = cutlass::const_min(numElementsSrc, ThreadTileN);
    // these should have max of thread tile also
    using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 1>;
    using Policy = cutlass::gemm::warp::MmaSimtPolicy<
            cutlass::MatrixShape<WarpNumThreadsM,
                                 WarpNumThreadsN>,             // WarpShape
            cutlass::layout::RowMajorInterleaved<LaneLayout>,  // LaneLayout
            LaneMmaShape>;

    using MmaWarpSimtWrapper = cutlass::gemm::warp::RegionRestrictedMmaSimt<
            WarpShape,             /// Size of the Gemm problem - concept:
                                   /// gemm::GemmShape<> 128, 128, 8
            ElementFilter,         /// Data type of Filter elements
            SmemLayoutFilter,      /// Layout of Filter Tensor's matrix
                                   /// (concept: MatrixLayout)
            ElementSrc,            /// Data type of Src elements
            SmemLayoutSrc,         /// Layout of Src Tensor's matrix (concept:
                                   /// MatrixLayout)
            ElementMaskOutput,     /// Data type of Filter elements
            SmemLayoutMaskOutput,  /// Layout of Filter Tensor's matrix
                                   /// (concept: MatrixLayout)
            ElementMaskInput,      /// Data type of Src elements
            SmemLayoutMaskInput,   /// Layout of Src Tensor's matrix
                                   /// (concept: MatrixLayout)
            ElementDst,            /// Element type of C matrix
            layout::RowMajor,      /// Layout of C matrix (concept:
                                   /// MatrixLayout)
            Policy,      /// Policy describing warp-level MmaSimtOp (concept:
                         /// MmaSimtOp policy)
            PartitionsK  /// Number of partitions along K dimension
            >;           /// Used for partial specialization

    struct MmaWarpSimt : public MmaWarpSimtWrapper {
        using ElementMaskInput = ElementMaskInput_;
        using LayoutMaskInput = typename MmaWarpSimtWrapper::LayoutMaskSecond;
        using IteratorMaskInput =
                typename MmaWarpSimtWrapper::IteratorMaskSecond;
        using FragmentMaskInput =
                typename MmaWarpSimtWrapper::FragmentMaskSecond;

        using ElementMaskOutput = ElementMaskOutput_;
        using LayoutMaskOutput = typename MmaWarpSimtWrapper::LayoutMaskFirst;
        using IteratorMaskOutput =
                typename MmaWarpSimtWrapper::IteratorMaskFirst;
        using FragmentMaskOutput =
                typename MmaWarpSimtWrapper::FragmentMaskFirst;
    };

    /// ThreadMap of iterator mask input
    using IteratorThreadMapMaskInput =
            transform::PitchLinearStripminedThreadMap<
                    layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
                    kAlignmentMaskInput>;

    using SmemThreadMapMaskInput = IteratorThreadMapMaskInput;

    /// Shared memory iterator to mask input
    using SmemIteratorMaskInput = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementMaskInput,
            SmemLayoutMaskInput, 0, SmemThreadMapMaskInput>;

    /// ThreadMap of iterator mask input
    using IteratorThreadMapMaskOutput =
            transform::PitchLinearStripminedThreadMap<
                    layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
                    kAlignmentMaskOutput>;

    using SmemThreadMapMaskOutput = IteratorThreadMapMaskOutput;

    /// Shared memory iterator to mask input
    using SmemIteratorMaskOutput = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementMaskOutput,
            SmemLayoutMaskOutput, 1, SmemThreadMapMaskOutput>;

    /// Policy used to define MmaPipelined
    using MmaPolicy =
            gemm::threadblock::MmaPolicy<MmaWarpSimt, MatrixShape<0, 0>,
                                         MatrixShape<0, 0>, WarpCount::kK>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace conv
}  // namespace cutlass

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
