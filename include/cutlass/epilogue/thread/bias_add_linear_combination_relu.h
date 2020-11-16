/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
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
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing linear scaling operations used by epilogues. Values
  are clamped before converting to the output element type.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/numeric_array_converter_policy.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/platform/platform.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements then
/// clamps the output before converting to the output element type.
///
/// D = alpha * accumulator + beta * bias + gamma * source
///
template <typename ElementOutput_,  ///< Data type used to load and store
                                    ///< tensors
          int Count,  ///< Number of elements computed per operation
          typename ElementAccumulator_ = ElementOutput_,  ///< Accumulator
                                                          ///< data type
          typename ElementBias_ = ElementOutput_,         ///< Bias data type

          typename ElementCompute_ = ElementOutput_,  ///< Data type used to
                                                      ///< compute linear
                                                      ///< combination
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
          typename Policy = NumericArrayConverterPolicy<
                  ElementOutput_, Count, ElementAccumulator_, ElementBias_,
                  ElementCompute_, Round>>
class BiasAddLinearCombinationRelu {
public:
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementBias = ElementBias_;
    using ElementCompute = ElementCompute_;

    static int const kCount = Count;

    using FragmentOutput = Array<ElementOutput, kCount>;
    using FragmentAccumulator = Array<ElementAccumulator, kCount>;
    using FragmentBias = Array<ElementBias, kCount>;
    using ComputeFragment = Array<ElementCompute, kCount>;
    using ComputeFragmentBias = Array<ElementCompute, kCount>;
    using SourceConverter = typename Policy::SourceConverter;
    using AccumulatorConverter = typename Policy::AccumulatorConverter;
    using BiasConverter = typename Policy::BiasConverter;
    using OutputConverter = typename Policy::OutputConverter;

    static FloatRoundStyle const kRound = Round;

    /// Host-constructable parameters structure
    struct Params {
        ElementCompute alpha;             ///< scales accumulators
        ElementCompute beta;              ///< scales bias tensor
        ElementCompute gamma;             ///< scales source tensor
        ElementCompute threshold;         ///< minimum value that is output
        ElementCompute const* alpha_ptr;  ///< pointer to accumulator scalar -
                                          ///< if not null, loads it from memory
        ElementCompute const* beta_ptr;   ///< pointer to bias scalar - if not
                                          ///< null loads it from memory
        ElementCompute const* gamma_ptr;  ///< pointer to source scalar - if not
                                          ///< null, loads it from memory
        ElementCompute const*
                threshold_ptr;  ///< pointer to threshold scalar - if not null,
                                ///< loads from memory

        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Params()
                : alpha(ElementCompute(1)),
                  beta(ElementCompute(1)),
                  gamma(ElementCompute(0)),
                  threshold(ElementCompute(0)),
                  alpha_ptr(nullptr),
                  beta_ptr(nullptr),
                  gamma_ptr(nullptr),
                  threshold_ptr(nullptr) {}

        CUTLASS_HOST_DEVICE
        Params(ElementCompute alpha, ElementCompute beta, ElementCompute gamma,
               ElementCompute threshold)
                : alpha(alpha),
                  beta(beta),
                  gamma(gamma),
                  threshold(threshold),
                  alpha_ptr(nullptr),
                  beta_ptr(nullptr),
                  gamma_ptr(nullptr),
                  threshold_ptr(nullptr) {}

        CUTLASS_HOST_DEVICE
        Params(ElementCompute const* alpha_ptr, ElementCompute const* beta_ptr,
               ElementCompute const* gamma_ptr,
               ElementCompute const* threshold_ptr)
                : alpha(0),
                  beta(0),
                  gamma(0),
                  alpha_ptr(alpha_ptr),
                  beta_ptr(beta_ptr),
                  gamma_ptr(gamma_ptr),
                  threshold_ptr(threshold_ptr) {}
    };

private:
    //
    // Data members
    //

    ElementCompute alpha_;
    ElementCompute beta_;
    ElementCompute gamma_;
    ElementCompute threshold_;

public:
    /// Constructs the function object, possibly loading from pointers in host
    /// memory
    CUTLASS_HOST_DEVICE
    BiasAddLinearCombinationRelu(Params const& params) {
        alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
        beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
        gamma_ = (params.gamma_ptr ? *params.gamma_ptr : params.gamma);
        threshold_ = (params.threshold_ptr ? *params.threshold_ptr
                                           : params.threshold);
    }

    /// Returns true if source is needed
    CUTLASS_HOST_DEVICE
    bool is_source_needed() const { return gamma_ != ElementCompute(0); }

    /// Computes linear scaling: D = alpha * accumulator + beta * bias + gamma *
    /// source
    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const& accumulator,
                              FragmentBias const& bias,
                              FragmentOutput const& source) const {
        // Convert source to interal compute numeric type
        SourceConverter source_converter;
        AccumulatorConverter accumulator_converter;
        BiasConverter bias_converter;

        ComputeFragment converted_source = source_converter(source);
        ComputeFragment converted_accumulator =
                accumulator_converter(accumulator);
        ComputeFragmentBias converted_bias = bias_converter(bias);

        // Perform binary operations

        ComputeFragment intermediate;

        multiplies<ComputeFragment> mul_add_source;
        multiply_add<ComputeFragment> mul_add_accumulator;
        multiply_add<ComputeFragmentBias> mul_add_bias;
        ReLu<ComputeFragment> relu;

        intermediate =
                mul_add_source(gamma_, converted_source);  // X =  gamma * C
        intermediate =
                mul_add_accumulator(alpha_, converted_accumulator,
                                    intermediate);  // D = alpha * Accum + X
        intermediate = mul_add_bias(beta_, converted_bias,
                                    intermediate);  // D = beta * bias + D

        // Compute threshold optionally
        intermediate = relu(threshold_, intermediate);

        // Convert to destination numeric type
        OutputConverter destination_converter;

        return destination_converter(intermediate);
    }

    /// Computes linear scaling: D = alpha * accumulator
    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const& accumulator,
                              FragmentBias const& bias) const {
        // Convert source to interal compute numeric type
        AccumulatorConverter accumulator_converter;
        BiasConverter bias_converter;

        ComputeFragment converted_accumulator =
                accumulator_converter(accumulator);
        ComputeFragmentBias converted_bias = bias_converter(bias);

        // Perform binary operations

        ComputeFragment intermediate;

        multiplies<ComputeFragment> mul_accumulator;
        multiply_add<ComputeFragmentBias> mul_add_bias;
        ReLu<ComputeFragment> relu;

        intermediate = mul_accumulator(
                alpha_, converted_accumulator);  // D = alpha * Accum
        intermediate = mul_add_bias(beta_, converted_bias,
                                    intermediate);  // D = beta * bias + D
        // Compute threshold optionally
        intermediate = relu(threshold_, intermediate);

        // Convert to destination numeric type
        OutputConverter destination_converter;

        return destination_converter(intermediate);
    }
};

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass
