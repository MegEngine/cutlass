# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cutlass_test_unit_add_executable(
  cutlass_test_unit_convolution_device

  BATCH_SOURCES ON
  BATCH_SIZE 4

  simt_int8_chwn4_iconv_sm61.cu
  simt_int8_chwn4_iconv_sm61_perf.cu
  simt_int8_nchw4_iconv_sm61.cu
  simt_int8_s8nchw4_s8nchw4_f32nchw_iconv_sm61.cu
  simt_int8_s8nchw4_s8nchw4_f32nchw_iconv_sm61_perf.cu
  simt_int8_nchw4_iconv_sm61_perf.cu
  simt_int8_s8nchw4_s8nchw4_s8nchw32_iconv_sm61_perf.cu
  simt_int8_s8nchw4_s8nchw4_s8nchw32_iconv_sm61.cu
  simt_int8_s8nchw4_s8nchw4_s4nhwc_iconv_sm75.cu
  simt_int8_s8nchw4_s8nchw4_u4nhwc_iconv_sm75.cu
  simt_int8_s8nchw4_s8nchw4_s4nhwc_iconv_sm75_perf.cu
  convolution_s8nchw32_s8chwn32_s8nchw32_tensor_op_s32_sm75_perf.cu
  convolution_s8nchw32_s8chwn32_s8nchw32_tensor_op_s32_sm75.cu
  convolution_s8nchw32_s8chwn32_s8nchw4_tensor_op_s32_sm75_perf.cu
  convolution_s8nchw32_s8chwn32_s8nchw4_tensor_op_s32_sm75.cu
  convolution_s4nchw64_s4chwn64_s4nchw64_tensor_op_s32_sm75.cu
  convolution_s4nchw64_s4chwn64_s4nchw64_tensor_op_s32_sm75_perf.cu
  convolution_u4nchw64_s4chwn64_s4nchw64_tensor_op_s32_sm75.cu
  conv2d_implicit_gemm_nt_s8ncxhwx_s8cxrskx_s8ncxhwx_s32_sm75
  convolution_s4nhwc_s4nhwcx_s4nhwc_tensor_op_s32_sm75.cu
  convolution_s4nhwc_s4nhwcx_s4nhwc_tensor_op_s32_sm75_perf.cu
)
