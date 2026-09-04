// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_DEBUG_DATA_INTERNAL_H_
#define LIB_JXL_ENC_DEBUG_DATA_INTERNAL_H_

#include <cstdint>
#include <string>

#include "lib/jxl/enc_debug_data.h"
#include "lib/jxl/image.h"

namespace jxl {

#if JPEGXL_ENABLE_ENCODER_DEBUG_DATA

struct RenderPipelineDebugTaps;

// These helpers preserve the logical image extent while honoring padded row
// strides. Image3 planes are separate allocations, so the three-plane helper
// makes a channel-first diagnostic copy only when the artifact is requested.
Status EmitDebugImageF(EncoderDebugDataSink* sink,
                       const DebugArtifactInfo& info, const ImageF& image);
Status EmitDebugImageI(EncoderDebugDataSink* sink,
                       const DebugArtifactInfo& info, const ImageI& image);
Status EmitDebugImageB(EncoderDebugDataSink* sink,
                       const DebugArtifactInfo& info, const ImageB& image);
Status EmitDebugImageSB(EncoderDebugDataSink* sink,
                        const DebugArtifactInfo& info, const ImageSB& image);
Status EmitDebugImage3F(EncoderDebugDataSink* sink,
                        const DebugArtifactInfo& info, const Image3F& image,
                        size_t xsize, size_t ysize);

struct RoundtripFilterDebugData {
  std::string prefix;
  int64_t iteration = -1;
  DebugGridInfo grid;
  size_t xsize = 0;
  size_t ysize = 0;
  Image3F before_loop_filter;
  Image3F after_gaborish;
  Image3F after_epf0;
  Image3F after_epf1;
  Image3F after_epf2;
  Image3F after_loop_filter;
  bool want_before_loop_filter = false;
  bool want_after_gaborish = false;
  bool want_after_epf0 = false;
  bool want_after_epf1 = false;
  bool want_after_epf2 = false;
  bool want_after_loop_filter = false;
};

// Configures non-mutating render-pipeline write stages only for artifacts the
// sink requests. The caller emits the resulting images after rendering joins.
void ConfigureRoundtripFilterDebugData(
    EncoderDebugDataSink* sink, const std::string& prefix, int64_t iteration,
    bool gaborish, size_t epf_iterations, const DebugGridInfo& grid,
    size_t xsize, size_t ysize, RenderPipelineDebugTaps* pipeline_taps,
    RoundtripFilterDebugData* debug_data);
Status EmitRoundtripFilterDebugData(EncoderDebugDataSink* sink,
                                    const RoundtripFilterDebugData& debug_data);

#endif  // JPEGXL_ENABLE_ENCODER_DEBUG_DATA

}  // namespace jxl

#endif  // LIB_JXL_ENC_DEBUG_DATA_INTERNAL_H_
