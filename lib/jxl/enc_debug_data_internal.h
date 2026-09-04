// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_DEBUG_DATA_INTERNAL_H_
#define LIB_JXL_ENC_DEBUG_DATA_INTERNAL_H_

#include "lib/jxl/enc_debug_data.h"
#include "lib/jxl/image.h"

namespace jxl {

#if JPEGXL_ENABLE_ENCODER_DEBUG_DATA

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

#endif  // JPEGXL_ENABLE_ENCODER_DEBUG_DATA

}  // namespace jxl

#endif  // LIB_JXL_ENC_DEBUG_DATA_INTERNAL_H_
