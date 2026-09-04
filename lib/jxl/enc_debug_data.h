// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_DEBUG_DATA_H_
#define LIB_JXL_ENC_DEBUG_DATA_H_

// Development-only observations of encoder-native, image-shaped data.

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/status.h"

#ifndef JPEGXL_ENABLE_ENCODER_DEBUG_DATA
#define JPEGXL_ENABLE_ENCODER_DEBUG_DATA 0
#endif

namespace jxl {

enum class DebugDataType : uint8_t {
  kUint8,
  kInt8,
  kUint16,
  kInt16,
  kUint32,
  kInt32,
  kUint64,
  kInt64,
  kFloat32,
  kFloat64,
};

enum class DebugGridKind : uint8_t {
  kPixel,
  kBlock,
  kColorTile,
  kGroup,
  kVariableBlock,
  kOther,
};

struct DebugRect {
  DebugRect() = default;
  DebugRect(int64_t x0_in, int64_t y0_in, size_t xsize_in, size_t ysize_in)
      : x0(x0_in), y0(y0_in), xsize(xsize_in), ysize(ysize_in) {}

  int64_t x0 = 0;
  int64_t y0 = 0;
  size_t xsize = 0;
  size_t ysize = 0;
};

// Describes how the final two tensor axes map to full-frame pixels. The
// footprint is the area represented by one sample. It can differ from spacing
// for overlapping or sparse observations.
struct DebugGridInfo {
  DebugGridKind kind = DebugGridKind::kOther;
  int64_t origin_x = 0;
  int64_t origin_y = 0;
  size_t spacing_x = 1;
  size_t spacing_y = 1;
  size_t footprint_x = 1;
  size_t footprint_y = 1;
  DebugRect valid_rect;
  DebugRect padded_rect;
  bool value_is_block_anchor = false;
};

struct DebugCategory {
  int64_t value = 0;
  const char* name = nullptr;
  size_t covered_blocks_x = 0;
  size_t covered_blocks_y = 0;
};

struct DebugArtifactInfo {
  const char* name = nullptr;
  const char* stage = nullptr;
  int64_t frame_index = 0;
  int64_t iteration = -1;
  DebugGridInfo grid;
  const char* units = nullptr;
  const char* semantic = nullptr;

  const char* const* axes = nullptr;
  size_t num_axes = 0;
  const char* const* channel_names = nullptr;
  size_t num_channel_names = 0;
  const DebugCategory* categories = nullptr;
  size_t num_categories = 0;

  const char* const* derived_from = nullptr;
  size_t num_derived_from = 0;
  const char* formula = nullptr;
};

// A synchronous, non-owning tensor view. Strides are in bytes and permit the
// sink to omit allocator padding without requiring encoder-side copies.
struct DebugTensorView {
  DebugDataType dtype = DebugDataType::kUint8;
  const void* data = nullptr;
  const size_t* shape = nullptr;
  const ptrdiff_t* byte_strides = nullptr;
  size_t rank = 0;
};

struct EncoderDebugRunInfo {
  size_t frame_xsize = 0;
  size_t frame_ysize = 0;
  float butteraugli_distance = 0.0f;
  int effort = 0;
  size_t decoding_speed_tier = 0;
  int resampling = 1;
  bool streaming_mode = false;
  uint32_t color_transform = 0;
  uint32_t input_color_space = 0;
  float intensity_target = 0.0f;
  bool gaborish = false;
  uint32_t epf_iterations = 0;
};

class EncoderDebugDataSink {
 public:
  virtual ~EncoderDebugDataSink() = default;

  virtual Status Begin(const EncoderDebugRunInfo& run_info) = 0;
  virtual bool Wants(const DebugArtifactInfo& info) const = 0;

  // Emit consumes the view synchronously. The encoder retains ownership.
  virtual Status Emit(const DebugArtifactInfo& info,
                      const DebugTensorView& tensor) = 0;
  virtual Status EmitScalar(const DebugArtifactInfo& info, double value) = 0;

  // Finalizes the run-level manifest. No calls are permitted after Finish.
  virtual Status Finish() = 0;
};

const char* DebugDataTypeName(DebugDataType dtype);
size_t DebugDataTypeSize(DebugDataType dtype);
const char* DebugGridKindName(DebugGridKind kind);

}  // namespace jxl

#endif  // LIB_JXL_ENC_DEBUG_DATA_H_
