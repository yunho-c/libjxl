// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_debug_data.h"

#include <cstring>
#include <limits>
#include <vector>

#include "lib/jxl/dec_cache.h"
#include "lib/jxl/enc_debug_data_internal.h"

namespace jxl {

const char* DebugDataTypeName(DebugDataType dtype) {
  switch (dtype) {
    case DebugDataType::kUint8:
      return "uint8";
    case DebugDataType::kInt8:
      return "int8";
    case DebugDataType::kUint16:
      return "uint16";
    case DebugDataType::kInt16:
      return "int16";
    case DebugDataType::kUint32:
      return "uint32";
    case DebugDataType::kInt32:
      return "int32";
    case DebugDataType::kUint64:
      return "uint64";
    case DebugDataType::kInt64:
      return "int64";
    case DebugDataType::kFloat32:
      return "float32";
    case DebugDataType::kFloat64:
      return "float64";
  }
  return "unknown";
}

size_t DebugDataTypeSize(DebugDataType dtype) {
  switch (dtype) {
    case DebugDataType::kUint8:
    case DebugDataType::kInt8:
      return 1;
    case DebugDataType::kUint16:
    case DebugDataType::kInt16:
      return 2;
    case DebugDataType::kUint32:
    case DebugDataType::kInt32:
    case DebugDataType::kFloat32:
      return 4;
    case DebugDataType::kUint64:
    case DebugDataType::kInt64:
    case DebugDataType::kFloat64:
      return 8;
  }
  return 0;
}

const char* DebugGridKindName(DebugGridKind kind) {
  switch (kind) {
    case DebugGridKind::kPixel:
      return "pixel";
    case DebugGridKind::kBlock:
      return "block";
    case DebugGridKind::kColorTile:
      return "color_tile";
    case DebugGridKind::kGroup:
      return "group";
    case DebugGridKind::kVariableBlock:
      return "variable_block";
    case DebugGridKind::kOther:
      return "other";
  }
  return "unknown";
}

#if JPEGXL_ENABLE_ENCODER_DEBUG_DATA

namespace {

template <typename T>
Status EmitDebugPlane(EncoderDebugDataSink* sink, const DebugArtifactInfo& info,
                      const Plane<T>& image, DebugDataType dtype) {
  if (sink == nullptr || !sink->Wants(info)) return true;
  const size_t shape[] = {image.ysize(), image.xsize()};
  const ptrdiff_t strides[] = {static_cast<ptrdiff_t>(image.bytes_per_row()),
                               static_cast<ptrdiff_t>(sizeof(T))};
  DebugTensorView tensor;
  tensor.dtype = dtype;
  tensor.data = image.ysize() == 0 ? nullptr : image.ConstRow(0);
  tensor.shape = shape;
  tensor.byte_strides = strides;
  tensor.rank = 2;
  return sink->Emit(info, tensor);
}

}  // namespace

Status EmitDebugImageF(EncoderDebugDataSink* sink,
                       const DebugArtifactInfo& info, const ImageF& image) {
  return EmitDebugPlane(sink, info, image, DebugDataType::kFloat32);
}

Status EmitDebugImageI(EncoderDebugDataSink* sink,
                       const DebugArtifactInfo& info, const ImageI& image) {
  return EmitDebugPlane(sink, info, image, DebugDataType::kInt32);
}

Status EmitDebugImageB(EncoderDebugDataSink* sink,
                       const DebugArtifactInfo& info, const ImageB& image) {
  return EmitDebugPlane(sink, info, image, DebugDataType::kUint8);
}

Status EmitDebugImageSB(EncoderDebugDataSink* sink,
                        const DebugArtifactInfo& info, const ImageSB& image) {
  return EmitDebugPlane(sink, info, image, DebugDataType::kInt8);
}

Status EmitDebugImage3F(EncoderDebugDataSink* sink,
                        const DebugArtifactInfo& info, const Image3F& image,
                        size_t xsize, size_t ysize) {
  if (sink == nullptr || !sink->Wants(info)) return true;
  JXL_ENSURE(xsize <= image.xsize());
  JXL_ENSURE(ysize <= image.ysize());
  if (xsize != 0 && ysize > std::numeric_limits<size_t>::max() / xsize) {
    return JXL_FAILURE("Debug image dimensions overflow size_t");
  }
  const size_t plane_size = xsize * ysize;
  if (plane_size > std::numeric_limits<size_t>::max() / 3) {
    return JXL_FAILURE("Debug image element count overflows size_t");
  }
  if (plane_size > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) /
                       sizeof(float)) {
    return JXL_FAILURE("Debug image byte stride overflows ptrdiff_t");
  }
  std::vector<float> packed(3 * plane_size);
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      memcpy(packed.data() + c * plane_size + y * xsize,
             image.ConstPlaneRow(c, y), xsize * sizeof(float));
    }
  }
  const size_t shape[] = {3, ysize, xsize};
  const ptrdiff_t strides[] = {
      static_cast<ptrdiff_t>(plane_size * sizeof(float)),
      static_cast<ptrdiff_t>(xsize * sizeof(float)),
      static_cast<ptrdiff_t>(sizeof(float))};
  DebugTensorView tensor;
  tensor.dtype = DebugDataType::kFloat32;
  tensor.data = packed.empty() ? nullptr : packed.data();
  tensor.shape = shape;
  tensor.byte_strides = strides;
  tensor.rank = 3;
  return sink->Emit(info, tensor);
}

namespace {

DebugArtifactInfo RoundtripFilterArtifactInfo(
    const RoundtripFilterDebugData& debug_data, const std::string& name,
    const char* semantic) {
  static const char* const kAxes[] = {"channel", "y", "x"};
  static const char* const kChannels[] = {"x", "y", "b"};
  DebugArtifactInfo info;
  info.name = name.c_str();
  info.stage = "roundtrip_loop_filter";
  info.iteration = debug_data.iteration;
  info.grid = debug_data.grid;
  info.units = "encoder_xyb";
  info.semantic = semantic;
  info.axes = kAxes;
  info.num_axes = 3;
  info.channel_names = kChannels;
  info.num_channel_names = 3;
  return info;
}

bool WantsRoundtripFilterArtifact(EncoderDebugDataSink* sink,
                                  const RoundtripFilterDebugData& debug_data,
                                  const char* leaf, const char* semantic) {
  if (sink == nullptr) return false;
  const std::string name = debug_data.prefix + leaf;
  const DebugArtifactInfo info =
      RoundtripFilterArtifactInfo(debug_data, name, semantic);
  return sink->Wants(info);
}

Status EmitRoundtripFilterArtifact(
    EncoderDebugDataSink* sink, const RoundtripFilterDebugData& debug_data,
    const char* leaf, const char* semantic, const Image3F& image, bool wanted) {
  if (!wanted) return true;
  const std::string name = debug_data.prefix + leaf;
  const DebugArtifactInfo info =
      RoundtripFilterArtifactInfo(debug_data, name, semantic);
  return EmitDebugImage3F(sink, info, image, debug_data.xsize,
                          debug_data.ysize);
}

}  // namespace

void ConfigureRoundtripFilterDebugData(
    EncoderDebugDataSink* sink, const std::string& prefix, int64_t iteration,
    bool gaborish, size_t epf_iterations, const DebugGridInfo& grid,
    size_t xsize, size_t ysize, RenderPipelineDebugTaps* pipeline_taps,
    RoundtripFilterDebugData* debug_data) {
  *pipeline_taps = RenderPipelineDebugTaps{};
  debug_data->prefix = prefix;
  debug_data->iteration = iteration;
  debug_data->grid = grid;
  debug_data->xsize = xsize;
  debug_data->ysize = ysize;

  debug_data->want_before_loop_filter = WantsRoundtripFilterArtifact(
      sink, *debug_data, "before_loop_filter_xyb",
      "Roundtrip reconstruction immediately before decoder loop filtering");
  debug_data->want_after_gaborish =
      gaborish && WantsRoundtripFilterArtifact(
                       sink, *debug_data, "after_gaborish_xyb",
                       "Roundtrip reconstruction after decoder Gaborish");
  debug_data->want_after_epf0 =
      epf_iterations >= 3 &&
      WantsRoundtripFilterArtifact(
          sink, *debug_data, "after_epf0_xyb",
          "Roundtrip reconstruction after decoder EPF stage zero");
  debug_data->want_after_epf1 =
      epf_iterations >= 1 &&
      WantsRoundtripFilterArtifact(
          sink, *debug_data, "after_epf1_xyb",
          "Roundtrip reconstruction after decoder EPF stage one");
  debug_data->want_after_epf2 =
      epf_iterations >= 2 &&
      WantsRoundtripFilterArtifact(
          sink, *debug_data, "after_epf2_xyb",
          "Roundtrip reconstruction after decoder EPF stage two");
  debug_data->want_after_loop_filter = WantsRoundtripFilterArtifact(
      sink, *debug_data, "after_loop_filter_xyb",
      "Roundtrip reconstruction after all decoder loop-filter stages");

  if (debug_data->want_before_loop_filter) {
    pipeline_taps->before_loop_filter = &debug_data->before_loop_filter;
  }
  if (debug_data->want_after_gaborish) {
    pipeline_taps->after_gaborish = &debug_data->after_gaborish;
  }
  if (debug_data->want_after_epf0) {
    pipeline_taps->after_epf0 = &debug_data->after_epf0;
  }
  if (debug_data->want_after_epf1) {
    pipeline_taps->after_epf1 = &debug_data->after_epf1;
  }
  if (debug_data->want_after_epf2) {
    pipeline_taps->after_epf2 = &debug_data->after_epf2;
  }
  if (debug_data->want_after_loop_filter) {
    pipeline_taps->after_loop_filter = &debug_data->after_loop_filter;
  }
}

Status EmitRoundtripFilterDebugData(
    EncoderDebugDataSink* sink,
    const RoundtripFilterDebugData& debug_data) {
  JXL_RETURN_IF_ERROR(EmitRoundtripFilterArtifact(
      sink, debug_data, "before_loop_filter_xyb",
      "Roundtrip reconstruction immediately before decoder loop filtering",
      debug_data.before_loop_filter, debug_data.want_before_loop_filter));
  JXL_RETURN_IF_ERROR(EmitRoundtripFilterArtifact(
      sink, debug_data, "after_gaborish_xyb",
      "Roundtrip reconstruction after decoder Gaborish",
      debug_data.after_gaborish, debug_data.want_after_gaborish));
  JXL_RETURN_IF_ERROR(EmitRoundtripFilterArtifact(
      sink, debug_data, "after_epf0_xyb",
      "Roundtrip reconstruction after decoder EPF stage zero",
      debug_data.after_epf0, debug_data.want_after_epf0));
  JXL_RETURN_IF_ERROR(EmitRoundtripFilterArtifact(
      sink, debug_data, "after_epf1_xyb",
      "Roundtrip reconstruction after decoder EPF stage one",
      debug_data.after_epf1, debug_data.want_after_epf1));
  JXL_RETURN_IF_ERROR(EmitRoundtripFilterArtifact(
      sink, debug_data, "after_epf2_xyb",
      "Roundtrip reconstruction after decoder EPF stage two",
      debug_data.after_epf2, debug_data.want_after_epf2));
  return EmitRoundtripFilterArtifact(
      sink, debug_data, "after_loop_filter_xyb",
      "Roundtrip reconstruction after all decoder loop-filter stages",
      debug_data.after_loop_filter, debug_data.want_after_loop_filter);
}

#endif  // JPEGXL_ENABLE_ENCODER_DEBUG_DATA

}  // namespace jxl
