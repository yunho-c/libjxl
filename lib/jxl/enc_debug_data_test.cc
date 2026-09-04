// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_debug_data.h"

#include <jxl/codestream_header.h>
#include <jxl/color_encoding.h>
#include <jxl/encode.h>
#include <jxl/encode_cxx.h>
#include <jxl/types.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/encode_internal.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

struct CapturedTensor {
  DebugDataType dtype;
  std::vector<size_t> shape;
  std::vector<ptrdiff_t> strides;
  std::vector<float> values;
  DebugGridInfo grid;
};

class RecordingDebugDataSink : public EncoderDebugDataSink {
 public:
  Status Begin(const EncoderDebugRunInfo& run_info) override {
    ++begin_count;
    run = run_info;
    return true;
  }

  bool Wants(const DebugArtifactInfo& info) const override {
    return info.name != nullptr &&
           (strcmp(info.name, "color/xyb_after_transform/y") == 0 ||
            strcmp(info.name, "aq/initial/quant_field") == 0);
  }

  Status Emit(const DebugArtifactInfo& info,
              const DebugTensorView& tensor) override {
    JXL_ENSURE(tensor.dtype == DebugDataType::kFloat32);
    JXL_ENSURE(tensor.rank == 2);
    CapturedTensor captured;
    captured.dtype = tensor.dtype;
    captured.shape.assign(tensor.shape, tensor.shape + tensor.rank);
    captured.strides.assign(tensor.byte_strides,
                            tensor.byte_strides + tensor.rank);
    captured.grid = info.grid;
    for (size_t y = 0; y < tensor.shape[0]; ++y) {
      const uint8_t* row =
          static_cast<const uint8_t*>(tensor.data) + y * tensor.byte_strides[0];
      for (size_t x = 0; x < tensor.shape[1]; ++x) {
        float value;
        memcpy(&value, row + x * tensor.byte_strides[1], sizeof(value));
        captured.values.push_back(value);
      }
    }
    tensors[info.name] = std::move(captured);
    return true;
  }

  Status EmitScalar(const DebugArtifactInfo&, double) override { return true; }
  Status Finish() override {
    ++finish_count;
    return true;
  }

  EncoderDebugRunInfo run;
  size_t begin_count = 0;
  size_t finish_count = 0;
  std::map<std::string, CapturedTensor> tensors;
};

std::vector<uint8_t> EncodeTestImage(EncoderDebugDataSink* sink) {
  constexpr size_t kXSize = 17;
  constexpr size_t kYSize = 9;
  JxlEncoderPtr encoder = JxlEncoderMake(nullptr);
  EXPECT_NE(nullptr, encoder.get());

  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = kXSize;
  info.ysize = kYSize;
  info.bits_per_sample = 8;
  info.exponent_bits_per_sample = 0;
  info.num_color_channels = 3;
  info.uses_original_profile = JXL_FALSE;
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetBasicInfo(encoder.get(), &info));
  JxlColorEncoding color_encoding;
  JxlColorEncodingSetToSRGB(&color_encoding, JXL_FALSE);
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderSetColorEncoding(encoder.get(), &color_encoding));

  JxlEncoderFrameSettings* settings =
      JxlEncoderFrameSettingsCreate(encoder.get(), nullptr);
  EXPECT_NE(nullptr, settings);
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetFrameDistance(settings, 1.0f));
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderFrameSettingsSetOption(
                                 settings, JXL_ENC_FRAME_SETTING_EFFORT, 7));
  if (sink != nullptr) JxlEncoderSetDebugDataSink(settings, sink);

  std::vector<uint8_t> pixels(kXSize * kYSize * 3);
  for (size_t y = 0; y < kYSize; ++y) {
    for (size_t x = 0; x < kXSize; ++x) {
      pixels[3 * (y * kXSize + x) + 0] = static_cast<uint8_t>(3 * x + y);
      pixels[3 * (y * kXSize + x) + 1] = static_cast<uint8_t>(x + 5 * y);
      pixels[3 * (y * kXSize + x) + 2] = static_cast<uint8_t>(2 * x + 7 * y);
    }
  }
  const JxlPixelFormat format = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  EXPECT_EQ(
      JXL_ENC_SUCCESS,
      JxlEncoderAddImageFrame(settings, &format, pixels.data(), pixels.size()));
  JxlEncoderCloseInput(encoder.get());

  std::vector<uint8_t> compressed(64);
  uint8_t* next_out = compressed.data();
  size_t avail_out = compressed.size();
  for (;;) {
    const JxlEncoderStatus status =
        JxlEncoderProcessOutput(encoder.get(), &next_out, &avail_out);
    if (status == JXL_ENC_SUCCESS) break;
    EXPECT_EQ(JXL_ENC_NEED_MORE_OUTPUT, status);
    if (status != JXL_ENC_NEED_MORE_OUTPUT) return {};
    const size_t used = next_out - compressed.data();
    compressed.resize(compressed.size() * 2);
    next_out = compressed.data() + used;
    avail_out = compressed.size() - used;
  }
  compressed.resize(next_out - compressed.data());
  return compressed;
}

#if JPEGXL_ENABLE_ENCODER_DEBUG_DATA
TEST(EncoderDebugDataTest, CapturesPixelAndBlockFieldsWithoutChangingOutput) {
  RecordingDebugDataSink sink;
  const std::vector<uint8_t> traced = EncodeTestImage(&sink);
  ASSERT_TRUE(sink.Finish());
  const std::vector<uint8_t> ordinary = EncodeTestImage(nullptr);
  EXPECT_EQ(ordinary, traced);

  ASSERT_EQ(1u, sink.begin_count);
  ASSERT_EQ(1u, sink.finish_count);
  EXPECT_EQ(17u, sink.run.frame_xsize);
  EXPECT_EQ(9u, sink.run.frame_ysize);
  EXPECT_EQ(7, sink.run.effort);
  EXPECT_FALSE(sink.run.streaming_mode);

  ASSERT_EQ(2u, sink.tensors.size());
  const CapturedTensor& pixel = sink.tensors.at("color/xyb_after_transform/y");
  EXPECT_EQ((std::vector<size_t>{9, 17}), pixel.shape);
  EXPECT_GT(pixel.strides[0], 17 * static_cast<ptrdiff_t>(sizeof(float)));
  EXPECT_EQ(17u, pixel.grid.valid_rect.xsize);
  EXPECT_EQ(9u, pixel.grid.valid_rect.ysize);
  EXPECT_EQ(24u, pixel.grid.padded_rect.xsize);
  EXPECT_EQ(16u, pixel.grid.padded_rect.ysize);

  const CapturedTensor& block = sink.tensors.at("aq/initial/quant_field");
  EXPECT_EQ((std::vector<size_t>{2, 3}), block.shape);
  EXPECT_GT(block.strides[0], 3 * static_cast<ptrdiff_t>(sizeof(float)));
  EXPECT_EQ(8u, block.grid.spacing_x);
  EXPECT_EQ(8u, block.grid.spacing_y);
  EXPECT_EQ(17u, block.grid.valid_rect.xsize);
  EXPECT_EQ(9u, block.grid.valid_rect.ysize);
  EXPECT_EQ(24u, block.grid.padded_rect.xsize);
  EXPECT_EQ(16u, block.grid.padded_rect.ysize);
  for (float value : block.values) {
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_GT(value, 0.0f);
  }
}
#endif

}  // namespace
}  // namespace jxl
