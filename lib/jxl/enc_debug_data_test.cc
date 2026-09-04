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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/encode_internal.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

struct CapturedTensor {
  DebugDataType dtype;
  std::vector<size_t> shape;
  std::vector<ptrdiff_t> strides;
  std::vector<uint8_t> values;
  std::vector<std::string> axes;
  std::vector<std::string> channel_names;
  size_t num_categories = 0;
  int64_t iteration = -1;
  DebugGridInfo grid;
};

template <typename T>
T CapturedValue(const CapturedTensor& tensor, size_t index) {
  T value;
  EXPECT_LE((index + 1) * sizeof(T), tensor.values.size());
  memcpy(&value, tensor.values.data() + index * sizeof(T), sizeof(T));
  return value;
}

class RecordingDebugDataSink : public EncoderDebugDataSink {
 public:
  Status Begin(const EncoderDebugRunInfo& run_info) override {
    ++begin_count;
    run = run_info;
    return true;
  }

  bool Wants(const DebugArtifactInfo& info) const override {
    return info.name != nullptr;
  }

  Status Emit(const DebugArtifactInfo& info,
              const DebugTensorView& tensor) override {
    CapturedTensor captured;
    captured.dtype = tensor.dtype;
    if (tensor.rank != 0) {
      captured.shape.assign(tensor.shape, tensor.shape + tensor.rank);
      captured.strides.assign(tensor.byte_strides,
                              tensor.byte_strides + tensor.rank);
    }
    captured.grid = info.grid;
    for (size_t i = 0; i < info.num_axes; ++i) {
      captured.axes.emplace_back(info.axes[i]);
    }
    for (size_t i = 0; i < info.num_channel_names; ++i) {
      captured.channel_names.emplace_back(info.channel_names[i]);
    }
    captured.num_categories = info.num_categories;
    captured.iteration = info.iteration;

    size_t num_elements = 1;
    for (size_t dimension : captured.shape) num_elements *= dimension;
    const size_t element_size = DebugDataTypeSize(tensor.dtype);
    captured.values.resize(num_elements * element_size);
    for (size_t index = 0; index < num_elements; ++index) {
      size_t remainder = index;
      ptrdiff_t source_offset = 0;
      for (size_t dim = tensor.rank; dim-- > 0;) {
        const size_t coordinate = remainder % tensor.shape[dim];
        remainder /= tensor.shape[dim];
        source_offset += coordinate * tensor.byte_strides[dim];
      }
      memcpy(captured.values.data() + index * element_size,
             static_cast<const uint8_t*>(tensor.data) + source_offset,
             element_size);
    }
    tensors[info.name] = std::move(captured);
    return true;
  }

  Status EmitScalar(const DebugArtifactInfo& info, double value) override {
    DebugTensorView tensor;
    tensor.dtype = DebugDataType::kFloat64;
    tensor.data = &value;
    return Emit(info, tensor);
  }
  Status Finish() override {
    ++finish_count;
    return true;
  }

  EncoderDebugRunInfo run;
  size_t begin_count = 0;
  size_t finish_count = 0;
  std::map<std::string, CapturedTensor> tensors;
};

std::vector<uint8_t> EncodeTestImage(EncoderDebugDataSink* sink,
                                     int effort = 7) {
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
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderFrameSettingsSetOption(
                settings, JXL_ENC_FRAME_SETTING_EFFORT, effort));
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
TEST(EncoderDebugDataTest, CapturesEffortSevenFieldsWithoutChangingOutput) {
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

  const std::set<std::string> expected_names = {
      "ac/final/covered_blocks_x",
      "ac/final/covered_blocks_y",
      "ac/final/is_first_block",
      "ac/final/strategy_id",
      "ac/search/candidate_coefficient_cost",
      "ac/search/candidate_information_loss",
      "ac/search/candidate_nonzero_cost",
      "ac/search/candidate_quant_norm",
      "ac/search/candidate_selected",
      "ac/search/candidate_total_cost",
      "ac/search/candidate_valid",
      "ac/search/merge_accepted",
      "ac/search/merge_candidate_cost",
      "ac/search/merge_current_cost",
      "ac/search/phase_executed",
      "ac/search/phase_id",
      "ac/search/priority_after_phase",
      "ac/search/selected_strategy_after_phase",
      "ac/search/strategy_id",
      "aq/final/quant_field",
      "aq/final/raw_quant_field",
      "aq/initial/mask_block",
      "aq/initial/mask_pixel",
      "aq/initial/quant_field",
      "aq/post_ac_adjust/quant_field",
      "aq/post_ac_adjust/raw_quant_field",
      "cfl/base/color_factor",
      "cfl/base/ytob",
      "cfl/base/ytox",
      "cfl/pass_0/ytob_code",
      "cfl/pass_0/ytob_ratio",
      "cfl/pass_0/ytox_code",
      "cfl/pass_0/ytox_ratio",
      "cfl/pass_1/ytob_code",
      "cfl/pass_1/ytob_ratio",
      "cfl/pass_1/ytox_code",
      "cfl/pass_1/ytox_ratio",
      "color/input_encoded",
      "color/xyb_after_transform",
      "color/xyb_after_transform/y",
      "epf/candidate_error",
      "epf/candidate_reconstruction_xyb",
      "epf/candidate_values",
      "epf/selected_sharpness",
      "gaborish/input_xyb",
      "gaborish/output_xyb",
  };
  std::set<std::string> actual_names;
  for (const auto& entry : sink.tensors) actual_names.insert(entry.first);
  EXPECT_EQ(expected_names, actual_names);

  const CapturedTensor& pixel = sink.tensors.at("color/xyb_after_transform/y");
  EXPECT_EQ((std::vector<size_t>{9, 17}), pixel.shape);
  EXPECT_GT(pixel.strides[0], 17 * static_cast<ptrdiff_t>(sizeof(float)));
  EXPECT_EQ(17u, pixel.grid.valid_rect.xsize);
  EXPECT_EQ(9u, pixel.grid.valid_rect.ysize);
  EXPECT_EQ(24u, pixel.grid.padded_rect.xsize);
  EXPECT_EQ(16u, pixel.grid.padded_rect.ysize);

  const CapturedTensor& input = sink.tensors.at("color/input_encoded");
  EXPECT_EQ((std::vector<size_t>{3, 9, 17}), input.shape);
  EXPECT_EQ((std::vector<std::string>{"channel", "y", "x"}), input.axes);
  EXPECT_EQ((std::vector<std::string>{"r", "g", "b"}), input.channel_names);
  const CapturedTensor& xyb = sink.tensors.at("color/xyb_after_transform");
  EXPECT_EQ((std::vector<size_t>{3, 9, 17}), xyb.shape);
  for (size_t i = 0; i < 9 * 17; ++i) {
    EXPECT_EQ(CapturedValue<float>(pixel, i),
              CapturedValue<float>(xyb, 9 * 17 + i));
  }

  const CapturedTensor& gaborish_input = sink.tensors.at("gaborish/input_xyb");
  const CapturedTensor& gaborish_output =
      sink.tensors.at("gaborish/output_xyb");
  EXPECT_EQ((std::vector<size_t>{3, 9, 17}), gaborish_input.shape);
  EXPECT_EQ(gaborish_input.shape, gaborish_output.shape);

  const CapturedTensor& block = sink.tensors.at("aq/initial/quant_field");
  EXPECT_EQ((std::vector<size_t>{2, 3}), block.shape);
  EXPECT_GT(block.strides[0], 3 * static_cast<ptrdiff_t>(sizeof(float)));
  EXPECT_EQ(8u, block.grid.spacing_x);
  EXPECT_EQ(8u, block.grid.spacing_y);
  EXPECT_EQ(17u, block.grid.valid_rect.xsize);
  EXPECT_EQ(9u, block.grid.valid_rect.ysize);
  EXPECT_EQ(24u, block.grid.padded_rect.xsize);
  EXPECT_EQ(16u, block.grid.padded_rect.ysize);
  for (size_t i = 0; i < block.values.size() / sizeof(float); ++i) {
    const float value = CapturedValue<float>(block, i);
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_GT(value, 0.0f);
  }

  const CapturedTensor& final_quant = sink.tensors.at("aq/final/quant_field");
  const CapturedTensor& raw_quant = sink.tensors.at("aq/final/raw_quant_field");
  EXPECT_EQ((std::vector<size_t>{2, 3}), final_quant.shape);
  EXPECT_EQ(DebugDataType::kInt32, raw_quant.dtype);
  EXPECT_EQ(sink.tensors.at("aq/post_ac_adjust/quant_field").values,
            final_quant.values);
  EXPECT_EQ(sink.tensors.at("aq/post_ac_adjust/raw_quant_field").values,
            raw_quant.values);
  for (size_t i = 0; i < raw_quant.values.size() / sizeof(int32_t); ++i) {
    EXPECT_GT(CapturedValue<int32_t>(raw_quant, i), 0);
  }

  const CapturedTensor& strategy = sink.tensors.at("ac/final/strategy_id");
  const CapturedTensor& is_first = sink.tensors.at("ac/final/is_first_block");
  EXPECT_EQ((std::vector<size_t>{2, 3}), strategy.shape);
  EXPECT_EQ(strategy.shape, is_first.shape);
  EXPECT_EQ(AcStrategy::kNumValidStrategies, strategy.num_categories);
  for (size_t i = 0; i < strategy.values.size(); ++i) {
    EXPECT_LT(strategy.values[i], AcStrategy::kNumValidStrategies);
    EXPECT_LE(is_first.values[i], 1);
  }

  const CapturedTensor& ac_valid =
      sink.tensors.at("ac/search/candidate_valid");
  const CapturedTensor& ac_selected =
      sink.tensors.at("ac/search/candidate_selected");
  const CapturedTensor& ac_total =
      sink.tensors.at("ac/search/candidate_total_cost");
  const CapturedTensor& ac_coefficients =
      sink.tensors.at("ac/search/candidate_coefficient_cost");
  const CapturedTensor& ac_nonzeros =
      sink.tensors.at("ac/search/candidate_nonzero_cost");
  const CapturedTensor& ac_loss =
      sink.tensors.at("ac/search/candidate_information_loss");
  EXPECT_EQ((std::vector<size_t>{4, AcStrategy::kNumValidStrategies, 2, 3}),
            ac_valid.shape);
  EXPECT_EQ(ac_valid.shape, ac_selected.shape);
  EXPECT_EQ(ac_valid.shape, ac_total.shape);
  size_t valid_candidates = 0;
  size_t selected_initial_candidates = 0;
  for (size_t i = 0; i < ac_valid.values.size(); ++i) {
    EXPECT_LE(ac_valid.values[i], 1);
    EXPECT_LE(ac_selected.values[i], 1);
    const float total = CapturedValue<float>(ac_total, i);
    if (ac_valid.values[i] == 0) {
      EXPECT_TRUE(std::isnan(total));
      continue;
    }
    ++valid_candidates;
    const float components = CapturedValue<float>(ac_coefficients, i) +
                             CapturedValue<float>(ac_nonzeros, i) +
                             CapturedValue<float>(ac_loss, i);
    EXPECT_TRUE(std::isfinite(total));
    EXPECT_NEAR(total, components,
                1e-5f * std::max(1.0f, std::abs(total)));
    if (i < AcStrategy::kNumValidStrategies * 2 * 3) {
      selected_initial_candidates += ac_selected.values[i];
    }
  }
  EXPECT_GT(valid_candidates, 0u);
  EXPECT_EQ(6u, selected_initial_candidates);

  const CapturedTensor& selected_after_phase =
      sink.tensors.at("ac/search/selected_strategy_after_phase");
  const CapturedTensor& priority_after_phase =
      sink.tensors.at("ac/search/priority_after_phase");
  const CapturedTensor& phase_executed =
      sink.tensors.at("ac/search/phase_executed");
  EXPECT_EQ((std::vector<size_t>{4, 2, 3}), selected_after_phase.shape);
  EXPECT_EQ(selected_after_phase.shape, priority_after_phase.shape);
  EXPECT_EQ((std::vector<size_t>{4}), phase_executed.shape);
  for (uint8_t executed : phase_executed.values) EXPECT_EQ(1, executed);
  EXPECT_EQ((std::vector<size_t>{4}),
            sink.tensors.at("ac/search/phase_id").shape);
  EXPECT_EQ((std::vector<size_t>{AcStrategy::kNumValidStrategies}),
            sink.tensors.at("ac/search/strategy_id").shape);

  const double color_factor =
      CapturedValue<double>(sink.tensors.at("cfl/base/color_factor"), 0);
  const double base_ytox =
      CapturedValue<double>(sink.tensors.at("cfl/base/ytox"), 0);
  const CapturedTensor& ytox_code = sink.tensors.at("cfl/pass_1/ytox_code");
  const CapturedTensor& ytox_ratio = sink.tensors.at("cfl/pass_1/ytox_ratio");
  ASSERT_EQ(ytox_code.values.size(), ytox_ratio.values.size() / sizeof(float));
  for (size_t i = 0; i < ytox_code.values.size(); ++i) {
    const int8_t code = CapturedValue<int8_t>(ytox_code, i);
    EXPECT_FLOAT_EQ(static_cast<float>(base_ytox + code / color_factor),
                    CapturedValue<float>(ytox_ratio, i));
  }

  const CapturedTensor& candidate_values =
      sink.tensors.at("epf/candidate_values");
  const CapturedTensor& candidate_error =
      sink.tensors.at("epf/candidate_error");
  const CapturedTensor& selected = sink.tensors.at("epf/selected_sharpness");
  EXPECT_EQ((std::vector<size_t>{3}), candidate_values.shape);
  EXPECT_EQ((std::vector<size_t>{3, 2, 3}), candidate_error.shape);
  EXPECT_EQ((std::vector<size_t>{2, 3}), selected.shape);
  const CapturedTensor& candidate_reconstruction =
      sink.tensors.at("epf/candidate_reconstruction_xyb");
  EXPECT_EQ((std::vector<size_t>{3, 3, 9, 17}),
            candidate_reconstruction.shape);
  EXPECT_EQ((std::vector<std::string>{"candidate", "channel", "y", "x"}),
            candidate_reconstruction.axes);
  EXPECT_EQ((std::vector<std::string>{"x", "y", "b"}),
            candidate_reconstruction.channel_names);
  const std::set<uint8_t> candidates(candidate_values.values.begin(),
                                     candidate_values.values.end());
  for (uint8_t value : selected.values) {
    EXPECT_NE(candidates.end(), candidates.find(value));
  }
}

TEST(EncoderDebugDataTest,
     CapturesButteraugliAqIterationsWithoutChangingOutput) {
  RecordingDebugDataSink sink;
  const std::vector<uint8_t> traced = EncodeTestImage(&sink, 8);
  ASSERT_TRUE(sink.Finish());
  const std::vector<uint8_t> ordinary = EncodeTestImage(nullptr, 8);
  EXPECT_EQ(ordinary, traced);

  const CapturedTensor& linear = sink.tensors.at("color/linear_srgb");
  EXPECT_EQ(DebugDataType::kFloat32, linear.dtype);
  EXPECT_EQ((std::vector<size_t>{3, 9, 17}), linear.shape);
  EXPECT_EQ((std::vector<std::string>{"r", "g", "b"}), linear.channel_names);
  for (size_t i = 0; i < linear.values.size() / sizeof(float); ++i) {
    EXPECT_TRUE(std::isfinite(CapturedValue<float>(linear, i)));
  }

  const CapturedTensor& post_ac_quant =
      sink.tensors.at("aq/post_ac_adjust/quant_field");
  const CapturedTensor& post_ac_raw =
      sink.tensors.at("aq/post_ac_adjust/raw_quant_field");
  EXPECT_EQ((std::vector<size_t>{2, 3}), post_ac_quant.shape);
  EXPECT_EQ(DebugDataType::kInt32, post_ac_raw.dtype);

  static const char* const kIterationLeaves[] = {
      "butteraugli_diffmap",
      "clamped_high",
      "clamped_low",
      "dc_quantizer",
      "decoded_linear_rgb",
      "encoder_target",
      "error_ratio",
      "initial_field_clamp",
      "quant_field_in",
      "quant_field_lower_bound",
      "quant_field_out",
      "quant_field_pre_update",
      "quant_field_upper_bound",
      "quant_rounding_stall",
      "quant_update_multiplier",
      "raw_quant_field",
      "score",
      "target",
      "tile_distmap",
      "update_applied",
      "update_exponent",
  };
  for (int iteration = 0; iteration < 3; ++iteration) {
    const std::string prefix = "aq/iter_00" + std::to_string(iteration) + "/";
    for (const char* leaf : kIterationLeaves) {
      const auto found = sink.tensors.find(prefix + leaf);
      ASSERT_NE(sink.tensors.end(), found) << leaf;
      EXPECT_EQ(iteration, found->second.iteration) << leaf;
    }

    const CapturedTensor& decoded =
        sink.tensors.at(prefix + "decoded_linear_rgb");
    const CapturedTensor& diffmap =
        sink.tensors.at(prefix + "butteraugli_diffmap");
    const CapturedTensor& tile = sink.tensors.at(prefix + "tile_distmap");
    const CapturedTensor& ratio = sink.tensors.at(prefix + "error_ratio");
    EXPECT_EQ((std::vector<size_t>{3, 9, 17}), decoded.shape);
    EXPECT_EQ((std::vector<std::string>{"r", "g", "b"}), decoded.channel_names);
    EXPECT_EQ((std::vector<size_t>{9, 17}), diffmap.shape);
    EXPECT_EQ((std::vector<size_t>{2, 3}), tile.shape);
    EXPECT_EQ(tile.shape, ratio.shape);
    const double target =
        CapturedValue<double>(sink.tensors.at(prefix + "target"), 0);
    for (size_t j = 0; j < tile.values.size() / sizeof(float); ++j) {
      EXPECT_FLOAT_EQ(CapturedValue<float>(tile, j) / target,
                      CapturedValue<float>(ratio, j));
    }
    const CapturedTensor& multiplier =
        sink.tensors.at(prefix + "quant_update_multiplier");
    if (iteration < 2) {
      const double exponent =
          CapturedValue<double>(sink.tensors.at(prefix + "update_exponent"), 0);
      for (size_t j = 0; j < ratio.values.size() / sizeof(float); ++j) {
        const float error_ratio = CapturedValue<float>(ratio, j);
        const float expected =
            error_ratio <= 1.0f
                ? static_cast<float>(std::pow(error_ratio, exponent))
                : error_ratio;
        EXPECT_FLOAT_EQ(expected, CapturedValue<float>(multiplier, j));
      }
    }

    const CapturedTensor& quant_in = sink.tensors.at(prefix + "quant_field_in");
    const CapturedTensor& quant_pre =
        sink.tensors.at(prefix + "quant_field_pre_update");
    const CapturedTensor& quant_out =
        sink.tensors.at(prefix + "quant_field_out");
    EXPECT_EQ((std::vector<size_t>{2, 3}), quant_in.shape);
    EXPECT_EQ(quant_in.shape, quant_pre.shape);
    EXPECT_EQ(quant_in.shape, quant_out.shape);
    const double lower = CapturedValue<double>(
        sink.tensors.at(prefix + "quant_field_lower_bound"), 0);
    const double upper = CapturedValue<double>(
        sink.tensors.at(prefix + "quant_field_upper_bound"), 0);
    for (size_t j = 0; j < quant_out.values.size() / sizeof(float); ++j) {
      const float value = CapturedValue<float>(quant_out, j);
      EXPECT_TRUE(std::isfinite(value));
      EXPECT_GE(value, lower);
      EXPECT_LE(value, upper);
    }

    for (const char* leaf : {"clamped_low", "clamped_high",
                             "quant_rounding_stall", "initial_field_clamp"}) {
      const CapturedTensor& mask = sink.tensors.at(prefix + leaf);
      EXPECT_EQ(DebugDataType::kUint8, mask.dtype);
      EXPECT_EQ((std::vector<size_t>{2, 3}), mask.shape);
      for (uint8_t value : mask.values) EXPECT_LE(value, 1);
    }
  }

  EXPECT_EQ(sink.tensors.at("aq/iter_000/quant_field_out").values,
            sink.tensors.at("aq/iter_001/quant_field_in").values);
  EXPECT_EQ(sink.tensors.at("aq/iter_001/quant_field_out").values,
            sink.tensors.at("aq/iter_002/quant_field_in").values);
  EXPECT_EQ(sink.tensors.at("aq/iter_002/quant_field_in").values,
            sink.tensors.at("aq/iter_002/quant_field_out").values);
  EXPECT_EQ(sink.tensors.at("aq/iter_002/quant_field_out").values,
            sink.tensors.at("aq/final/quant_field").values);
  EXPECT_DOUBLE_EQ(0.0, CapturedValue<double>(
                            sink.tensors.at("aq/iter_002/update_applied"), 0));
  const CapturedTensor& terminal_multiplier =
      sink.tensors.at("aq/iter_002/quant_update_multiplier");
  for (size_t i = 0; i < terminal_multiplier.values.size() / sizeof(float);
       ++i) {
    EXPECT_FLOAT_EQ(1.0f, CapturedValue<float>(terminal_multiplier, i));
  }
}
#endif

}  // namespace
}  // namespace jxl
