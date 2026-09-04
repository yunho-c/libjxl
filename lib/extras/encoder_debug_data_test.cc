// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/encoder_debug_data.h"

#include <stdio.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#include <process.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

#include "lib/jxl/enc_debug_data.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace extras {
namespace {

int ProcessId() {
#if defined(_WIN32)
  return _getpid();
#else
  return static_cast<int>(getpid());
#endif
}

void RemoveDirectory(const std::string& path) {
#if defined(_WIN32)
  _rmdir(path.c_str());
#else
  rmdir(path.c_str());
#endif
}

std::vector<uint8_t> ReadFile(const std::string& path) {
  FILE* file = fopen(path.c_str(), "rb");
  EXPECT_NE(nullptr, file);
  if (file == nullptr) return {};
  EXPECT_EQ(0, fseek(file, 0, SEEK_END));
  const long size = ftell(file);
  EXPECT_GE(size, 0);
  EXPECT_EQ(0, fseek(file, 0, SEEK_SET));
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  if (!bytes.empty()) {
    EXPECT_EQ(bytes.size(), fread(bytes.data(), 1, bytes.size(), file));
  }
  EXPECT_EQ(0, fclose(file));
  return bytes;
}

size_t NpyPayloadOffset(const std::vector<uint8_t>& bytes) {
  EXPECT_GE(bytes.size(), 10u);
  if (bytes.size() < 10) return bytes.size();
  EXPECT_EQ(0x93, bytes[0]);
  EXPECT_EQ('N', bytes[1]);
  EXPECT_EQ('U', bytes[2]);
  EXPECT_EQ('M', bytes[3]);
  EXPECT_EQ('P', bytes[4]);
  EXPECT_EQ('Y', bytes[5]);
  EXPECT_EQ(1, bytes[6]);
  EXPECT_EQ(0, bytes[7]);
  const size_t header_size = bytes[8] | (static_cast<size_t>(bytes[9]) << 8);
  EXPECT_EQ(0u, (10 + header_size) % 64);
  return 10 + header_size;
}

class OutputCleanup {
 public:
  explicit OutputCleanup(std::string path) : path_(std::move(path)) {}
  ~OutputCleanup() {
    for (const std::string& file : files_) {
      std::remove((path_ + "/" + file).c_str());
    }
    std::remove((path_ + "/manifest.json").c_str());
    for (auto directory = directories_.rbegin();
         directory != directories_.rend(); ++directory) {
      RemoveDirectory(path_ + "/" + *directory);
    }
    RemoveDirectory(path_);
  }

  void AddFile(const std::string& path) { files_.push_back(path); }
  void AddDirectory(const std::string& path) { directories_.push_back(path); }

 private:
  std::string path_;
  std::vector<std::string> files_;
  std::vector<std::string> directories_;
};

DebugArtifactInfo MakeInfo(const char* name, const char* const* axes,
                           size_t num_axes) {
  DebugArtifactInfo info;
  info.name = name;
  info.stage = "serializer_test";
  info.units = "test_units";
  info.semantic = "Serializer test tensor";
  info.axes = axes;
  info.num_axes = num_axes;
  info.grid.kind = DebugGridKind::kPixel;
  info.grid.valid_rect = DebugRect(3, 5, 3, 2);
  info.grid.padded_rect = DebugRect(3, 5, 5, 2);
  info.grid.origin_x = 3;
  info.grid.origin_y = 5;
  return info;
}

TEST(EncoderDebugDataTest, WritesStridedLittleEndianNpyAndSortedManifest) {
  const std::string output_dir =
      "encoder_debug_data_test_" + std::to_string(ProcessId());
  OutputCleanup cleanup(output_dir);
  cleanup.AddFile("a/strided.npy");
  cleanup.AddFile("z/scalar.npy");
  cleanup.AddDirectory("a");
  cleanup.AddDirectory("z");

  FileEncoderDebugDataSinkOptions options;
  options.output_dir = output_dir;
  options.profile = "phase1-test";
  options.libjxl_revision = "test-revision";
  std::unique_ptr<EncoderDebugDataSink> sink =
      CreateFileEncoderDebugDataSink(options);
  ASSERT_NE(nullptr, sink);

  EncoderDebugRunInfo run_info;
  run_info.frame_xsize = 3;
  run_info.frame_ysize = 2;
  run_info.butteraugli_distance = 1.5f;
  run_info.effort = 7;
  run_info.resampling = 1;
  ASSERT_TRUE(sink->Begin(run_info));

  DebugArtifactInfo scalar_info = MakeInfo("z/scalar", nullptr, 0);
  scalar_info.grid.kind = DebugGridKind::kOther;
  ASSERT_TRUE(sink->EmitScalar(scalar_info, -2.25));

  const float padded[2][5] = {
      {91.0f, 1.0f, 2.0f, 3.0f, 92.0f},
      {93.0f, 4.0f, 5.0f, 6.0f, 94.0f},
  };
  static const char* const kAxes[] = {"y", "x"};
  DebugArtifactInfo info = MakeInfo("a/strided", kAxes, 2);
  const size_t shape[] = {2, 3};
  const ptrdiff_t strides[] = {static_cast<ptrdiff_t>(sizeof(padded[0])),
                               static_cast<ptrdiff_t>(sizeof(float))};
  DebugTensorView tensor;
  tensor.dtype = DebugDataType::kFloat32;
  tensor.data = &padded[0][1];
  tensor.shape = shape;
  tensor.byte_strides = strides;
  tensor.rank = 2;
  ASSERT_TRUE(sink->Emit(info, tensor));
  EXPECT_FALSE(sink->Emit(info, tensor));
  ASSERT_TRUE(sink->Finish());

  const std::vector<uint8_t> npy = ReadFile(output_dir + "/a/strided.npy");
  const size_t payload_offset = NpyPayloadOffset(npy);
  ASSERT_EQ(payload_offset + 6 * sizeof(float), npy.size());
  float values[6] = {};
  memcpy(values, npy.data() + payload_offset, sizeof(values));
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(static_cast<float>(i + 1), values[i]);
  }

  const std::vector<uint8_t> manifest_bytes =
      ReadFile(output_dir + "/manifest.json");
  const std::string manifest(manifest_bytes.begin(), manifest_bytes.end());
  EXPECT_NE(std::string::npos,
            manifest.find("\"schema_version\": {\"major\": 1, \"minor\": 0}"));
  EXPECT_NE(std::string::npos,
            manifest.find("\"valid_rect_px\": [3, 5, 3, 2]"));
  EXPECT_NE(std::string::npos, manifest.find("\"bytes\": 24"));
  EXPECT_LT(manifest.find("\"name\": \"a/strided\""),
            manifest.find("\"name\": \"z/scalar\""));
}

TEST(EncoderDebugDataTest, SupportsAllDtypesAndOneToFourDimensions) {
  struct TypeCase {
    DebugDataType dtype;
    const char* descriptor;
  };
  const TypeCase types[] = {
      {DebugDataType::kUint8, "|u1"},   {DebugDataType::kInt8, "|i1"},
      {DebugDataType::kUint16, "<u2"},  {DebugDataType::kInt16, "<i2"},
      {DebugDataType::kUint32, "<u4"},  {DebugDataType::kInt32, "<i4"},
      {DebugDataType::kUint64, "<u8"},  {DebugDataType::kInt64, "<i8"},
      {DebugDataType::kFloat32, "<f4"}, {DebugDataType::kFloat64, "<f8"},
  };
  const std::string output_dir =
      "encoder_debug_data_types_test_" + std::to_string(ProcessId());
  OutputCleanup cleanup(output_dir);
  cleanup.AddDirectory("types");

  FileEncoderDebugDataSinkOptions options;
  options.output_dir = output_dir;
  std::unique_ptr<EncoderDebugDataSink> sink =
      CreateFileEncoderDebugDataSink(options);
  EncoderDebugRunInfo run_info;
  run_info.frame_xsize = 1;
  run_info.frame_ysize = 1;
  ASSERT_TRUE(sink->Begin(run_info));

  static const char* const kAxes[] = {"a", "b", "c", "d"};
  for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); ++i) {
    const std::string name = "types/type_" + std::to_string(i);
    cleanup.AddFile(name + ".npy");
    DebugArtifactInfo info = MakeInfo(name.c_str(), kAxes, i % 4 + 1);
    info.grid.kind = DebugGridKind::kOther;
    const size_t rank = info.num_axes;
    size_t shape[] = {1, 1, 1, 2};
    size_t first = 4 - rank;
    ptrdiff_t strides[4] = {};
    strides[3] = static_cast<ptrdiff_t>(DebugDataTypeSize(types[i].dtype));
    for (size_t dim = 3; dim > first; --dim) {
      strides[dim - 1] = strides[dim] * static_cast<ptrdiff_t>(shape[dim]);
    }
    uint64_t storage[2] = {0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL};
    if (types[i].dtype == DebugDataType::kFloat32) {
      const uint32_t float_bits[] = {0x7f800000u, 0x7fc00001u};
      memcpy(storage, float_bits, sizeof(float_bits));
    } else if (types[i].dtype == DebugDataType::kFloat64) {
      storage[0] = 0x7ff0000000000000ULL;
      storage[1] = 0x7ff8000000000001ULL;
    }
    DebugTensorView tensor;
    tensor.dtype = types[i].dtype;
    tensor.data = storage;
    tensor.shape = shape + first;
    tensor.byte_strides = strides + first;
    tensor.rank = rank;
    ASSERT_TRUE(sink->Emit(info, tensor));
  }
  ASSERT_TRUE(sink->Finish());

  for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); ++i) {
    const std::string path =
        output_dir + "/types/type_" + std::to_string(i) + ".npy";
    const std::vector<uint8_t> bytes = ReadFile(path);
    const size_t payload_offset = NpyPayloadOffset(bytes);
    const std::string header(bytes.begin() + 10,
                             bytes.begin() + payload_offset);
    EXPECT_NE(std::string::npos, header.find(std::string("'descr': '") +
                                             types[i].descriptor + "'"));
    EXPECT_EQ(2 * DebugDataTypeSize(types[i].dtype),
              bytes.size() - payload_offset);
    std::vector<uint8_t> expected(2 * DebugDataTypeSize(types[i].dtype));
    for (size_t j = 0; j < expected.size(); ++j) {
      expected[j] = static_cast<uint8_t>(j);
    }
    if (types[i].dtype == DebugDataType::kFloat32) {
      expected = {0x00, 0x00, 0x80, 0x7f, 0x01, 0x00, 0xc0, 0x7f};
    } else if (types[i].dtype == DebugDataType::kFloat64) {
      expected = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x7f,
                  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f};
    }
    EXPECT_EQ(expected, std::vector<uint8_t>(bytes.begin() + payload_offset,
                                             bytes.end()));
  }
}

TEST(EncoderDebugDataTest, IncludeAndExcludeFilters) {
  FileEncoderDebugDataSinkOptions options;
  options.output_dir = "unused";
  options.include_prefixes.push_back("aq/");
  options.exclude_prefixes.push_back("aq/deep/");
  std::unique_ptr<EncoderDebugDataSink> sink =
      CreateFileEncoderDebugDataSink(options);
  DebugArtifactInfo info;
  info.name = "aq/initial/quant_field";
  EXPECT_TRUE(sink->Wants(info));
  info.name = "aq/deep/activity";
  EXPECT_FALSE(sink->Wants(info));
  info.name = "color/xyb";
  EXPECT_FALSE(sink->Wants(info));
}

}  // namespace
}  // namespace extras
}  // namespace jxl
