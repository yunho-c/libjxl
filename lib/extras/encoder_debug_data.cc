// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/encoder_debug_data.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#include <sys/stat.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "lib/jxl/base/status.h"

namespace jxl {
namespace extras {
namespace {

constexpr uint32_t kSchemaMajor = 1;
constexpr uint32_t kSchemaMinor = 0;

struct StoredCategory {
  int64_t value;
  std::string name;
  size_t covered_blocks_x;
  size_t covered_blocks_y;
};

struct StoredArtifact {
  std::string name;
  std::string path;
  std::string dtype;
  std::vector<size_t> shape;
  std::vector<std::string> axes;
  std::vector<std::string> channel_names;
  std::vector<StoredCategory> categories;
  std::vector<std::string> derived_from;
  std::string stage;
  std::string units;
  std::string semantic;
  std::string formula;
  int64_t frame_index;
  int64_t iteration;
  DebugGridInfo grid;
  size_t bytes;
  size_t file_bytes;
};

std::string StringOrEmpty(const char* text) {
  return text == nullptr ? std::string() : std::string(text);
}

std::string JoinPath(const std::string& dir, const std::string& basename) {
  if (dir.empty()) return basename;
  const char last = dir.back();
  if (last == '/' || last == '\\') return dir + basename;
  return dir + "/" + basename;
}

bool IsDirectory(const std::string& path) {
#if defined(_WIN32)
  struct _stat info;
  return _stat(path.c_str(), &info) == 0 && (info.st_mode & _S_IFDIR) != 0;
#else
  struct stat info;
  return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
#endif
}

Status MakeOneDirectory(const std::string& path) {
  if (path.empty() || IsDirectory(path)) return true;
#if defined(_WIN32)
  const int result = _mkdir(path.c_str());
#else
  const int result = mkdir(path.c_str(), 0777);
#endif
  if (result == 0 || (errno == EEXIST && IsDirectory(path))) return true;
  return JXL_FAILURE("Failed to create debug data directory %s: %s",
                     path.c_str(), strerror(errno));
}

Status MakeDirectories(const std::string& path) {
  if (path.empty()) return JXL_FAILURE("Debug data output directory is empty");
  std::string normalized = path;
  std::replace(normalized.begin(), normalized.end(), '\\', '/');

  // Create each component. Skip a leading slash and a Windows drive prefix.
  for (size_t i = 1; i <= normalized.size(); ++i) {
    if (i != normalized.size() && normalized[i] != '/') continue;
    std::string component = normalized.substr(0, i);
    if (component.empty() || component == "/" ||
        (component.size() == 2 && component[1] == ':')) {
      continue;
    }
    JXL_RETURN_IF_ERROR(MakeOneDirectory(component));
  }
  return true;
}

bool IsSafeArtifactName(const std::string& name) {
  if (name.empty() || name.front() == '/' || name.front() == '\\' ||
      name.back() == '/' || name.back() == '\\') {
    return false;
  }
  std::string segment;
  for (char c : name) {
    if (c == '/' || c == '\\') {
      if (segment.empty() || segment == "." || segment == "..") return false;
      segment.clear();
      continue;
    }
    const bool safe = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                      (c >= '0' && c <= '9') || c == '_' || c == '-' ||
                      c == '.';
    if (!safe) return false;
    segment.push_back(c);
  }
  return !segment.empty() && segment != "." && segment != "..";
}

std::string JsonEscape(const std::string& input) {
  std::ostringstream output;
  for (unsigned char c : input) {
    switch (c) {
      case '\\':
        output << "\\\\";
        break;
      case '"':
        output << "\\\"";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (c < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned>(c) << std::dec;
        } else {
          output << static_cast<char>(c);
        }
        break;
    }
  }
  return output.str();
}

const char* NpyDescriptor(DebugDataType dtype) {
  switch (dtype) {
    case DebugDataType::kUint8:
      return "|u1";
    case DebugDataType::kInt8:
      return "|i1";
    case DebugDataType::kUint16:
      return "<u2";
    case DebugDataType::kInt16:
      return "<i2";
    case DebugDataType::kUint32:
      return "<u4";
    case DebugDataType::kInt32:
      return "<i4";
    case DebugDataType::kUint64:
      return "<u8";
    case DebugDataType::kInt64:
      return "<i8";
    case DebugDataType::kFloat32:
      return "<f4";
    case DebugDataType::kFloat64:
      return "<f8";
  }
  return nullptr;
}

Status TensorSize(const DebugTensorView& tensor, size_t* num_elements,
                  size_t* num_bytes) {
  const size_t element_size = DebugDataTypeSize(tensor.dtype);
  JXL_ENSURE(element_size != 0);
  if (tensor.rank != 0) {
    JXL_ENSURE(tensor.shape != nullptr);
    JXL_ENSURE(tensor.byte_strides != nullptr);
  }
  size_t elements = 1;
  size_t max_offset = 0;
  for (size_t i = 0; i < tensor.rank; ++i) {
    JXL_ENSURE(tensor.byte_strides[i] >= 0);
    if (tensor.shape[i] != 0 &&
        elements > std::numeric_limits<size_t>::max() / tensor.shape[i]) {
      return JXL_FAILURE("Debug tensor element count overflows size_t");
    }
    elements *= tensor.shape[i];
    if (tensor.shape[i] > 1) {
      const size_t stride = static_cast<size_t>(tensor.byte_strides[i]);
      const size_t count = tensor.shape[i] - 1;
      if (stride != 0 &&
          count > (std::numeric_limits<size_t>::max() - max_offset) / stride) {
        return JXL_FAILURE("Debug tensor byte offset overflows size_t");
      }
      max_offset += count * stride;
    }
  }
  if (elements != 0) JXL_ENSURE(tensor.data != nullptr);
  if (elements > std::numeric_limits<size_t>::max() / element_size) {
    return JXL_FAILURE("Debug tensor byte count overflows size_t");
  }
  if (elements != 0 &&
      max_offset > std::numeric_limits<size_t>::max() - element_size) {
    return JXL_FAILURE("Debug tensor address range overflows size_t");
  }
  *num_elements = elements;
  *num_bytes = elements * element_size;
  return true;
}

std::string NpyShape(const DebugTensorView& tensor) {
  std::ostringstream output;
  output << '(';
  for (size_t i = 0; i < tensor.rank; ++i) {
    if (i != 0) output << ", ";
    output << tensor.shape[i];
  }
  if (tensor.rank == 1) output << ',';
  output << ')';
  return output.str();
}

bool HostIsLittleEndian() {
  const uint16_t value = 1;
  return *reinterpret_cast<const uint8_t*>(&value) == 1;
}

bool WriteElement(FILE* file, const uint8_t* data, size_t element_size) {
  if (element_size == 1 || HostIsLittleEndian()) {
    return fwrite(data, 1, element_size, file) == element_size;
  }
  uint8_t reversed[8];
  for (size_t i = 0; i < element_size; ++i) {
    reversed[i] = data[element_size - i - 1];
  }
  return fwrite(reversed, 1, element_size, file) == element_size;
}

bool WriteTensorElements(FILE* file, const DebugTensorView& tensor, size_t dim,
                         const uint8_t* data, size_t element_size) {
  if (dim == tensor.rank) return WriteElement(file, data, element_size);
  for (size_t i = 0; i < tensor.shape[dim]; ++i) {
    const ptrdiff_t offset =
        static_cast<ptrdiff_t>(i) * tensor.byte_strides[dim];
    if (!WriteTensorElements(file, tensor, dim + 1, data + offset,
                             element_size)) {
      return false;
    }
  }
  return true;
}

Status WriteNpy(const std::string& path, const DebugTensorView& tensor,
                size_t* payload_bytes, size_t* file_bytes) {
  size_t num_elements = 0;
  JXL_RETURN_IF_ERROR(TensorSize(tensor, &num_elements, payload_bytes));
  const char* descriptor = NpyDescriptor(tensor.dtype);
  JXL_ENSURE(descriptor != nullptr);

  std::string header =
      std::string("{'descr': '") + descriptor +
      "', 'fortran_order': False, 'shape': " + NpyShape(tensor) + ", }";
  constexpr size_t kPreambleSize = 10;
  constexpr size_t kAlignment = 64;
  const size_t padding =
      (kAlignment - ((kPreambleSize + header.size() + 1) % kAlignment)) %
      kAlignment;
  header.append(padding, ' ');
  header.push_back('\n');
  if (header.size() > 0xFFFFu) {
    return JXL_FAILURE("Debug .npy header is too large");
  }

  FILE* file = fopen(path.c_str(), "wb");
  if (file == nullptr) {
    return JXL_FAILURE("Failed to open debug tensor %s: %s", path.c_str(),
                       strerror(errno));
  }
  const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  const uint16_t header_size = static_cast<uint16_t>(header.size());
  const uint8_t header_size_le[] = {
      static_cast<uint8_t>(header_size & 0xFF),
      static_cast<uint8_t>((header_size >> 8) & 0xFF)};
  bool ok = fwrite(magic, 1, sizeof(magic), file) == sizeof(magic) &&
            fwrite(header_size_le, 1, sizeof(header_size_le), file) ==
                sizeof(header_size_le) &&
            fwrite(header.data(), 1, header.size(), file) == header.size();
  if (ok && num_elements != 0) {
    ok = WriteTensorElements(file, tensor, 0,
                             static_cast<const uint8_t*>(tensor.data),
                             DebugDataTypeSize(tensor.dtype));
  }
  const int saved_errno = errno;
  if (fclose(file) != 0) {
    return JXL_FAILURE("Failed to close debug tensor %s: %s", path.c_str(),
                       strerror(errno));
  }
  if (!ok) {
    return JXL_FAILURE("Failed to write debug tensor %s: %s", path.c_str(),
                       strerror(saved_errno));
  }
  *file_bytes =
      sizeof(magic) + sizeof(header_size_le) + header.size() + *payload_bytes;
  return true;
}

Status WriteTextFile(const std::string& path, const std::string& text) {
  FILE* file = fopen(path.c_str(), "wb");
  if (file == nullptr) {
    return JXL_FAILURE("Failed to open debug manifest %s: %s", path.c_str(),
                       strerror(errno));
  }
  const bool ok =
      text.empty() || fwrite(text.data(), 1, text.size(), file) == text.size();
  const int saved_errno = errno;
  if (fclose(file) != 0) {
    return JXL_FAILURE("Failed to close debug manifest %s: %s", path.c_str(),
                       strerror(errno));
  }
  if (!ok) {
    return JXL_FAILURE("Failed to write debug manifest %s: %s", path.c_str(),
                       strerror(saved_errno));
  }
  return true;
}

void WriteStringArray(std::ostream& output,
                      const std::vector<std::string>& values) {
  output << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) output << ", ";
    output << '"' << JsonEscape(values[i]) << '"';
  }
  output << ']';
}

void WriteSizeArray(std::ostream& output, const std::vector<size_t>& values) {
  output << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) output << ", ";
    output << values[i];
  }
  output << ']';
}

void WriteRect(std::ostream& output, const DebugRect& rect) {
  output << '[' << rect.x0 << ", " << rect.y0 << ", " << rect.xsize << ", "
         << rect.ysize << ']';
}

class FileEncoderDebugDataSink : public EncoderDebugDataSink {
 public:
  explicit FileEncoderDebugDataSink(FileEncoderDebugDataSinkOptions options)
      : options_(std::move(options)) {}

  Status Begin(const EncoderDebugRunInfo& run_info) override {
    std::lock_guard<std::mutex> lock(mutex_);
    JXL_ENSURE(!begun_);
    JXL_ENSURE(!finished_);
    JXL_RETURN_IF_ERROR(MakeDirectories(options_.output_dir));
    run_info_ = run_info;
    begun_ = true;
    return true;
  }

  bool Wants(const DebugArtifactInfo& info) const override {
    if (info.name == nullptr) return false;
    const size_t name_size = strlen(info.name);
    const auto matches = [&info, name_size](const std::string& prefix) {
      return prefix.size() <= name_size &&
             memcmp(info.name, prefix.data(), prefix.size()) == 0;
    };
    for (const std::string& prefix : options_.exclude_prefixes) {
      if (matches(prefix)) return false;
    }
    if (options_.include_prefixes.empty()) return true;
    for (const std::string& prefix : options_.include_prefixes) {
      if (matches(prefix)) return true;
    }
    return false;
  }

  Status Emit(const DebugArtifactInfo& info,
              const DebugTensorView& tensor) override {
    if (!Wants(info)) return true;
    std::lock_guard<std::mutex> lock(mutex_);
    JXL_ENSURE(begun_);
    JXL_ENSURE(!finished_);
    JXL_ENSURE(info.name != nullptr);
    const std::string name(info.name);
    if (!IsSafeArtifactName(name)) {
      return JXL_FAILURE("Unsafe debug artifact name: %s", name.c_str());
    }
    if (!artifact_names_.insert(name).second) {
      return JXL_FAILURE("Duplicate debug artifact name: %s", name.c_str());
    }
    JXL_ENSURE(info.num_axes == tensor.rank);
    if (info.num_axes != 0) JXL_ENSURE(info.axes != nullptr);
    if (info.num_channel_names != 0) {
      JXL_ENSURE(info.channel_names != nullptr);
      JXL_ENSURE(tensor.rank != 0);
      JXL_ENSURE(info.num_channel_names == tensor.shape[0]);
    }
    if (info.num_categories != 0) JXL_ENSURE(info.categories != nullptr);
    if (info.num_derived_from != 0) JXL_ENSURE(info.derived_from != nullptr);
    JXL_ENSURE(info.grid.spacing_x != 0);
    JXL_ENSURE(info.grid.spacing_y != 0);
    JXL_ENSURE(info.grid.footprint_x != 0);
    JXL_ENSURE(info.grid.footprint_y != 0);

    const std::string relative_path = name + ".npy";
    const size_t slash = relative_path.find_last_of("/\\");
    if (slash != std::string::npos) {
      JXL_RETURN_IF_ERROR(MakeDirectories(
          JoinPath(options_.output_dir, relative_path.substr(0, slash))));
    }
    const std::string full_path = JoinPath(options_.output_dir, relative_path);
    size_t payload_bytes = 0;
    size_t file_bytes = 0;
    JXL_RETURN_IF_ERROR(
        WriteNpy(full_path, tensor, &payload_bytes, &file_bytes));

    StoredArtifact artifact;
    artifact.name = name;
    artifact.path = relative_path;
    artifact.dtype = DebugDataTypeName(tensor.dtype);
    if (tensor.rank != 0) {
      artifact.shape.assign(tensor.shape, tensor.shape + tensor.rank);
    }
    for (size_t i = 0; i < info.num_axes; ++i) {
      artifact.axes.emplace_back(StringOrEmpty(info.axes[i]));
    }
    for (size_t i = 0; i < info.num_channel_names; ++i) {
      artifact.channel_names.emplace_back(StringOrEmpty(info.channel_names[i]));
    }
    for (size_t i = 0; i < info.num_categories; ++i) {
      const DebugCategory& category = info.categories[i];
      artifact.categories.push_back(
          StoredCategory{category.value, StringOrEmpty(category.name),
                         category.covered_blocks_x, category.covered_blocks_y});
    }
    for (size_t i = 0; i < info.num_derived_from; ++i) {
      artifact.derived_from.emplace_back(StringOrEmpty(info.derived_from[i]));
    }
    artifact.stage = StringOrEmpty(info.stage);
    artifact.units = StringOrEmpty(info.units);
    artifact.semantic = StringOrEmpty(info.semantic);
    artifact.formula = StringOrEmpty(info.formula);
    artifact.frame_index = info.frame_index;
    artifact.iteration = info.iteration;
    artifact.grid = info.grid;
    artifact.bytes = payload_bytes;
    artifact.file_bytes = file_bytes;
    artifacts_.push_back(std::move(artifact));
    return true;
  }

  Status EmitScalar(const DebugArtifactInfo& info, double value) override {
    DebugTensorView tensor;
    tensor.dtype = DebugDataType::kFloat64;
    tensor.data = &value;
    return Emit(info, tensor);
  }

  Status Finish() override {
    std::lock_guard<std::mutex> lock(mutex_);
    JXL_ENSURE(begun_);
    JXL_ENSURE(!finished_);
    std::sort(artifacts_.begin(), artifacts_.end(),
              [](const StoredArtifact& a, const StoredArtifact& b) {
                return a.name < b.name;
              });

    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "{\n";
    output << "  \"schema_version\": {\"major\": " << kSchemaMajor
           << ", \"minor\": " << kSchemaMinor << "},\n";
    output << "  \"libjxl_revision\": \""
           << JsonEscape(options_.libjxl_revision) << "\",\n";
    output << "  \"profile\": \"" << JsonEscape(options_.profile) << "\",\n";
    output << "  \"frame\": {\"xsize\": " << run_info_.frame_xsize
           << ", \"ysize\": " << run_info_.frame_ysize << "},\n";
    output << "  \"encoder\": {\n";
    output << "    \"distance\": " << run_info_.butteraugli_distance << ",\n";
    output << "    \"effort\": " << run_info_.effort << ",\n";
    output << "    \"decoding_speed_tier\": " << run_info_.decoding_speed_tier
           << ",\n";
    output << "    \"resampling\": " << run_info_.resampling << ",\n";
    output << "    \"streaming_mode\": "
           << (run_info_.streaming_mode ? "true" : "false") << ",\n";
    output << "    \"color_transform\": " << run_info_.color_transform << ",\n";
    output << "    \"input_color_space\": " << run_info_.input_color_space
           << ",\n";
    output << "    \"intensity_target\": " << run_info_.intensity_target
           << ",\n";
    output << "    \"gaborish\": " << (run_info_.gaborish ? "true" : "false")
           << ",\n";
    output << "    \"epf_iterations\": " << run_info_.epf_iterations << "\n";
    output << "  },\n";
    output << "  \"artifacts\": [\n";
    for (size_t i = 0; i < artifacts_.size(); ++i) {
      const StoredArtifact& artifact = artifacts_[i];
      output << "    {\n";
      output << "      \"name\": \"" << JsonEscape(artifact.name) << "\",\n";
      output << "      \"path\": \"" << JsonEscape(artifact.path) << "\",\n";
      output << "      \"stage\": \"" << JsonEscape(artifact.stage) << "\",\n";
      output << "      \"dtype\": \"" << artifact.dtype << "\",\n";
      output << "      \"shape\": ";
      WriteSizeArray(output, artifact.shape);
      output << ",\n      \"axes\": ";
      WriteStringArray(output, artifact.axes);
      output << ",\n";
      output << "      \"grid\": {\n";
      output << "        \"kind\": \"" << DebugGridKindName(artifact.grid.kind)
             << "\",\n";
      output << "        \"origin_px\": [" << artifact.grid.origin_x << ", "
             << artifact.grid.origin_y << "],\n";
      output << "        \"spacing_px\": [" << artifact.grid.spacing_x << ", "
             << artifact.grid.spacing_y << "],\n";
      output << "        \"footprint_px\": [" << artifact.grid.footprint_x
             << ", " << artifact.grid.footprint_y << "],\n";
      output << "        \"valid_rect_px\": ";
      WriteRect(output, artifact.grid.valid_rect);
      output << ",\n        \"padded_rect_px\": ";
      WriteRect(output, artifact.grid.padded_rect);
      output << ",\n        \"value_is_block_anchor\": "
             << (artifact.grid.value_is_block_anchor ? "true" : "false")
             << "\n      },\n";
      output << "      \"units\": \"" << JsonEscape(artifact.units) << "\",\n";
      output << "      \"semantic\": \"" << JsonEscape(artifact.semantic)
             << "\",\n";
      output << "      \"frame_index\": " << artifact.frame_index << ",\n";
      if (artifact.iteration >= 0) {
        output << "      \"iteration\": " << artifact.iteration << ",\n";
      }
      if (!artifact.channel_names.empty()) {
        output << "      \"channel_names\": ";
        WriteStringArray(output, artifact.channel_names);
        output << ",\n";
      }
      if (!artifact.categories.empty()) {
        output << "      \"categories\": [\n";
        for (size_t j = 0; j < artifact.categories.size(); ++j) {
          const StoredCategory& category = artifact.categories[j];
          output << "        {\"value\": " << category.value << ", \"name\": \""
                 << JsonEscape(category.name) << "\", \"covered_blocks\": ["
                 << category.covered_blocks_x << ", "
                 << category.covered_blocks_y << "]}";
          if (j + 1 != artifact.categories.size()) output << ',';
          output << '\n';
        }
        output << "      ],\n";
      }
      if (!artifact.derived_from.empty()) {
        output << "      \"derived_from\": ";
        WriteStringArray(output, artifact.derived_from);
        output << ",\n      \"formula\": \"" << JsonEscape(artifact.formula)
               << "\",\n";
      }
      output << "      \"bytes\": " << artifact.bytes << ",\n";
      output << "      \"file_bytes\": " << artifact.file_bytes << "\n";
      output << "    }";
      if (i + 1 != artifacts_.size()) output << ',';
      output << '\n';
    }
    output << "  ]\n}\n";
    JXL_RETURN_IF_ERROR(WriteTextFile(
        JoinPath(options_.output_dir, "manifest.json"), output.str()));
    finished_ = true;
    return true;
  }

 private:
  FileEncoderDebugDataSinkOptions options_;
  EncoderDebugRunInfo run_info_;
  bool begun_ = false;
  bool finished_ = false;
  std::set<std::string> artifact_names_;
  std::vector<StoredArtifact> artifacts_;
  mutable std::mutex mutex_;
};

}  // namespace

std::unique_ptr<EncoderDebugDataSink> CreateFileEncoderDebugDataSink(
    const FileEncoderDebugDataSinkOptions& options) {
  return std::unique_ptr<EncoderDebugDataSink>(
      new FileEncoderDebugDataSink(options));
}

}  // namespace extras
}  // namespace jxl
