// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_EXTRAS_ENCODER_DEBUG_DATA_H_
#define LIB_EXTRAS_ENCODER_DEBUG_DATA_H_

#include <memory>
#include <string>
#include <vector>

#include "lib/jxl/enc_debug_data.h"

namespace jxl {
namespace extras {

struct FileEncoderDebugDataSinkOptions {
  std::string output_dir;

  // Empty includes mean all artifacts. Entries are literal artifact-name
  // prefixes; excludes take precedence.
  std::vector<std::string> include_prefixes;
  std::vector<std::string> exclude_prefixes;

  std::string profile = "custom";
  std::string libjxl_revision = "unknown";
};

// Creates a development-only filesystem sink. The directory is created by
// Begin, and Finish writes manifest.json. Each tensor is stored as a separate
// C-contiguous, little-endian .npy file.
std::unique_ptr<EncoderDebugDataSink> CreateFileEncoderDebugDataSink(
    const FileEncoderDebugDataSinkOptions& options);

}  // namespace extras
}  // namespace jxl

#endif  // LIB_EXTRAS_ENCODER_DEBUG_DATA_H_
