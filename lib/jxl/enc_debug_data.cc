// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_debug_data.h"

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

}  // namespace jxl
