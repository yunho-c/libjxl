// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_AC_STRATEGY_H_
#define LIB_JXL_ENC_AC_STRATEGY_H_

#include <jxl/memory_manager.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/quant_weights.h"

// `FindBestAcStrategy` uses heuristics to choose which AC strategy should be
// used in each block, as well as the initial quantization field.

namespace jxl {

struct AuxOut;
class AcStrategyImage;

#if JPEGXL_ENABLE_ENCODER_DEBUG_DATA

enum class AcStrategyDebugPhase : uint8_t {
  kInitial8x8 = 0,
  kAlignedMerge = 1,
  kNonAligned16 = 2,
  kNonAligned32 = 3,
};

constexpr size_t kAcStrategyDebugNumPhases = 4;
constexpr size_t kAcStrategyDebugTileBlocks = 64;

struct AcStrategyDebugCost {
  float total_cost = 0.0f;
  float coefficient_cost = 0.0f;
  float nonzero_cost = 0.0f;
  float information_loss = 0.0f;
  float quant_norm = 0.0f;
};

struct AcStrategyDebugCandidate {
  AcStrategyDebugPhase phase = AcStrategyDebugPhase::kInitial8x8;
  uint8_t strategy = 0;
  size_t block_x = 0;
  size_t block_y = 0;
  bool valid = false;
  bool selected = false;
  bool is_merge = false;
  float merge_current_cost = 0.0f;
  AcStrategyDebugCost cost;
};

struct AcStrategyDebugSnapshot {
  AcStrategyDebugPhase phase = AcStrategyDebugPhase::kInitial8x8;
  std::array<uint8_t, kAcStrategyDebugTileBlocks> strategy{};
  std::array<uint8_t, kAcStrategyDebugTileBlocks> priority{};
};

// One worker owns each instance. Records remain tile-local until the caller's
// worker pool has joined, at which point they are assembled for the sink.
struct AcStrategyDebugTile {
  Rect rect;
  std::vector<AcStrategyDebugCandidate> candidates;
  std::vector<AcStrategyDebugSnapshot> snapshots;
};

#endif  // JPEGXL_ENABLE_ENCODER_DEBUG_DATA

// AC strategy selection: utility struct.

struct ACSConfig {
  const DequantMatrices* JXL_RESTRICT dequant;
  const float* JXL_RESTRICT quant_field_row;
  size_t quant_field_stride;
  const float* JXL_RESTRICT masking_field_row;
  size_t masking_field_stride;
  const float* JXL_RESTRICT masking1x1_field_row;
  size_t masking1x1_field_stride;
  size_t mask1x1_xsize;
  const float* JXL_RESTRICT src_rows[3];
  size_t src_stride;
  float info_loss_multiplier;
  float cost_delta;
  float zeros_mul;
  const float& Pixel(size_t c, size_t x, size_t y) const {
    return src_rows[c][y * src_stride + x];
  }
  float Masking(size_t bx, size_t by) const {
    JXL_DASSERT(masking_field_row[by * masking_field_stride + bx] > 0);
    return masking_field_row[by * masking_field_stride + bx];
  }
  const float* MaskingPtr1x1(size_t bx, size_t by) const {
    JXL_DASSERT(masking1x1_field_row[by * masking1x1_field_stride + bx] > 0);
    return &masking1x1_field_row[by * masking1x1_field_stride + bx];
  }
  float Quant(size_t bx, size_t by) const {
    JXL_DASSERT(quant_field_row[by * quant_field_stride + bx] > 0);
    return quant_field_row[by * quant_field_stride + bx];
  }
};

struct AcStrategyHeuristics {
  explicit AcStrategyHeuristics(JxlMemoryManager* memory_manager,
                                const CompressParams& cparams)
      : memory_manager(memory_manager),
        cparams(cparams),
        mem_per_thread(0),
        qmem_per_thread(0) {}
  Status Init(const Image3F& src, const Rect& rect_in,
              const ImageF& quant_field, const ImageF& mask,
              const ImageF& mask1x1, DequantMatrices* matrices);
  Status PrepareForThreads(std::size_t num_threads);
  Status ProcessRect(const Rect& rect, const ColorCorrelationMap& cmap,
                     AcStrategyImage* ac_strategy, size_t thread
#if JPEGXL_ENABLE_ENCODER_DEBUG_DATA
                     ,
                     AcStrategyDebugTile* debug_tile = nullptr
#endif
  );
  Status Finalize(const FrameDimensions& frame_dim,
                  const AcStrategyImage& ac_strategy, AuxOut* aux_out);
  JxlMemoryManager* memory_manager;
  const CompressParams& cparams;
  ACSConfig config = {};
  size_t mem_per_thread;
  AlignedMemory mem;
  size_t qmem_per_thread;
  AlignedMemory qmem;
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_AC_STRATEGY_H_
