// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license in LICENSE.
// Standalone test: compile with JPEGXL_ENABLE_STAGE_PROFILER=1, -pthread,
// and the source/build lib/include directories on the include path.
#include "lib/jxl/enc_stage_profile.h"
#include <cassert>
#include <chrono>
#include <numeric>
#include <thread>

int main() {
  jxl::EncoderStageProfileSink sink;
  {
    jxl::EncoderStageProfileSession session(&sink);
    jxl::EncoderWallTimer root(jxl::EncoderWallStage::kFrameOther);
    {
      jxl::EncoderWallTimer refine(jxl::EncoderWallStage::kQuantizerRefinement);
      assert(jxl::EncoderWallTimer::InRefinement());
      jxl::EncoderWallTimer::RefinementIteration();
      // A worker scope must not enter the caller's additive partition.
      std::thread worker([] {
        jxl::EncoderWallTimer timer(jxl::EncoderWallStage::kAcCflTiles);
      });
      worker.join();
      {
        // Recursive internal frame uses the same sink without double counting.
        jxl::EncoderStageProfileSession nested_session(&sink);
        jxl::EncoderWallTimer nested(jxl::EncoderWallStage::kFrameOther);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    assert(!jxl::EncoderWallTimer::InRefinement());
  }
  assert(!sink.overflowed);
  assert(sink.frame_invocations == 2);
  assert(sink.refinement_iterations == 1);
  assert(sink.wall_invocations[static_cast<size_t>(jxl::EncoderWallStage::kAcCflTiles)] == 0);
  assert(std::accumulate(sink.wall_exclusive_nanoseconds.begin(),
                         sink.wall_exclusive_nanoseconds.end(), uint64_t{0}) == sink.wall_root_nanoseconds);
  for (size_t i = 0; i < jxl::kEncoderWallStageCount; ++i) {
    assert(sink.wall_exclusive_nanoseconds[i] <= sink.wall_inclusive_nanoseconds[i]);
  }
  const auto root = sink.wall_root_nanoseconds;
  {
    jxl::EncoderStageProfileSession disabled(nullptr);
    jxl::EncoderWallTimer timer(jxl::EncoderWallStage::kFrameOther);
  }
  assert(root == sink.wall_root_nanoseconds);
}
