// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_STAGE_PROFILE_H_
#define LIB_JXL_ENC_STAGE_PROFILE_H_

#include <jxl/encode.h>
#include <jxl/jxl_export.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>

#ifndef JPEGXL_ENABLE_STAGE_PROFILER
#define JPEGXL_ENABLE_STAGE_PROFILER 0
#endif

namespace jxl {

// Stable identifiers for the benchmark-only encoder stage profiler. These are
// intentionally independent of implementation function names.
enum class EncoderProfilePhase : size_t {
  kCoefficientTokenization,
  kEntropyModelConstruction,
  kModelAndTokenEmission,
  kFramingAndAssembly,
  kCompleteSerializer,
  kCount,
};

enum class EncoderProfileWork : size_t {
  kCoefficientTokenization,
  kHistogramPopulation,
  kHistogramClustering,
  kHybridUintSelection,
  kEntropyModelConstruction,
  kHistogramSerialization,
  kTokenEncodingAndBitWriting,
  kModularAndDcSideDataEncoding,
  kOutputAssemblyAndCopying,
  kCount,
};

enum class EncoderProfileCount : size_t {
  kTokenCount,
  kHistogramCount,
  kModelBits,
  kTokenBits,
  kOutputBytes,
  kCount,
};

constexpr size_t kEncoderProfilePhaseCount =
    static_cast<size_t>(EncoderProfilePhase::kCount);
constexpr size_t kEncoderProfileWorkCount =
    static_cast<size_t>(EncoderProfileWork::kCount);
constexpr size_t kEncoderProfileCountCount =
    static_cast<size_t>(EncoderProfileCount::kCount);

struct EncoderStageProfileAccumulator {
  std::array<uint64_t, kEncoderProfilePhaseCount> phase_nanoseconds{};
  std::array<uint64_t, kEncoderProfileWorkCount> work_nanoseconds{};
  std::array<uint64_t, kEncoderProfilePhaseCount> phase_invocations{};
  std::array<uint64_t, kEncoderProfileWorkCount> work_invocations{};
  std::array<uint64_t, kEncoderProfileCountCount> counts{};
  bool overflowed = false;
};

// The comparison harness owns and serializes this sink. The library only
// accumulates numeric measurements into it.
struct EncoderStageProfileSink : public EncoderStageProfileAccumulator {};

#if JPEGXL_ENABLE_STAGE_PROFILER

namespace stage_profile_internal {

inline thread_local EncoderStageProfileAccumulator* current = nullptr;
inline thread_local EncoderStageProfileAccumulator* session = nullptr;

inline void AddChecked(uint64_t value, uint64_t* destination,
                       bool* overflowed) {
  if (value > std::numeric_limits<uint64_t>::max() - *destination) {
    *destination = std::numeric_limits<uint64_t>::max();
    *overflowed = true;
    return;
  }
  *destination += value;
}

inline void Merge(const EncoderStageProfileAccumulator& source,
                  EncoderStageProfileAccumulator* destination) {
  for (size_t i = 0; i < source.phase_nanoseconds.size(); ++i) {
    AddChecked(source.phase_nanoseconds[i], &destination->phase_nanoseconds[i],
               &destination->overflowed);
    AddChecked(source.phase_invocations[i], &destination->phase_invocations[i],
               &destination->overflowed);
  }
  for (size_t i = 0; i < source.work_nanoseconds.size(); ++i) {
    AddChecked(source.work_nanoseconds[i], &destination->work_nanoseconds[i],
               &destination->overflowed);
    AddChecked(source.work_invocations[i], &destination->work_invocations[i],
               &destination->overflowed);
  }
  for (size_t i = 0; i < source.counts.size(); ++i) {
    AddChecked(source.counts[i], &destination->counts[i],
               &destination->overflowed);
  }
  destination->overflowed |= source.overflowed;
}

inline uint64_t ElapsedNanoseconds(
    std::chrono::steady_clock::time_point begin) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - begin)
          .count());
}

}  // namespace stage_profile_internal

class EncoderStageProfileScope {
 public:
  explicit EncoderStageProfileScope(EncoderStageProfileAccumulator* current)
      : previous_(stage_profile_internal::current) {
    stage_profile_internal::current = current;
  }
  ~EncoderStageProfileScope() { stage_profile_internal::current = previous_; }

  EncoderStageProfileScope(const EncoderStageProfileScope&) = delete;
  EncoderStageProfileScope& operator=(const EncoderStageProfileScope&) = delete;

 private:
  EncoderStageProfileAccumulator* previous_;
};

class EncoderStageProfileSession {
 public:
  explicit EncoderStageProfileSession(EncoderStageProfileSink* sink)
      : sink_(sink), previous_session_(stage_profile_internal::session) {
    stage_profile_internal::session = sink == nullptr ? nullptr : &accumulator_;
  }
  ~EncoderStageProfileSession() {
    stage_profile_internal::session = previous_session_;
    if (sink_ == nullptr) return;
    const size_t complete =
        static_cast<size_t>(EncoderProfilePhase::kCompleteSerializer);
    accumulator_.phase_invocations[complete] =
        accumulator_.phase_nanoseconds[complete] == 0 ? 0 : 1;
    stage_profile_internal::Merge(accumulator_, sink_);
  }

  EncoderStageProfileSession(const EncoderStageProfileSession&) = delete;
  EncoderStageProfileSession& operator=(const EncoderStageProfileSession&) =
      delete;

 private:
  EncoderStageProfileSink* sink_;
  EncoderStageProfileAccumulator* previous_session_;
  EncoderStageProfileAccumulator accumulator_;
};

class EncoderStageProfilePhaseTimer {
 public:
  explicit EncoderStageProfilePhaseTimer(EncoderProfilePhase phase)
      : accumulator_(stage_profile_internal::session),
        previous_current_(stage_profile_internal::current),
        phase_(phase) {
    if (accumulator_ != nullptr) {
      stage_profile_internal::current = accumulator_;
      begin_ = std::chrono::steady_clock::now();
    }
  }
  ~EncoderStageProfilePhaseTimer() { Stop(); }

  void Stop() {
    if (accumulator_ == nullptr) return;
    const uint64_t elapsed = stage_profile_internal::ElapsedNanoseconds(begin_);
    const size_t phase = static_cast<size_t>(phase_);
    const size_t complete =
        static_cast<size_t>(EncoderProfilePhase::kCompleteSerializer);
    stage_profile_internal::AddChecked(elapsed,
                                       &accumulator_->phase_nanoseconds[phase],
                                       &accumulator_->overflowed);
    stage_profile_internal::AddChecked(
        1, &accumulator_->phase_invocations[phase], &accumulator_->overflowed);
    if (phase_ != EncoderProfilePhase::kCompleteSerializer) {
      stage_profile_internal::AddChecked(
          elapsed, &accumulator_->phase_nanoseconds[complete],
          &accumulator_->overflowed);
    }
    stage_profile_internal::current = previous_current_;
    accumulator_ = nullptr;
  }

  EncoderStageProfilePhaseTimer(const EncoderStageProfilePhaseTimer&) = delete;
  EncoderStageProfilePhaseTimer& operator=(
      const EncoderStageProfilePhaseTimer&) = delete;

 private:
  EncoderStageProfileAccumulator* accumulator_;
  EncoderStageProfileAccumulator* previous_current_;
  EncoderProfilePhase phase_;
  std::chrono::steady_clock::time_point begin_{};
};

class EncoderStageProfileWorkTimer {
 public:
  explicit EncoderStageProfileWorkTimer(EncoderProfileWork work)
      : accumulator_(stage_profile_internal::current), work_(work) {
    if (accumulator_ != nullptr) begin_ = std::chrono::steady_clock::now();
  }
  ~EncoderStageProfileWorkTimer() { Stop(); }

  void Stop() {
    if (accumulator_ == nullptr) return;
    const uint64_t elapsed = stage_profile_internal::ElapsedNanoseconds(begin_);
    const size_t work = static_cast<size_t>(work_);
    stage_profile_internal::AddChecked(elapsed,
                                       &accumulator_->work_nanoseconds[work],
                                       &accumulator_->overflowed);
    stage_profile_internal::AddChecked(1, &accumulator_->work_invocations[work],
                                       &accumulator_->overflowed);
    accumulator_ = nullptr;
  }

  EncoderStageProfileWorkTimer(const EncoderStageProfileWorkTimer&) = delete;
  EncoderStageProfileWorkTimer& operator=(const EncoderStageProfileWorkTimer&) =
      delete;

 private:
  EncoderStageProfileAccumulator* accumulator_;
  EncoderProfileWork work_;
  std::chrono::steady_clock::time_point begin_{};
};

inline EncoderStageProfileAccumulator* CurrentEncoderStageProfile() {
  return stage_profile_internal::current;
}

inline void MergeEncoderStageProfile(
    const EncoderStageProfileAccumulator& source) {
  if (stage_profile_internal::current != nullptr) {
    stage_profile_internal::Merge(source, stage_profile_internal::current);
  }
}

inline void EncoderStageProfileAddCount(EncoderProfileCount count,
                                        uint64_t value) {
  EncoderStageProfileAccumulator* accumulator = stage_profile_internal::current;
  if (accumulator == nullptr) return;
  stage_profile_internal::AddChecked(
      value, &accumulator->counts[static_cast<size_t>(count)],
      &accumulator->overflowed);
}

#else

class EncoderStageProfileScope {
 public:
  explicit EncoderStageProfileScope(EncoderStageProfileAccumulator*) {}
};

class EncoderStageProfileSession {
 public:
  explicit EncoderStageProfileSession(EncoderStageProfileSink*) {}
};

class EncoderStageProfilePhaseTimer {
 public:
  explicit EncoderStageProfilePhaseTimer(EncoderProfilePhase) {}
  void Stop() {}
};

class EncoderStageProfileWorkTimer {
 public:
  explicit EncoderStageProfileWorkTimer(EncoderProfileWork) {}
  void Stop() {}
};

inline EncoderStageProfileAccumulator* CurrentEncoderStageProfile() {
  return nullptr;
}
inline void MergeEncoderStageProfile(const EncoderStageProfileAccumulator&) {}
inline void EncoderStageProfileAddCount(EncoderProfileCount, uint64_t) {}

#endif  // JPEGXL_ENABLE_STAGE_PROFILER

}  // namespace jxl

#if JPEGXL_ENABLE_STAGE_PROFILER
// Private benchmark hook. This declaration is deliberately outside the public
// include tree and is absent from ordinary libjxl builds.
extern "C" JXL_EXPORT JxlEncoderStatus
JxlEncoderFrameSettingsSetStageProfileForBenchmark(
    JxlEncoderFrameSettings* frame_settings,
    jxl::EncoderStageProfileSink* profile);
#endif

#endif  // LIB_JXL_ENC_STAGE_PROFILE_H_
