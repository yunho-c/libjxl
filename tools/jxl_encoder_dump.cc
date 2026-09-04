// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/encode.h>
#include <jxl/thread_parallel_runner.h>
#include <jxl/thread_parallel_runner_cxx.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "lib/extras/dec/color_hints.h"
#include "lib/extras/dec/decode.h"
#include "lib/extras/enc/jxl.h"
#include "lib/extras/encoder_debug_data.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/enc_debug_data.h"
#include "lib/jxl/encode_internal.h"
#include "tools/args.h"
#include "tools/cmdline.h"
#include "tools/file_io.h"
#include "tools/tool_version.h"

namespace jpegxl {
namespace tools {
namespace {

struct EncoderDumpArgs {
  void AddCommandLineOptions(CommandLineParser* cmdline) {
    cmdline->AddPositionalOption(
        "INPUT", true, "input image (JPEG input is rejected)", &input);
    cmdline->AddPositionalOption("OUTPUT", true, "encoded JPEG XL output",
                                 &output);
    cmdline->AddOptionValue('\0', "debug_dump_dir", "DIRECTORY",
                            "directory for .npy files and manifest.json",
                            &dump_dir, &ParseString);
    cmdline->AddOptionValue('\0', "debug_dump_profile", "PROFILE",
                            "artifact profile: overview (default) or all",
                            &profile, &ParseString);
    cmdline->AddOptionValue('d', "distance", "DISTANCE",
                            "VarDCT target distance, default = 1.0", &distance,
                            &ParseFloat);
    cmdline->AddOptionValue('e', "effort", "EFFORT",
                            "encoder effort from 1 through 10, default = 7",
                            &effort, &ParseUnsigned);
    cmdline->AddOptionValue('\0', "num_threads", "THREADS",
                            "worker threads; -1 uses the machine default",
                            &num_threads, &ParseSigned);
    cmdline->AddOptionValue('\0', "color_space", "DESCRIPTION",
                            "color-space hint for raw formats such as PPM",
                            &color_space, &ParseString);
  }

  const char* input = nullptr;
  const char* output = nullptr;
  std::string dump_dir;
  std::string profile = "overview";
  std::string color_space;
  float distance = 1.0f;
  size_t effort = 7;
  int num_threads = -1;
};

void AttachDebugDataSink(JxlEncoderFrameSettings* settings, void* opaque) {
  JxlEncoderSetDebugDataSink(settings,
                             static_cast<jxl::EncoderDebugDataSink*>(opaque));
}

bool ConfigureProfile(const std::string& profile,
                      jxl::extras::FileEncoderDebugDataSinkOptions* options) {
  if (profile == "all") return true;
  if (profile != "overview") {
    fprintf(stderr, "Unknown debug dump profile: %s\n", profile.c_str());
    return false;
  }
  options->include_prefixes = {
      "color/input_encoded",
      "color/linear_srgb",
      "color/xyb_after_transform",
      "gaborish/",
      "aq/initial/",
      "aq/final/",
      "cfl/",
      "ac/final/",
      "epf/",
  };
  return true;
}

bool ValidateArgs(const EncoderDumpArgs& args) {
  if (args.input == nullptr || args.output == nullptr) return false;
  if (args.dump_dir.empty()) {
    fprintf(stderr, "--debug_dump_dir is required.\n");
    return false;
  }
  if (!(args.distance > 0.0f && args.distance <= 25.0f)) {
    fprintf(stderr, "--distance must be greater than 0 and at most 25.\n");
    return false;
  }
  if (args.effort < 1 || args.effort > 10) {
    fprintf(stderr, "--effort must be from 1 through 10.\n");
    return false;
  }
  if (args.num_threads < -1) {
    fprintf(stderr, "--num_threads must be -1 or greater.\n");
    return false;
  }
  return true;
}

int Run(const EncoderDumpArgs& args) {
  std::vector<uint8_t> input_bytes;
  if (!ReadFile(args.input, &input_bytes)) {
    fprintf(stderr, "Failed to read input image: %s\n", args.input);
    return EXIT_FAILURE;
  }
  if (jxl::extras::DetectCodec(jxl::Bytes(input_bytes)) ==
      jxl::extras::Codec::kJPG) {
    fprintf(stderr,
            "JPEG input is not supported by this one-shot pixel frontend.\n");
    return EXIT_FAILURE;
  }

  jxl::extras::ColorHints color_hints;
  if (!args.color_space.empty()) {
    color_hints.Add("color_space", args.color_space);
  }
  jxl::extras::PackedPixelFile ppf;
  jxl::extras::Codec source_codec = jxl::extras::Codec::kUnknown;
  if (!jxl::extras::DecodeBytes(jxl::Bytes(input_bytes), color_hints, &ppf,
                                nullptr, &source_codec)) {
    fprintf(stderr, "Failed to decode input image: %s\n", args.input);
    return EXIT_FAILURE;
  }
  if (source_codec == jxl::extras::Codec::kJPG) {
    fprintf(stderr,
            "JPEG input is not supported by this one-shot pixel frontend.\n");
    return EXIT_FAILURE;
  }
  if (ppf.info.have_animation || ppf.frames.size() != 1 ||
      !ppf.chunked_frames.empty()) {
    fprintf(stderr,
            "Only a single, non-animated, non-streaming frame is supported.\n");
    return EXIT_FAILURE;
  }

  jxl::extras::FileEncoderDebugDataSinkOptions dump_options;
  dump_options.output_dir = args.dump_dir;
  dump_options.profile = args.profile;
  dump_options.libjxl_revision = kJpegxlVersion;
  if (!ConfigureProfile(args.profile, &dump_options)) return EXIT_FAILURE;
  std::unique_ptr<jxl::EncoderDebugDataSink> sink =
      jxl::extras::CreateFileEncoderDebugDataSink(dump_options);

  size_t num_threads = JxlThreadParallelRunnerDefaultNumWorkerThreads();
  if (args.num_threads >= 0) {
    num_threads = static_cast<size_t>(args.num_threads);
  }
  JxlThreadParallelRunnerPtr runner =
      JxlThreadParallelRunnerMake(nullptr, num_threads);
  if (!runner) {
    fprintf(stderr, "Failed to create encoder thread runner.\n");
    return EXIT_FAILURE;
  }

  jxl::extras::JXLCompressParams params;
  params.distance = args.distance;
  params.runner = JxlThreadParallelRunner;
  params.runner_opaque = runner.get();
  params.AddOption(JXL_ENC_FRAME_SETTING_EFFORT,
                   static_cast<int64_t>(args.effort));
  params.AddOption(JXL_ENC_FRAME_SETTING_MODULAR, 0);
  params.frame_settings_callback = AttachDebugDataSink;
  params.frame_settings_callback_opaque = sink.get();

  std::vector<uint8_t> compressed;
  if (!jxl::extras::EncodeImageJXL(params, ppf, nullptr, &compressed)) {
    fprintf(stderr, "JPEG XL encoding failed.\n");
    return EXIT_FAILURE;
  }
  if (!sink->Finish()) {
    fprintf(stderr, "Failed to finalize debug dump manifest.\n");
    return EXIT_FAILURE;
  }
  if (!WriteFile(args.output, compressed)) {
    fprintf(stderr, "Failed to write JPEG XL output: %s\n", args.output);
    return EXIT_FAILURE;
  }
  fprintf(stderr, "Wrote %zu codestream bytes and debug data to %s\n",
          compressed.size(), args.dump_dir.c_str());
  return EXIT_SUCCESS;
}

}  // namespace

int EncoderDumpMain(int argc, const char* argv[]) {
  EncoderDumpArgs args;
  CommandLineParser cmdline;
  args.AddCommandLineOptions(&cmdline);
  if (!cmdline.Parse(argc, argv)) {
    fprintf(stderr, "Use '%s -h' for more information.\n", argv[0]);
    return EXIT_FAILURE;
  }
  if (cmdline.HelpFlagPassed() || args.input == nullptr) {
    cmdline.PrintHelp();
    return EXIT_SUCCESS;
  }
  if (!ValidateArgs(args)) return EXIT_FAILURE;
  return Run(args);
}

}  // namespace tools
}  // namespace jpegxl

int main(int argc, char** argv) {
  return jpegxl::tools::EncoderDumpMain(argc, const_cast<const char**>(argv));
}
