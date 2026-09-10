// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Warm linear-RGB-to-codestream batch wall time. See
// doc/image-batch-benchmark.md.
#include <jxl/encode.h>
#include <jxl/thread_parallel_runner.h>
#include <jxl/thread_parallel_runner_cxx.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "lib/extras/dec/decode.h"
#include "lib/extras/dec/jxl.h"
#include "lib/extras/enc/jxl.h"
#include "tools/file_io.h"

namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using jxl::extras::PackedPixelFile;

struct Options {
  std::vector<fs::path> inputs;
  fs::path raw_samples;
  std::vector<size_t> batch_sizes = {1, 2, 4, 8};
  size_t samples = 3;
  size_t warmups = 1;
  size_t threads =
      std::max<size_t>(1, JxlThreadParallelRunnerDefaultNumWorkerThreads());
  int effort = 7;
  float distance = 1.2f;
};

void Usage(const char* executable) {
  std::cout
      << "Usage: " << executable
      << " --input FILE.pfm|DIRECTORY [--input ...]"
         " [--batch-sizes 1,2,4,8] [--samples N] [--warmups N]"
         " [--threads-per-image N] [--distance VALUE] [--effort 1..10]"
         " [--raw-samples NEW.csv]\n"
         "RGB PFMs must contain finite linear sRGB pixels and have scale "
         "+/-1.\n"
         "Directories are non-recursive; original dimensions are preserved.\n"
         "Defaults: effort 7, distance 1.2, 3 pairs, 1 warmup, hardware "
         "threads\n"
         "per image (resolved once). N=1 runs each encode on its outer "
         "worker.\n";
}

size_t PositiveInteger(const std::string& text) {
  if (text.empty() ||
      text.find_first_not_of("0123456789") != std::string::npos) {
    throw std::runtime_error("Expected a positive integer: " + text);
  }
  errno = 0;
  const auto value = std::strtoull(text.c_str(), nullptr, 10);
  // The runner implementation accepts an int worker count.
  if (errno == ERANGE || value == 0 ||
      value > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Integer out of range: " + text);
  }
  return static_cast<size_t>(value);
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      Usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (++i == argc) throw std::runtime_error("Missing value for " + arg);
    const std::string value = argv[i];
    if (arg == "--input") {
      options.inputs.emplace_back(value);
    } else if (arg == "--raw-samples") {
      if (value.empty()) throw std::runtime_error("Empty output path");
      options.raw_samples = value;
    } else if (arg == "--samples") {
      options.samples = PositiveInteger(value);
    } else if (arg == "--warmups") {
      options.warmups = PositiveInteger(value);
    } else if (arg == "--threads-per-image") {
      options.threads = PositiveInteger(value);
    } else if (arg == "--effort") {
      options.effort = static_cast<int>(PositiveInteger(value));
      if (options.effort > 10) throw std::runtime_error("Effort must be 1..10");
    } else if (arg == "--distance") {
      char* end = nullptr;
      errno = 0;
      options.distance = std::strtof(value.c_str(), &end);
      if (errno == ERANGE || end != value.c_str() + value.size() ||
          !std::isfinite(options.distance) || options.distance <= 0.0f) {
        throw std::runtime_error("Distance must be finite and positive");
      }
    } else if (arg == "--batch-sizes") {
      options.batch_sizes.clear();
      size_t begin = 0;
      while (true) {
        const size_t end = value.find(',', begin);
        options.batch_sizes.push_back(
            PositiveInteger(value.substr(begin, end - begin)));
        if (end == std::string::npos) break;
        begin = end + 1;
      }
      if (!std::is_sorted(options.batch_sizes.begin(),
                          options.batch_sizes.end()) ||
          std::adjacent_find(options.batch_sizes.begin(),
                             options.batch_sizes.end()) !=
              options.batch_sizes.end()) {
        throw std::runtime_error("Batch sizes must be unique and increasing");
      }
    } else {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }
  if (options.inputs.empty())
    throw std::runtime_error("At least one --input is required");
  return options;
}

bool IsPfm(const fs::path& path) {
  std::string ext = path.extension().string();
  for (char& c : ext) {
    if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
  }
  return ext == ".pfm";
}

std::vector<fs::path> ResolveInputs(const std::vector<fs::path>& inputs) {
  std::vector<fs::path> files;
  for (const auto& input : inputs) {
    if (fs::is_directory(input)) {
      const size_t before = files.size();
      for (const auto& entry : fs::directory_iterator(input)) {
        if (entry.is_regular_file() && IsPfm(entry.path())) {
          files.push_back(fs::canonical(entry.path()));
        }
      }
      if (files.size() == before)
        throw std::runtime_error("No PFMs in " + input.string());
    } else if (fs::is_regular_file(input) && IsPfm(input)) {
      files.push_back(fs::canonical(input));
    } else {
      throw std::runtime_error("Expected a PFM or directory: " +
                               input.string());
    }
  }
  std::sort(files.begin(), files.end());
  files.erase(std::unique(files.begin(), files.end()), files.end());
  return files;
}

void ValidatePixels(const PackedPixelFile& image) {
  if (image.frames.size() != 1 || image.info.num_color_channels != 3 ||
      image.info.num_extra_channels != 0 || image.info.xsize == 0 ||
      image.info.ysize == 0) {
    throw std::runtime_error("Expected one RGB image without extra channels");
  }
  const auto& pixels = image.frames[0].color;
  for (size_t y = 0; y < pixels.ysize; ++y) {
    for (size_t x = 0; x < pixels.xsize; ++x) {
      for (size_t c = 0; c < 3; ++c) {
        if (!std::isfinite(pixels.GetPixelValue(y, x, c))) {
          throw std::runtime_error("Image contains a non-finite pixel");
        }
      }
    }
  }
}

PackedPixelFile LoadImage(const fs::path& path) {
  // GJXL applies the PFM scale magnitude; libjxl ignores it. Require unit scale
  // so that both benchmarks measure exactly the same pixels.
  std::ifstream header(path, std::ios::binary);
  auto line = [&]() {
    std::string value;
    while (std::getline(header, value)) {
      if (!value.empty() && value.back() == '\r') value.pop_back();
      if (!value.empty() && value[0] != '#') return value;
    }
    throw std::runtime_error("Truncated PFM header: " + path.string());
  };
  if (line() != "PF")
    throw std::runtime_error("Expected an RGB PFM: " + path.string());
  (void)line();  // Dimensions are validated by the existing decoder.
  std::istringstream scale_line(line());
  float scale = 0;
  std::string trailing;
  if (!(scale_line >> scale) || (scale_line >> trailing) ||
      std::abs(scale) != 1.0f) {
    throw std::runtime_error("PFM scale must be +1 or -1: " + path.string());
  }
  std::vector<uint8_t> bytes;
  PackedPixelFile image;
  jxl::extras::ColorHints hints;
  hints.Add("color_space", "RGB_D65_SRG_Rel_Lin");
  if (!jpegxl::tools::ReadFile(path.string(), &bytes) ||
      !jxl::extras::DecodeBytes(jxl::Bytes(bytes), hints, &image)) {
    throw std::runtime_error("Unable to load PFM: " + path.string());
  }
  ValidatePixels(image);
  return image;
}

JxlThreadParallelRunnerPtr MakeRunner(size_t workers) {
  auto runner = JxlThreadParallelRunnerMake(nullptr, workers);
  if (!runner) throw std::runtime_error("Unable to create thread pool");
  return runner;
}

jxl::extras::JXLCompressParams MakeParams(const Options& options,
                                          void* runner) {
  jxl::extras::JXLCompressParams params;
  params.distance = options.distance;
  params.runner_opaque = runner;
  params.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, options.effort);
  return params;
}

size_t InnerWorkers(const Options& options) {
  return options.threads == 1 ? 0 : options.threads;
}

std::vector<uint8_t> Reference(const PackedPixelFile& image,
                               const Options& options) {
  auto runner = MakeRunner(InnerWorkers(options));
  const auto params = MakeParams(options, runner.get());
  std::vector<uint8_t> bytes;
  if (!jxl::extras::EncodeImageJXL(params, image, nullptr, &bytes) ||
      bytes.empty()) {
    throw std::runtime_error("Reference encode failed");
  }
  jxl::extras::JXLDecompressParams decode;
  decode.runner = JxlThreadParallelRunner;
  decode.runner_opaque = runner.get();
  decode.color_space = "RGB_D65_SRG_Rel_Lin";
  decode.accepted_formats.push_back({3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0});
  size_t decoded_bytes = 0;
  PackedPixelFile decoded;
  if (!jxl::extras::DecodeImageJXL(bytes.data(), bytes.size(), decode,
                                   &decoded_bytes, &decoded)) {
    throw std::runtime_error("Reference decode failed");
  }
  if (decoded_bytes != bytes.size() || decoded.info.xsize != image.info.xsize ||
      decoded.info.ysize != image.info.ysize) {
    throw std::runtime_error(
        "Reference decode mismatch: consumed=" + std::to_string(decoded_bytes) +
        "/" + std::to_string(bytes.size()) +
        " dimensions=" + std::to_string(decoded.info.xsize) + "x" +
        std::to_string(decoded.info.ysize) +
        " expected=" + std::to_string(image.info.xsize) + "x" +
        std::to_string(image.info.ysize));
  }
  ValidatePixels(decoded);
  return bytes;
}

struct Result {
  std::vector<uint8_t> bytes;
  bool ok = false;
  std::exception_ptr error;
};

// Reuse libjxl's runner for image scheduling as well as intra-image work.
// Each outer worker owns a separate inner runner. No pool is entered twice.
class BatchEncoder {
 public:
  BatchEncoder(size_t workers, const Options& options)
      : outer_(MakeRunner(workers)) {
    for (size_t i = 0; i < workers; ++i) {
      inner_.push_back(MakeRunner(InnerWorkers(options)));
      params_.push_back(MakeParams(options, inner_.back().get()));
    }
  }

  std::vector<Result> Encode(const PackedPixelFile& image, size_t count) {
    // Include result allocation and output-buffer growth in the timed call.
    std::vector<Result> results(count);
    Work work{image, params_, results};
    if (JxlThreadParallelRunner(outer_.get(), &work, Init, EncodeOne, 0,
                                static_cast<uint32_t>(count)) != 0) {
      throw std::runtime_error("Batch runner failed");
    }
    return results;
  }

 private:
  struct Work {
    const PackedPixelFile& image;
    const std::vector<jxl::extras::JXLCompressParams>& params;
    std::vector<Result>& results;
  };
  static JxlParallelRetCode Init(void* opaque, size_t threads) {
    return threads == static_cast<Work*>(opaque)->params.size() ? 0 : -1;
  }
  static void EncodeOne(void* opaque, uint32_t index, size_t thread) {
    auto& work = *static_cast<Work*>(opaque);
    auto& result = work.results[index];
    try {
      result.ok = jxl::extras::EncodeImageJXL(work.params[thread], work.image,
                                              nullptr, &result.bytes);
    } catch (...) {
      result.error = std::current_exception();
    }
  }
  JxlThreadParallelRunnerPtr outer_;
  std::vector<JxlThreadParallelRunnerPtr> inner_;
  std::vector<jxl::extras::JXLCompressParams> params_;
};

int64_t TimeBatch(BatchEncoder& encoder, const PackedPixelFile& image,
                  size_t count, const std::vector<uint8_t>& reference) {
  const auto start = Clock::now();
  const auto results = encoder.Encode(image, count);
  const int64_t ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
          .count();
  for (const auto& result : results) {
    if (result.error) std::rethrow_exception(result.error);
    if (!result.ok || result.bytes != reference) {
      throw std::runtime_error(
          "Encode failed or differs from single-image reference");
    }
  }
  if (ns <= 0) throw std::runtime_error("Non-positive elapsed time");
  return ns;
}

std::string Csv(const std::string& text) {
  std::string escaped = "\"";
  for (const char c : text) {
    if (c == '"') escaped += '"';
    escaped += c;
  }
  return escaped + '"';
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  return values.size() % 2 ? values[mid]
                           : 0.5 * (values[mid - 1] + values[mid]);
}

void Benchmark(const fs::path& path, const PackedPixelFile& image,
               const std::vector<uint8_t>& reference, size_t count,
               const Options& options, std::ostream* raw) {
  BatchEncoder serial(1, options);
  BatchEncoder batch(count, options);
  auto pair = [&](bool batch_first) {
    int64_t serial_ns, batch_ns;
    if (batch_first) {
      batch_ns = TimeBatch(batch, image, count, reference);
      serial_ns = TimeBatch(serial, image, count, reference);
    } else {
      serial_ns = TimeBatch(serial, image, count, reference);
      batch_ns = TimeBatch(batch, image, count, reference);
    }
    return std::make_pair(serial_ns, batch_ns);
  };
  for (size_t i = 0; i < options.warmups; ++i) (void)pair(i % 2 != 0);
  std::vector<double> serial_ms, batch_ms, ratios;
  for (size_t i = 0; i < options.samples; ++i) {
    const bool batch_first = i % 2 != 0;
    const auto ns = pair(batch_first);
    serial_ms.push_back(static_cast<double>(ns.first) / 1e6);
    batch_ms.push_back(static_cast<double>(ns.second) / 1e6);
    ratios.push_back(static_cast<double>(ns.first) / ns.second);
    if (raw) {
      *raw << "libjxl," << Csv(path.string()) << ',' << Csv(path.string())
           << ',' << image.info.xsize << ',' << image.info.ysize << ',' << count
           << ',' << i << ',' << (batch_first ? "batch-first" : "serial-first")
           << ",cpu,cpu,n/a,"
           << std::setprecision(std::numeric_limits<float>::max_digits10)
           << options.distance << ',' << options.effort
           << ",fixed_per_image:" << options.threads
           << ",linear_rgb_to_in_memory_codestream," << ns.first << ','
           << ns.second << ',' << reference.size() << '\n'
           << std::flush;
      if (!*raw) throw std::runtime_error("Unable to write raw samples");
    }
  }
  const double batch_median = Median(batch_ms);
  std::cout << Csv(path.string()) << ',' << image.info.xsize << ','
            << image.info.ysize << ',' << count << ',' << std::fixed
            << std::setprecision(3) << Median(serial_ms) << ',' << batch_median
            << ',' << batch_median / count << ','
            << 1000.0 * count / batch_median << ',' << Median(ratios) << ','
            << *std::min_element(ratios.begin(), ratios.end()) << ','
            << *std::max_element(ratios.begin(), ratios.end()) << '\n'
            << std::flush;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    const auto inputs = ResolveInputs(options.inputs);
    std::ofstream raw;
    if (!options.raw_samples.empty()) {
      if (fs::exists(fs::symlink_status(options.raw_samples))) {
        throw std::runtime_error("Raw sample output already exists: " +
                                 options.raw_samples.string());
      }
      raw.open(options.raw_samples);
      if (!raw)
        throw std::runtime_error("Unable to open raw sample output: " +
                                 options.raw_samples.string());
      raw << "codec,workload,source,width,height,batch_size,sample,order,"
             "requested_backend,backend,aq_mode,distance,effort,thread_policy,"
             "timing_boundary,serial_ns,batch_ns,encoded_bytes_per_image\n";
    }
    std::cerr << "libjxl image batch benchmark: effort=" << options.effort
              << " distance=" << options.distance
              << " threads_per_image=" << options.threads
              << " inner_workers_per_image=" << InnerWorkers(options)
              << " samples=" << options.samples
              << " warmups=" << options.warmups << '\n';
    std::cout
        << "workload,width,height,batch_size,serial_median_ms,"
           "batch_median_ms,batch_ms_per_image,batch_images_per_second,"
           "paired_speedup_median,paired_speedup_min,paired_speedup_max\n";
    for (const auto& path : inputs) {
      const auto image = LoadImage(path);
      const auto reference = Reference(image, options);
      for (const size_t count : options.batch_sizes) {
        Benchmark(path, image, reference, count, options,
                  raw.is_open() ? &raw : nullptr);
      }
    }
    if (raw.is_open()) {
      raw.close();
      if (!raw) throw std::runtime_error("Unable to close raw sample output");
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "Benchmark error: " << error.what() << '\n';
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "Benchmark error: unexpected failure\n";
    return EXIT_FAILURE;
  }
}
