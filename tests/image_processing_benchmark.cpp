// Optional microbenchmark, never a pass/fail test. Compare the pinned original
// algorithms with the linked SDK on identical deterministic float32 images.
#include "ouster/image_processing.h"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "image_processing_legacy.h"

namespace {
using ouster::sdk::core::img_t;
namespace current = ouster::sdk::core::image;
using Clock = std::chrono::steady_clock;

struct Timing {
    double cpu_ms, wall_ms;
    float checksum;
};

std::vector<img_t<float>> make_frames(bool masked) {
    std::vector<img_t<float>> frames;
    for (int frame = 0; frame < 8; ++frame) {
        img_t<float> image(128, 2048);
        uint32_t random = 0x8aa86b9u + static_cast<uint32_t>(frame);
        for (int row = 0; row < image.rows(); ++row) {
            for (int column = 0; column < image.cols(); ++column) {
                random = random * 1664525u + 1013904223u;
                image(row, column) = masked && (column < 512 || column >= 1536)
                    ? 0.0f : static_cast<float>(1u + random % 65535u);
            }
        }
        frames.push_back(std::move(image));
    }
    return frames;
}

template <typename Apply>
Timing measure(const std::vector<img_t<float>>& inputs, int count, Apply apply) {
    img_t<float> image(inputs.front().rows(), inputs.front().cols());
    // Include steady-state modulo-three/eight scheduling, after initialization.
    for (int frame = 0; frame < 24; ++frame) {
        image = inputs[static_cast<size_t>(frame) % inputs.size()];
        apply(image);
    }
    volatile float checksum = 0.0f;
    const auto wall_start = Clock::now();
    const auto cpu_start = std::clock();
    for (int frame = 0; frame < count; ++frame) {
        // Copying fresh source pixels is included identically for both versions;
        // these timings are not a pure percentile-selection benchmark.
        image = inputs[static_cast<size_t>(frame) % inputs.size()];
        apply(image);
        checksum += image(0, 1024);
    }
    const auto cpu_end = std::clock();
    const auto wall_end = Clock::now();
    return {1000.0 * static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC,
            std::chrono::duration<double, std::milli>(wall_end - wall_start).count(),
            checksum};
}

Timing benchmark(const std::string& algorithm, bool legacy,
                 const std::vector<img_t<float>>& inputs, int count) {
    if (algorithm == "AE") {
        if (legacy) {
            image_processing_legacy::AutoExposure processor;
            return measure(inputs, count, [&](img_t<float>& image) {
                processor.update<float>(image, true);
            });
        }
        current::AutoExposure processor;
        return measure(inputs, count, [&](img_t<float>& image) { processor.update(image); });
    }
    if (legacy) {
        image_processing_legacy::BeamUniformityCorrector processor;
        return measure(inputs, count, [&](img_t<float>& image) {
            processor.update<float>(image, true);
        });
    }
    current::BeamUniformityCorrector processor;
    return measure(inputs, count, [&](img_t<float>& image) { processor.update(image); });
}
}  // namespace

int main(int argc, char** argv) {
    const int frames = argc > 1 ? std::stoi(argv[1]) : 96;
    const int rounds = argc > 2 ? std::stoi(argv[2]) : 5;
    if (frames < 1 || rounds < 1) throw std::invalid_argument("frames and rounds must be positive");
    std::cout << "algorithm,mask,round,order,implementation,frames,cpu_ms,wall_ms,checksum\n";
    std::cout << std::setprecision(9);
    for (const bool masked : {false, true}) {
        const auto inputs = make_frames(masked);
        for (const std::string algorithm : {"AE", "BUC"}) {
            for (int round = 0; round < rounds; ++round) {
                for (int order = 0; order < 2; ++order) {
                    const bool legacy = (round + order) % 2 == 0;
                    const auto time = benchmark(algorithm, legacy, inputs, frames);
                    std::cout << algorithm << ',' << (masked ? "half" : "full") << ','
                              << round << ',' << order << ',' << (legacy ? "old" : "new")
                              << ',' << frames << ',' << time.cpu_ms << ',' << time.wall_ms
                              << ',' << time.checksum << '\n';
                }
            }
        }
    }
}
