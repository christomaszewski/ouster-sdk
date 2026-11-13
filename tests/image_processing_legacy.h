// Frozen grayscale auto-exposure reference from SDK 8aa86b94b1c75dfcb99159a4147b44d973e52323.
// Keep the old algorithm independent of the linked SDK implementation.
#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include "ouster/typedefs.h"

namespace image_processing_legacy {
using ouster::sdk::core::img_t;
constexpr size_t AE_STRIDE = 4;
constexpr size_t AE_MIN_NONZERO_POINTS = 100;
class AutoExposure {
    const double lo_percentile_, hi_percentile_;
    const int ae_update_every_;
    const double damping_;
    double lo_state_ = -1.0, hi_state_ = -1.0, lo_ = -1.0, hi_ = -1.0;
    bool initialized_ = false;
    int counter_ = 0;
   public:
    AutoExposure(double lo = 0.1, double hi = 0.1, int update_every = 3,
                 double damping = 0.9)
        : lo_percentile_(lo), hi_percentile_(hi),
          ae_update_every_(update_every), damping_(damping) {}
template <typename T>
void update(Eigen::Ref<img_t<T>> image, bool update_state) {
    Eigen::Map<Eigen::Array<T, -1, 1>> key_eigen(image.data(), image.size());

    if (counter_ == 0 && update_state) {
        const size_t num_elements = key_eigen.rows();
        std::vector<size_t> indices;
        indices.reserve(num_elements);
        for (size_t i = 0; i < num_elements; i += AE_STRIDE) {
            // ignore 0 values, which are often due to dropped packets etc
            if (key_eigen[i] > 0) {
                indices.push_back(i);
            }
        }
        if (indices.size() < AE_MIN_NONZERO_POINTS) {
            // too few nonzero values, nothing to do
            return;
        }
        auto cmp = [&](const size_t a, const size_t b) { return key_eigen(a) < key_eigen(b); };

        const size_t lo_kth_extreme =
            static_cast<size_t>(static_cast<double>(indices.size()) * lo_percentile_);
        std::nth_element(indices.begin(),
                         indices.begin() + static_cast<std::ptrdiff_t>(lo_kth_extreme),
                         indices.end(), cmp);
        lo_ = key_eigen[*(indices.begin() + static_cast<std::ptrdiff_t>(lo_kth_extreme))];

        const size_t hi_kth_extreme =
            static_cast<size_t>(static_cast<double>(indices.size()) * hi_percentile_);
        std::nth_element(indices.begin() + static_cast<std::ptrdiff_t>(lo_kth_extreme),
                         indices.end() - static_cast<std::ptrdiff_t>(hi_kth_extreme) - 1,
                         indices.end(), cmp);
        hi_ = key_eigen[*(indices.end() - static_cast<std::ptrdiff_t>(hi_kth_extreme) - 1)];

        if (!initialized_) {
            initialized_ = true;
            lo_state_ = lo_;
            hi_state_ = hi_;
        }
    }
    if (!initialized_) {
        return;
    }

    // we use the simplest form of exponential smoothing
    if (update_state) {
        lo_state_ = damping_ * lo_state_ + (1.0 - damping_) * lo_;
        hi_state_ = damping_ * hi_state_ + (1.0 - damping_) * hi_;
    }

    // Apply affine transformation mapping lo_state to lo_percentile and
    // hi_state to 1 - hi_percentile. If it would map 0 to positive number,
    // instead map using only hi_state
    double lo_hi_scale = (1.0 - (lo_percentile_ + hi_percentile_)) / (hi_state_ - lo_state_);

    if (std::isinf(lo_hi_scale) || std::isnan(lo_hi_scale)) {
        // map everything relative to hi_state being 0.5 due to small spread or
        // nan
        key_eigen *= 0.5 / hi_state_;
    } else if (lo_hi_scale * (0.0 - lo_state_) + lo_percentile_ <= 0.00) {
        // apply affine transformation
        key_eigen -= lo_state_;
        key_eigen *= lo_hi_scale;
        key_eigen += lo_percentile_;
    } else {
        // lo_hi_state transformation would map 0 to positive number
        // instead, map using only hi_state
        key_eigen *= (1.0 - hi_percentile_) / (hi_state_);
    }

    // clamp
    key_eigen = key_eigen.max(static_cast<T>(0)).min(static_cast<T>(1));

    if (update_state) {
        counter_ = (counter_ + 1) % ae_update_every_;
    }
}

};
}  // namespace image_processing_legacy
