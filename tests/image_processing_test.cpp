/** Regression comparisons against the image algorithms at SDK 8aa86b9. */
#include "ouster/image_processing.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "image_processing_legacy.h"

namespace {
using ouster::sdk::core::img_t;
using ouster::sdk::core::image::AutoExposure;
using ouster::sdk::core::image::BeamUniformityCorrector;

enum class Pattern {
    Dense, Ties, Zero, Sparse99, Exactly100, Exceptional, Infinity,
    Tiny, Large, MaskedColumns,
};

template <typename T>
img_t<T> frame(Pattern pattern, int iteration, int height = 8, int width = 128) {
    img_t<T> image(height, width);
    uint32_t random = 0x8aa86b9u + static_cast<uint32_t>(iteration);
    for (Eigen::Index i = 0; i < image.size(); ++i) {
        random = random * 1664525u + 1013904223u;
        image.data()[i] = static_cast<T>(1u + random % 65535u);
    }
    switch (pattern) {
        case Pattern::Dense: break;
        case Pattern::Ties:
            for (Eigen::Index i = 0; i < image.size(); ++i)
                image.data()[i] = static_cast<T>(7 + (i + iteration) % 3);
            break;
        case Pattern::Zero: image.setZero(); break;
        case Pattern::Sparse99:
        case Pattern::Exactly100: {
            image.setZero();
            const int positive_count = pattern == Pattern::Sparse99 ? 99 : 100;
            for (int i = 0; i < positive_count && i * 4 < image.size(); ++i)
                image.data()[i * 4] = static_cast<T>(i + iteration + 1);
            // Unsampled positive values must not change the 100-sample threshold.
            for (Eigen::Index i = 1; i < image.size(); i += 4)
                image.data()[i] = static_cast<T>(1000);
            break;
        }
        case Pattern::Exceptional:
            for (Eigen::Index i = 0; i < image.size(); ++i) {
                if (i % 13 == 0) image.data()[i] = std::numeric_limits<T>::quiet_NaN();
                else if (i % 17 == 0) image.data()[i] = std::numeric_limits<T>::infinity();
                else if (i % 19 == 0) image.data()[i] = -std::numeric_limits<T>::infinity();
                else if (i % 23 == 0) image.data()[i] = static_cast<T>(-0.0);
                else if (i % 29 == 0) image.data()[i] = static_cast<T>(-11);
            }
            break;
        case Pattern::Infinity: image.setConstant(std::numeric_limits<T>::infinity()); break;
        case Pattern::Tiny:
            image *= std::numeric_limits<T>::denorm_min();
            break;
        case Pattern::Large:
            image *= std::numeric_limits<T>::max() / static_cast<T>(131072);
            break;
        case Pattern::MaskedColumns:
            for (int column = 0; column < width; ++column)
                if (column % 5 != 0) image.col(column).setZero();
            break;
    }
    return image;
}

// Finite values, infinities and signed zero must match bit-for-bit. NaN
// classification/position must match; payload bits are not an API contract.
template <typename T>
void expect_same_image(const img_t<T>& expected, const img_t<T>& actual) {
    ASSERT_EQ(expected.rows(), actual.rows());
    ASSERT_EQ(expected.cols(), actual.cols());
    for (Eigen::Index i = 0; i < expected.size(); ++i) {
        if (std::isnan(expected.data()[i])) {
            ASSERT_TRUE(std::isnan(actual.data()[i])) << "pixel " << i;
        } else {
            ASSERT_EQ(0, std::memcmp(expected.data() + i, actual.data() + i, sizeof(T)))
                << "pixel " << i << ": " << expected.data()[i] << " vs " << actual.data()[i];
        }
    }
}

template <typename T>
class ImageProcessingEquivalence : public ::testing::Test {};
using ImageTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ImageProcessingEquivalence, ImageTypes);

TYPED_TEST(ImageProcessingEquivalence, AutoExposurePatternsAndStateCadence) {
    for (const auto pattern : {Pattern::Dense, Pattern::Ties, Pattern::Zero,
                               Pattern::Sparse99, Pattern::Exactly100,
                               Pattern::Exceptional, Pattern::Infinity,
                               Pattern::Tiny, Pattern::Large, Pattern::MaskedColumns}) {
        for (const int update_every : {1, 3, 5}) {
            AutoExposure actual(0.1, 0.1, update_every, 0.9);
            image_processing_legacy::AutoExposure expected(0.1, 0.1, update_every, 0.9);
            for (int iteration = 0; iteration < 40; ++iteration) {
                SCOPED_TRACE(::testing::Message() << "pattern=" << static_cast<int>(pattern)
                             << " cadence=" << update_every << " frame=" << iteration);
                // Exercise update_state=false before initialization and in every
                // subsequent counter phase, while advancing through many cycles.
                const bool update_state = iteration % 5 != 0 && iteration % 7 != 0;
                auto reference = frame<TypeParam>(pattern, iteration);
                auto candidate = reference;
                expected.template update<TypeParam>(reference, update_state);
                actual.update(candidate, update_state);
                expect_same_image(reference, candidate);
            }
        }
    }
}

TYPED_TEST(ImageProcessingEquivalence, AutoExposureSparseFramesPreserveExistingState) {
    for (const double damping : {0.0, 0.9, 1.0}) {
        AutoExposure actual(0.2, 0.05, 3, damping);
        image_processing_legacy::AutoExposure expected(0.2, 0.05, 3, damping);
        for (int iteration = 0; iteration < 64; ++iteration) {
            SCOPED_TRACE(::testing::Message() << "damping=" << damping << " frame=" << iteration);
            const Pattern patterns[] = {Pattern::Dense, Pattern::Zero, Pattern::Sparse99,
                                        Pattern::Exactly100, Pattern::Ties, Pattern::Exceptional};
            auto reference = frame<TypeParam>(patterns[iteration % 6], iteration);
            auto candidate = reference;
            const bool update_state = iteration % 4 != 1;
            expected.template update<TypeParam>(reference, update_state);
            actual.update(candidate, update_state);
            expect_same_image(reference, candidate);
        }
    }
}

TYPED_TEST(ImageProcessingEquivalence, AutoExposureShapesPercentilesAndPrecision) {
    for (const double percentile : {0.0, 0.1, 0.49}) {
        AutoExposure actual(percentile, percentile, 3, 0.9);
        image_processing_legacy::AutoExposure expected(percentile, percentile, 3, 0.9);
        const int widths[] = {0, 1, 397, 400, 401, 511, 2048};
        for (int iteration = 0; iteration < 42; ++iteration) {
            SCOPED_TRACE(::testing::Message() << "percentile=" << percentile << " frame=" << iteration);
            const int width = widths[iteration % 7];
            auto reference = frame<TypeParam>(Pattern::Dense, iteration, width == 2048 ? 128 : 1, width);
            auto candidate = reference;
            expected.template update<TypeParam>(reference, iteration % 4 != 0);
            actual.update(candidate, iteration % 4 != 0);
            expect_same_image(reference, candidate);
        }
    }
}

TYPED_TEST(ImageProcessingEquivalence, AutoExposureFalseUpdatesDoNotAdvanceState) {
    AutoExposure actual;
    image_processing_legacy::AutoExposure expected;
    for (int iteration = 0; iteration < 30; ++iteration) {
        auto reference = frame<TypeParam>(Pattern::Dense, iteration);
        auto candidate = reference;
        const bool update_state = iteration < 4 || iteration >= 24;
        expected.template update<TypeParam>(reference, update_state);
        actual.update(candidate, update_state);
        expect_same_image(reference, candidate);
    }
}

TYPED_TEST(ImageProcessingEquivalence, BeamCorrectionPatternsAndStateCadence) {
    for (const auto pattern : {Pattern::Dense, Pattern::Ties, Pattern::Zero,
                               Pattern::Sparse99, Pattern::Exactly100,
                               Pattern::Exceptional, Pattern::Infinity,
                               Pattern::Tiny, Pattern::Large, Pattern::MaskedColumns}) {
        BeamUniformityCorrector actual;
        image_processing_legacy::BeamUniformityCorrector expected;
        for (int iteration = 0; iteration < 48; ++iteration) {
            SCOPED_TRACE(::testing::Message() << "pattern=" << static_cast<int>(pattern)
                         << " frame=" << iteration);
            auto reference = frame<TypeParam>(pattern, iteration);
            auto candidate = reference;
            // BUC advances its modulo-eight counter even when update_state is
            // false; different phases must remain observationally equivalent.
            const bool update_state = iteration % 5 != 0 && iteration % 7 != 0;
            expected.template update<TypeParam>(reference, update_state);
            actual.update(candidate, update_state);
            expect_same_image(reference, candidate);
        }
    }
}

TYPED_TEST(ImageProcessingEquivalence, BeamCorrectionShapeChangesAndMaskedColumns) {
    BeamUniformityCorrector actual;
    image_processing_legacy::BeamUniformityCorrector expected;
    const int heights[] = {1, 8, 16, 128};
    const int widths[] = {0, 1, 2, 17, 128, 2048};
    const Pattern patterns[] = {Pattern::Dense, Pattern::MaskedColumns, Pattern::Zero, Pattern::Ties};
    for (int iteration = 0; iteration < 48; ++iteration) {
        SCOPED_TRACE(::testing::Message() << "frame=" << iteration);
        const int height = heights[(iteration / 3) % 4];
        const int width = widths[iteration % 6];
        auto reference = frame<TypeParam>(patterns[iteration % 4], iteration, height, width);
        auto candidate = reference;
        // A changed height recomputes dark counts even with update_state=false.
        const bool update_state = iteration % 3 == 2;
        expected.template update<TypeParam>(reference, update_state);
        actual.update(candidate, update_state);
        expect_same_image(reference, candidate);
    }
}

TYPED_TEST(ImageProcessingEquivalence, BeamCorrectionFalseUpdatesStillAdvanceCadence) {
    BeamUniformityCorrector actual;
    image_processing_legacy::BeamUniformityCorrector expected;
    for (int iteration = 0; iteration < 40; ++iteration) {
        SCOPED_TRACE(::testing::Message() << "frame=" << iteration);
        auto reference = frame<TypeParam>(Pattern::Dense, iteration);
        auto candidate = reference;
        const bool update_state = iteration == 0 || iteration >= 23;
        expected.template update<TypeParam>(reference, update_state);
        actual.update(candidate, update_state);
        expect_same_image(reference, candidate);
    }
}
}  // namespace
