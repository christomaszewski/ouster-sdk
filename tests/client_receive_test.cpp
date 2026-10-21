/**
 * Copyright (c) 2026, Ouster, Inc.
 * All rights reserved.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "ouster/client.h"
#include "ouster/impl/netcompat.h"
#include "ouster/packet.h"

#ifndef _WIN32
#include <unistd.h>
#endif

namespace {

namespace core = ouster::sdk::core;
namespace sensor = ouster::sdk::sensor;

struct ReceiveProfile {
    bool is_imu;
    size_t packet_size;
    const char* name;
};

// Exercise both old client sockets without connecting to or configuring a
// physical sensor, including the larger 128-beam legacy lidar packets.
class ClientReceiveTest : public ::testing::TestWithParam<ReceiveProfile> {
   protected:
    std::shared_ptr<sensor::Client> client_;
    SOCKET sender_{static_cast<SOCKET>(SOCKET_ERROR)};
    sockaddr_in destination_{};

    void SetUp() override {
        client_ = sensor::init_client("127.0.0.1", 0, 0);
        ASSERT_TRUE(client_);
        sender_ = socket(AF_INET, SOCK_DGRAM, 0);
        ASSERT_NE(sender_, SOCKET_ERROR);
        destination_.sin_family = AF_INET;
        destination_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        destination_.sin_port = htons(GetParam().is_imu
                                         ? sensor::get_imu_port(*client_)
                                         : sensor::get_lidar_port(*client_));
    }

    void TearDown() override {
        if (sender_ != SOCKET_ERROR) {
#ifdef _WIN32
            closesocket(sender_);
#else
            close(sender_);
#endif
        }
    }

    size_t packet_size() const { return GetParam().packet_size; }

    void send(const std::vector<uint8_t>& payload) {
        // A non-null pointer also works for the zero-length datagram test.
        const char empty = 0;
        const char* data = payload.empty()
                               ? &empty
                               : reinterpret_cast<const char*>(payload.data());
        auto sent = sendto(sender_, data, static_cast<int>(payload.size()), 0,
                           reinterpret_cast<const sockaddr*>(&destination_),
                           sizeof(destination_));
        ASSERT_EQ(sent, static_cast<decltype(sent)>(payload.size()));
        auto state = sensor::poll_client(*client_, 1);
        ASSERT_TRUE(state &
                    (GetParam().is_imu ? sensor::IMU_DATA : sensor::LIDAR_DATA));
    }

    bool receive(uint8_t* data) {
        return GetParam().is_imu
                   ? sensor::read_imu_packet(*client_, data, packet_size())
                   : sensor::read_lidar_packet(*client_, data, packet_size());
    }

    bool receive(core::ImuPacket& imu, core::LidarPacket& lidar) {
        return GetParam().is_imu ? sensor::read_imu_packet(*client_, imu)
                                 : sensor::read_lidar_packet(*client_, lidar);
    }
};

TEST_P(ClientReceiveTest, ExactSizedDatagramsArePreserved) {
    std::vector<uint8_t> expected(packet_size());
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<uint8_t>(i);
    }
    std::vector<uint8_t> received(packet_size());
    ASSERT_NO_FATAL_FAILURE(send(expected));
    ASSERT_TRUE(receive(received.data()));
    EXPECT_EQ(received, expected);
}

TEST_P(ClientReceiveTest, RejectsWrongSizesWithoutWritingPastTheBuffer) {
    // The final byte is a canary outside the buffer size passed to the SDK.
    // This catches the regression even when not running under a sanitizer.
    for (size_t size : {size_t{0}, packet_size() - 1, packet_size() + 1,
                        packet_size() + 2048}) {
        std::vector<uint8_t> received(packet_size() + 1, 0x6d);
        ASSERT_NO_FATAL_FAILURE(send(std::vector<uint8_t>(size, 0xa5)));
        EXPECT_FALSE(receive(received.data())) << "payload size: " << size;
        EXPECT_EQ(received.back(), 0x6d) << "payload size: " << size;

        // Rejection must consume exactly one datagram, allowing recovery on
        // the next packet rather than leaving a truncated packet queued.
        std::vector<uint8_t> expected(packet_size(), 0x3c);
        ASSERT_NO_FATAL_FAILURE(send(expected));
        ASSERT_TRUE(receive(received.data()));
        EXPECT_EQ(std::vector<uint8_t>(received.begin(), received.end() - 1),
                  expected);
        EXPECT_EQ(received.back(), 0x6d);
    }
}

TEST_P(ClientReceiveTest, PacketOverloadsStayBoundedAfterRawPublisherSwaps) {
    core::ImuPacket imu;
    core::LidarPacket lidar;
    auto& packet = GetParam().is_imu ? static_cast<core::Packet&>(imu)
                                    : static_cast<core::Packet&>(lidar);
    packet.buf.resize(packet_size());
    std::vector<uint8_t> raw_message(packet_size());

    for (int i = 0; i < 4; ++i) {
        std::vector<uint8_t> expected(packet_size(), static_cast<uint8_t>(i));
        ASSERT_NO_FATAL_FAILURE(send(expected));
        ASSERT_TRUE(receive(imu, lidar));
        EXPECT_GT(packet.host_timestamp, 0u);
        EXPECT_EQ(packet.buf, expected);

        // This mirrors ouster_ros's raw publication, including replacement of
        // the packet's allocation by a vector without reserved spare bytes.
        packet.buf.swap(raw_message);
        EXPECT_EQ(raw_message, expected);
        for (size_t extra : {size_t{1}, size_t{2048}}) {
            ASSERT_NO_FATAL_FAILURE(
                send(std::vector<uint8_t>(packet_size() + extra, 0xa5)));
            EXPECT_FALSE(receive(imu, lidar));
            EXPECT_EQ(packet.buf.size(), packet_size());
            // The allocation handed to the raw publisher must remain intact
            // while the SDK receives into its replacement allocation.
            EXPECT_EQ(raw_message, expected);
        }
    }
}

INSTANTIATE_TEST_CASE_P(
    LegacyProfiles, ClientReceiveTest,
    ::testing::Values(ReceiveProfile{true, 48, "Imu"},
                      ReceiveProfile{false, 12608, "Lidar64"},
                      ReceiveProfile{false, 24896, "Lidar128"}),
    [](const ::testing::TestParamInfo<ReceiveProfile>& info) {
        return info.param.name;
    });

}  // namespace
