// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <atomic>
#include <cstddef>
#include <list>
#include <memory>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "common/common_types.h"
#include "input_common/input_engine.h"

namespace InputCommon {

/**
 * Reads the motion of Valve's controllers straight from their raw reports, next to Steam, which
 * keeps them: nothing is sent to them but the request to turn their motion sensors on while they
 * are silent. Reads the Steam Deck and the Steam Controller (2026) (see Scan).
 * Experimental (Read Valve Controllers Directly), under Steam in Modern Console; Linux only.
 */
class ValveMotion final : public InputEngine {
public:
    explicit ValveMotion(std::string input_engine_);
    ~ValveMotion() override;

    /// Whether the reader runs this session (decided at startup)
    [[nodiscard]] static bool Enabled();

    /// A Valve controller being read: its port in this engine, its device and product, whether
    /// a button is held (to tell identical controllers apart) and whether its motion arrives
    struct Device {
        std::size_t port{};
        std::string path;
        u16 product{};
        bool button_held{};
        bool moving{};
        /// Its last report: a dongle's slot stays open when its controller turns off, and a
        /// device that stopped reporting is no longer one
        std::chrono::steady_clock::time_point seen{};
    };
    [[nodiscard]] std::vector<Device> GetDevices() const;

    /// The product family of a Valve controller (the Deck, the Steam Controller or the Steam
    /// Controller (2026), whatever its connection), 0 for others
    [[nodiscard]] static int Family(u16 product);

private:
    void Scan(std::stop_token stop);
    void Read(std::stop_token stop, std::string path, std::size_t port, u16 product);

    mutable std::mutex mutex;
    std::map<std::string, Device> devices;
    std::set<std::string> reading;
    /// Nodes that couldn't be opened: tried again only once they are gone and back (else a new
    /// reader every scan, every 2 seconds)
    std::set<std::string> unreadable;
    std::size_t next_port{};
    /// The readers, with whether each is done (a done one is dropped at the next scan)
    struct Reader {
        std::shared_ptr<std::atomic_bool> done;
        std::jthread thread;
    };
    std::list<Reader> readers;
    std::jthread scanner;
};

} // namespace InputCommon
