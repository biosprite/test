// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include "dynarmic/tests/rand_int.h"

namespace {

// Reseeds RandInt from --rng-seed before every run of a test case, so any randomized failure
// replays with: dynarmic_tests "<test name>" --rng-seed <seed printed at start-up>.
class RandIntSeeder final : public Catch::EventListenerBase {
public:
    using Catch::EventListenerBase::EventListenerBase;

    void testCasePartialStarting(Catch::TestCaseInfo const&, uint64_t) override {
        detail::g_rand_int_generator.seed(Catch::getSeed());
    }
};

}  // namespace

CATCH_REGISTER_LISTENER(RandIntSeeder)
