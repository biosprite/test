// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// This process only runs on Windows.
// Technically you could port it to other platforms pretty easily.

#include <cstdint>
#include <string>
#include <vector>
#include <charconv>
#include <print>
#include <filesystem>

#include <boost/asio/io_context.hpp>
#include <boost/process/v2/process.hpp>

#include <windows.h>

bool waitForPid(uint64_t pid) {
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, DWORD(pid));
    if (!h) return false;

    DWORD result = WaitForSingleObject(h, 60);
    CloseHandle(h);
    return result == WAIT_OBJECT_0;
}

// TODO: Log file.
int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    std::vector<std::string> args{argv + 1, argv + argc};

    if (args.size() < 4)
        return 1;

    // pid
    uint64_t pid = 0;
    const auto pid_str = args[0];
    auto fres = std::from_chars(pid_str.data(), pid_str.data() + pid_str.size(), pid);

    if (fres.ec != std::errc{}) {
        std::println(stderr, "Invalid PID {}", pid_str);
        return 2;
    }

    // "Current" app dir
    const auto app_dir = args[1];

    // Old directory (move app dir to this if user asked)
    const auto old_dir = args[2];

    // Staging dir (contains new executable)
    const auto staging_dir = args[3];

    // Whether or not to keep the old dir (if arg is 1)
    const auto keep_old_str = args[4];
    const auto keep_old = keep_old_str == "1";

    // Wait for old PID.
    if (!waitForPid(pid)) {
        std::println(stderr, "Old process did not exit cleanly after 60 seconds.");
        return 3;
    }

    // Remove or rename app dir
    if (keep_old)
        fs::rename(app_dir, old_dir);
    else
        fs::remove_all(app_dir);

    // Rename staging dir
    fs::rename(staging_dir, app_dir);

    // spawn new proc and leave
    // TODO(crueter); configurable exe name?
    namespace bp = boost::process::v2;
    const auto exe = fs::path{app_dir} / "eden.exe";

    boost::asio::io_context ctx;
    bp::process child(ctx, exe.string(), {});
    child.detach();

    return 0;
}
