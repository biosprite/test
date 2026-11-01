// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include "core/file_sys/vfs/vfs.h"

namespace FileSys::Tests {

class MetadataFile final : public VfsFile {
public:
    explicit MetadataFile(std::vector<u8> bytes_) : bytes(std::move(bytes_)) {}
    std::string GetName() const override { return "metadata"; }
    std::size_t GetSize() const override { return bytes.size(); }
    bool Resize(std::size_t) override { return false; }
    VirtualDir GetContainingDirectory() const override { return nullptr; }
    bool IsWritable() const override { return false; }
    bool IsReadable() const override { return true; }
    std::size_t Read(u8* out, std::size_t length, std::size_t offset) const override {
        if (offset >= bytes.size()) {
            return 0;
        }
        const auto size = (std::min)(length, bytes.size() - offset);
        if (size == 0) {
            return 0;
        }
        std::memcpy(out, bytes.data() + offset, size);
        return size;
    }
    std::size_t Write(const u8*, std::size_t, std::size_t) override { return 0; }
    bool Rename(std::string_view) override { return false; }

private:
    std::vector<u8> bytes;
};

} // namespace FileSys::Tests
