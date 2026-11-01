#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "common/fs/path_util.h"
#include "core/core.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "core/file_sys/vfs/vfs_vector.h"
#include "core/hle/kernel/k_process.h"
#include "core/loader/nsp.h"

namespace {

class TestNand {
public:
    TestNand() : previous(Common::FS::GetEdenPath(Common::FS::EdenPath::NANDDir)) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        root = std::filesystem::temp_directory_path() / ("eden-loader-test-" + std::to_string(stamp));
        REQUIRE(std::filesystem::create_directory(root));
        Common::FS::SetEdenPath(Common::FS::EdenPath::NANDDir, root);
    }

    ~TestNand() {
        Common::FS::SetEdenPath(Common::FS::EdenPath::NANDDir, previous);
        std::error_code error;
        if (std::filesystem::weakly_canonical(root).parent_path() ==
            std::filesystem::weakly_canonical(std::filesystem::temp_directory_path())) {
            std::filesystem::remove_all(root, error);
        }
    }

private:
    std::filesystem::path previous;
    std::filesystem::path root;
};

void Put32(std::vector<u8>& bytes, std::size_t offset, u32 value) {
    const u32_le little = value;
    std::memcpy(bytes.data() + offset, &little, sizeof(little));
}

void Put64(std::vector<u8>& bytes, std::size_t offset, u64 value) {
    const u64_le little = value;
    std::memcpy(bytes.data() + offset, &little, sizeof(little));
}

std::vector<u8> Npdm(bool ounce) {
    const u32 body = ounce ? 0x300U : 0x200U;
    const u32 aci = ounce ? 0x400U : 0x300U;
    std::vector<u8> bytes(aci + 0x60);
    std::memcpy(bytes.data(), "META", 4);
    bytes[0xC] = 0x27;
    bytes[0xE] = 0x2C;
    Put32(bytes, 0x70, aci);
    Put32(bytes, 0x74, 0x60);
    Put32(bytes, 0x78, 0x80);
    Put32(bytes, 0x7C, body + 0x74);
    std::memcpy(bytes.data() + 0x80 + body, "ACID", 4);
    Put32(bytes, 0x80 + body + 0x20, body + 0x40);
    Put32(bytes, 0x80 + body + 0x24, 0x2C);
    Put32(bytes, 0x80 + body + 0x30, body + 0x70);
    Put32(bytes, 0x80 + body + 0x34, 4);
    bytes[0x80 + body + 0x40] = 1;
    std::memcpy(bytes.data() + aci, "ACI0", 4);
    Put64(bytes, aci + 0x10, 0x0400000000010000ULL);
    Put32(bytes, aci + 0x20, 0x40);
    Put32(bytes, aci + 0x24, 0x1C);
    Put32(bytes, aci + 0x30, 0x5C);
    Put32(bytes, aci + 0x34, 4);
    bytes[aci + 0x40] = 1;
    return bytes;
}

FileSys::VirtualFile Package(bool ounce, bool with_romfs) {
    std::vector<std::pair<std::string, std::vector<u8>>> entries{
        {"main", {}}, {"main.npdm", Npdm(ounce)}};
    if (with_romfs) {
        entries.emplace_back("game.romfs", std::vector<u8>{0x50, 0x12, 0x34});
    }
    std::string names;
    std::vector<u32> name_offsets;
    std::size_t payload_size{};
    for (const auto& [name, data] : entries) {
        name_offsets.push_back(static_cast<u32>(names.size()));
        names += name;
        names.push_back('\0');
        payload_size += data.size();
    }
    const auto header_size = 0x10 + entries.size() * 0x18 + names.size();
    std::vector<u8> bytes(header_size + payload_size);
    std::memcpy(bytes.data(), "PFS0", 4);
    Put32(bytes, 4, static_cast<u32>(entries.size()));
    Put32(bytes, 8, static_cast<u32>(names.size()));
    std::memcpy(bytes.data() + 0x10 + entries.size() * 0x18, names.data(), names.size());
    std::size_t offset{};
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto entry = 0x10 + i * 0x18;
        const auto& data = entries[i].second;
        Put64(bytes, entry, offset);
        Put64(bytes, entry + 8, data.size());
        Put32(bytes, entry + 0x10, name_offsets[i]);
        if (!data.empty()) {
            std::memcpy(bytes.data() + header_size + offset, data.data(), data.size());
        }
        offset += data.size();
    }
    return std::make_shared<FileSys::VectorVfsFile>(std::move(bytes), "fixture.nsp");
}

}

TEST_CASE("Extracted NSP forwards RomFS and reports a missing image", "[core][loader]") {
    const bool with_romfs = GENERATE(false, true);
    TestNand nand;
    Core::System system;
    FileSys::ContentProviderUnion provider;
    Loader::AppLoader_NSP loader(Package(false, with_romfs), system.GetFileSystemController(),
                                 provider, 0, 0);
    FileSys::VirtualFile romfs = std::make_shared<FileSys::VectorVfsFile>();
    const auto result = loader.ReadRomFS(romfs);
    REQUIRE(result == (with_romfs ? Loader::ResultStatus::Success
                                 : Loader::ResultStatus::ErrorNoRomFS));
    if (with_romfs) {
        REQUIRE(romfs != nullptr);
        REQUIRE(romfs->ReadAllBytes() == std::vector<u8>{0x50, 0x12, 0x34});
    } else {
        REQUIRE(romfs == nullptr);
    }
}

TEST_CASE("Ounce execution is rejected before module loading while NX reaches its loader", "[core][loader]") {
    const bool ounce = GENERATE(false, true);
    TestNand nand;
    Core::System system;
    FileSys::ContentProviderUnion provider;
    Loader::AppLoader_NSP loader(Package(ounce, true), system.GetFileSystemController(), provider, 0, 0);
    Kernel::KProcess process(system.Kernel());
    const auto result = loader.Load(process, system);
    REQUIRE(result.first == (ounce ? Loader::ResultStatus::ErrorUnsupportedSwitch2
                                  : Loader::ResultStatus::ErrorLoadingNSO));
    REQUIRE_FALSE(result.second.has_value());
    REQUIRE_FALSE(system.IsPoweredOn());
}

TEST_CASE("Supplied extracted NSP exposes its RomFS and rejects unsupported execution", "[.local_package][loader]") {
    const char* path = std::getenv("NSP2_SAMPLE_PACKAGE");
    REQUIRE(path != nullptr);
    TestNand nand;
    Core::System system;
    FileSys::ContentProviderUnion provider;
    FileSys::RealVfsFilesystem fs;
    const auto file = fs.OpenFile(path, FileSys::OpenMode::Read);
    REQUIRE(file != nullptr);
    Loader::AppLoader_NSP loader(file, system.GetFileSystemController(), provider, 0, 0);
    FileSys::VirtualFile romfs;
    REQUIRE(loader.ReadRomFS(romfs) == Loader::ResultStatus::Success);
    REQUIRE(romfs != nullptr);
    REQUIRE(romfs->GetSize() > 9'000'000'000ULL);
    REQUIRE(romfs->ReadBytes(8) == std::vector<u8>{0x50, 0, 0, 0, 0, 0, 0, 0});
    Kernel::KProcess process(system.Kernel());
    REQUIRE(loader.Load(process, system).first == Loader::ResultStatus::ErrorUnsupportedSwitch2);
    REQUIRE_FALSE(system.IsPoweredOn());
}
