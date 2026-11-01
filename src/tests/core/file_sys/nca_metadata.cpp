// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "core/file_sys/nca_metadata.h"
#include "tests/core/file_sys/metadata_test_file.h"

namespace FileSys {

TEST_CASE("CNMT serializes platform values", "[core][file_sys]") {
    const auto platform = GENERATE(ContentMetaPlatform::Nx, ContentMetaPlatform::Ounce,
                                   static_cast<ContentMetaPlatform>(0x7f));
    CNMTHeader header{};
    header.title_id = 0x0100000000000000;
    header.type = TitleType::Application;
    header.platform = platform;
    header.table_offset = sizeof(OptionalHeader);
    CNMT metadata(header, {}, {}, {});

    const auto serialized = metadata.Serialize();
    REQUIRE(serialized[0xD] == static_cast<u8>(platform));
    CNMT parsed(std::make_shared<Tests::MetadataFile>(serialized));
    REQUIRE(parsed.GetPlatform() == platform);
    REQUIRE(parsed.Serialize() == serialized);
}

TEST_CASE("CNMT does not merge platforms", "[core][file_sys]") {
    CNMTHeader nx_header{};
    nx_header.type = TitleType::Application;
    nx_header.platform = ContentMetaPlatform::Nx;
    nx_header.table_offset = sizeof(OptionalHeader);
    auto ounce_header = nx_header;
    ounce_header.platform = ContentMetaPlatform::Ounce;
    ounce_header.number_content_entries = 1;

    ContentRecord content{};
    content.nca_id[0] = 1;
    CNMT nx(nx_header, {}, {}, {});
    CNMT ounce(ounce_header, {}, {content}, {});
    REQUIRE_FALSE(nx.UnionRecords(ounce));
    REQUIRE(nx.GetContentRecords().empty());

    auto target_header = ounce_header;
    target_header.number_content_entries = 0;
    CNMT ounce_target(target_header, {}, {}, {});
    REQUIRE(ounce_target.UnionRecords(ounce));
    REQUIRE(ounce_target.GetContentRecords().size() == 1);
}

} // namespace FileSys
