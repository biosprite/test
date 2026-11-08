// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <QString>

namespace QtCommon::Mod {

QStringList GetModFolders(const QString& root, const QString& fallbackName, bool prompt_name = true);

// Extracts each zip to its own temporary directory. Zips that fail to extract get an empty string.
QStringList ExtractMods(const QStringList& paths);

} // namespace QtCommon::Mod
