// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader {
struct HostTranslateInfo;
}

namespace Shader::Maxwell {

[[nodiscard]] IR::Program TranslateProgram(ObjectPool<IR::Inst>& inst_pool,
                                           ObjectPool<IR::Block>& block_pool, Environment& env,
                                           Flow::CFG& cfg, const HostTranslateInfo& host_info);

[[nodiscard]] IR::Program MergeDualVertexPrograms(IR::Program& vertex_a, IR::Program& vertex_b,
                                                  Environment& env_vertex_b);

void ConvertLegacyToGeneric(IR::Program& program, const RuntimeInfo& runtime_info);

// Folds a passthrough geometry shader into the vertex program for hosts without geometry
// shaders but with Layer/ViewportIndex outputs in vertex shaders. The geometry program must be
// translated with support_geometry_shader_passthrough set so it was not lowered to EmitVertex.
// Returns std::nullopt when the geometry program can't be expressed per vertex.
// Constant buffers read by the geometry program are moved to vertex slots it doesn't use,
// cbuf_remap receives the (geometry slot, vertex slot) pairs the host has to bind.
// failure_reason, when not null, receives why the fold was rejected.
[[nodiscard]] std::optional<IR::Program> FoldGeometryPassthrough(
    IR::Program& vertex, IR::Program& geometry, Environment& env_vertex,
    std::vector<std::pair<u32, u32>>& cbuf_remap, std::string_view* failure_reason = nullptr);

// Maxwell v1 and older Nvidia cards don't support setting gl_Layer from non-geometry stages.
// This creates a workaround by setting the layer as a generic output and creating a
// passthrough geometry shader that reads the generic and sets the layer.
[[nodiscard]] IR::Program GenerateGeometryPassthrough(ObjectPool<IR::Inst>& inst_pool,
                                                      ObjectPool<IR::Block>& block_pool,
                                                      const HostTranslateInfo& host_info,
                                                      IR::Program& source_program,
                                                      Shader::OutputTopology output_topology);

} // namespace Shader::Maxwell
