// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/hle/service/cmif_serialization.h"
#include "core/hle/service/vi/application_display_service.h"
#include "core/hle/service/vi/application_root_service.h"
#include "core/hle/service/vi/container.h"
#include "core/hle/service/vi/service_creator.h"
#include "core/hle/service/vi/vi.h"
#include "core/hle/service/vi/vi_types.h"

namespace Service::VI {

IApplicationRootService::IApplicationRootService(Core::System& system_,
                                                 std::shared_ptr<Container> container)
    : ServiceFramework{system_, "vi:u"}, m_container{std::move(container)} {
    static const FunctionInfo functions[] = {
        {0, C<&IApplicationRootService::GetDisplayService>, "GetDisplayService"},
        {1, C<&IApplicationRootService::GetDisplayServiceWithProxyNameExchange>, "GetDisplayServiceWithProxyNameExchange"},
    };
    RegisterHandlers(functions);
}

IApplicationRootService::~IApplicationRootService() = default;

Result IApplicationRootService::GetDisplayService(
    Out<SharedPointer<IApplicationDisplayService>> out_application_display_service, Policy policy) {
    LOG_DEBUG(Service_VI, "called");
    R_RETURN(GetApplicationDisplayService(out_application_display_service, system, m_container,
                                          Permission::User, policy));
}

Result IApplicationRootService::GetDisplayServiceWithProxyNameExchange(
    Out<SharedPointer<IApplicationDisplayService>> out_application_display_service, Policy policy) {
    // Same input as GetDisplayService for vi:u (only the vi:m variant also takes a proxy name).
    // Used by applications when vi:m and vi:s are denied by their service access control.
    LOG_DEBUG(Service_VI, "called, policy={}", policy);
    R_RETURN(GetApplicationDisplayService(out_application_display_service, system, m_container,
                                          Permission::User, policy));
}

} // namespace Service::VI
