# SPDX-License-Identifier: GPL-3.0-or-later

# Aplicativo UWP do Eden (somente quando CMAKE_SYSTEM_NAME e WindowsStore)
add_executable(eden-uwp main.cpp)

set_target_properties(eden-uwp PROPERTIES WIN32_EXECUTABLE ON)

target_link_libraries(eden-uwp PRIVATE WindowsApp.lib)
target_link_options(eden-uwp PRIVATE /APPCONTAINER)
