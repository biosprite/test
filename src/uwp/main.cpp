// SPDX-License-Identifier: GPL-3.0-or-later
//
// eden-UWP: ponto de entrada minimo do aplicativo UWP (C++/WinRT, sem XAML).
// Primeiro objetivo: abrir uma janela vazia, so para validar compilacao,
// pacote, assinatura e instalacao. O emulador sera ligado aqui depois.

#include <windows.h>

#include <winrt/Windows.ApplicationModel.Activation.h>
#include <winrt/Windows.ApplicationModel.Core.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.UI.Core.h>

using namespace winrt::Windows::ApplicationModel::Activation;
using namespace winrt::Windows::ApplicationModel::Core;
using namespace winrt::Windows::UI::Core;

struct App : winrt::implements<App, IFrameworkViewSource, IFrameworkView> {
    IFrameworkView CreateView() {
        return *this;
    }

    void Initialize(CoreApplicationView const& view) {
        view.Activated({this, &App::OnActivated});
    }

    void SetWindow(CoreWindow const&) {}

    void Load(winrt::hstring const&) {}

    void Run() {
        CoreWindow window = CoreWindow::GetForCurrentThread();
        window.Activate();
        window.Dispatcher().ProcessEvents(CoreProcessEventsOption::ProcessUntilQuit);
    }

    void Uninitialize() {}

    void OnActivated(CoreApplicationView const&, IActivatedEventArgs const&) {
        CoreWindow::GetForCurrentThread().Activate();
    }
};

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    CoreApplication::Run(winrt::make<App>());
    return 0;
}
