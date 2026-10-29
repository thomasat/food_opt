using FoodOptimizer.Core;
using FoodOptimizer.Core.Windows;
using Microsoft.Web.WebView2.Core;

namespace FoodOptimizer.Shell;

/// <summary>Keeps the web view an app window: navigations stay on our
/// server and page, web links open in the default browser, nothing loads
/// from the internet, the page's alert/confirm/prompt are native Food
/// Optimizer dialogs and the right-click menu keeps only editing commands.
/// Attached again to each new web view after a browser-process crash.
/// The addresses it logs leave out the base path (M3).</summary>
sealed class WebRules
{
    const int BlockedLogLimit = 20;

    readonly Form _owner;
    readonly LaunchController _launch;
    readonly PageHost _page;
    readonly ShellLog _log;
    int _blockedLogged;

    public WebRules(Form owner, LaunchController launch, PageHost page, ShellLog log)
    {
        _owner = owner;
        _launch = launch;
        _page = page;
        _log = log;
    }

    NavContext Context => new(_launch.Port, _launch.Loaded, _page.ExpectingPage);

    public void Attach(CoreWebView2 core, CoreWebView2Environment env)
    {
        core.NavigationStarting += OnNavigationStarting;
        core.FrameNavigationStarting += OnFrameNavigationStarting;
        core.NewWindowRequested += OnNewWindowRequested;
        core.AddWebResourceRequestedFilter("*", CoreWebView2WebResourceContext.All,
            CoreWebView2WebResourceRequestSourceKinds.All);
        core.WebResourceRequested += (_, e) => OnWebResourceRequested(env, e);
        core.ContextMenuRequested += OnContextMenuRequested;
        core.ScriptDialogOpening += OnScriptDialogOpening;
    }

    void OpenExternally(string uri)
    {
        _log.Write("open.external", LogRedaction.Redact(uri, _launch.Base));
        _ = WinShell.Open(uri);
    }

    void OnNavigationStarting(object? sender, CoreWebView2NavigationStartingEventArgs e)
    {
        var history = e.NavigationKind == CoreWebView2NavigationKind.BackOrForward;
        switch (NavigationRules.ForMainFrame(e.Uri, history, Context))
        {
            case NavDecision.Allow:
                return;
            case NavDecision.OpenExternally:
                e.Cancel = true;
                OpenExternally(e.Uri);
                return;
            default:
                e.Cancel = true;
                if (history) _log.Write("nav.history_blocked");
                else _log.Write("nav.blocked", LogRedaction.Redact(e.Uri, _launch.Base));
                return;
        }
    }

    void OnFrameNavigationStarting(object? sender, CoreWebView2NavigationStartingEventArgs e)
    {
        if (NavigationRules.ForFrame(e.Uri, Context) == NavDecision.Allow) return;
        e.Cancel = true;
        _log.Write("frame.blocked", LogRedaction.Redact(e.Uri, _launch.Base));
    }

    void OnNewWindowRequested(object? sender, CoreWebView2NewWindowRequestedEventArgs e)
    {
        e.Handled = true;
        if (NavigationRules.ForNewWindow(e.Uri) == NavDecision.OpenExternally) OpenExternally(e.Uri);
        else _log.Write("window.blocked", LogRedaction.Redact(e.Uri, _launch.Base));
    }

    void OnWebResourceRequested(CoreWebView2Environment env, CoreWebView2WebResourceRequestedEventArgs e)
    {
        var uri = e.Request.Uri;
        if (NavigationRules.IsResourceAllowed(uri, _launch.Port)) return;
        e.Response = env.CreateWebResourceResponse(null, 403, "Blocked", "");
        if (_blockedLogged++ < BlockedLogLimit) _log.Write("request.blocked", LogRedaction.Redact(uri, _launch.Base));
    }

    void OnContextMenuRequested(object? sender, CoreWebView2ContextMenuRequestedEventArgs e)
    {
        var items = e.MenuItems;
        for (var i = items.Count - 1; i >= 0; i--)
            if (!ContextMenuRules.Keep(items[i].Name)) items.RemoveAt(i);
        if (items.Count == 0)
        {
            e.Handled = true;
            _log.Write("context_menu.suppressed");
        }
        else
        {
            _log.Write("context_menu.kept", string.Join(",", items.Select(i => i.Name)));
        }
    }

    void OnScriptDialogOpening(object? sender, CoreWebView2ScriptDialogOpeningEventArgs e)
    {
        var deferral = e.GetDeferral();
        _owner.BeginInvoke(() =>
        {
            try
            {
                switch (e.Kind)
                {
                    case CoreWebView2ScriptDialogKind.Alert:
                        ShellDialogs.Info(_owner, "", e.Message);
                        break;
                    case CoreWebView2ScriptDialogKind.Confirm:
                        if (ShellDialogs.Confirm(_owner, e.Message)) e.Accept();
                        break;
                    case CoreWebView2ScriptDialogKind.Prompt:
                        if (ShellDialogs.Prompt(_owner, e.Message, e.DefaultText) is { } text)
                        {
                            e.ResultText = text;
                            e.Accept();
                        }
                        break;
                    case CoreWebView2ScriptDialogKind.Beforeunload:
                        e.Accept();
                        break;
                }
            }
            catch (Exception ex)
            {
                _log.Write("dialog.failed", ex.GetType().Name);
            }
            finally
            {
                deferral.Complete();
            }
        });
    }
}
