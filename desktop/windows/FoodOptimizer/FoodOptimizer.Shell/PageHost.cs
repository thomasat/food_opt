using FoodOptimizer.Core;
using Microsoft.Web.WebView2.Core;

namespace FoodOptimizer.Shell;

/// <summary>Shows the shared setup and error page (desktop/shared/pages)
/// from https://page.foodopt.example/ and runs page calls on it once it has
/// loaded. Also navigates to the app once the server is up.</summary>
sealed class PageHost
{
    readonly Messages _messages;
    readonly InstallLayout _layout;
    readonly ShellLog _log;
    readonly Queue<string> _queue = new();
    CoreWebView2? _core;
    string? _bootScriptId;
    bool _pageReady;
    bool _hasPending;
    PageView? _pendingView;
    PageView? _lastView;
    readonly List<string> _history = [];
    const int HistoryLimit = 200;
    int _showGeneration;

    public PageHost(Messages messages, InstallLayout layout, ShellLog log)
    {
        _messages = messages;
        _layout = layout;
        _log = log;
    }

    public bool ShowingApp { get; private set; }

    /// <summary>True from <see cref="ShowPageAsync"/> until that page has loaded.</summary>
    public bool ExpectingPage { get; private set; }

    public event EventHandler? RetryRequested;

    /// <summary>Wires up a (new) web view and shows what was asked for meanwhile.</summary>
    public void Attach(CoreWebView2 core, Uri? app)
    {
        _core = core;
        _bootScriptId = null;
        _pageReady = false;
        core.SetVirtualHostNameToFolderMapping(PageBridge.Host, _layout.PagesDir, CoreWebView2HostResourceAccessKind.Deny);
        core.NavigationCompleted += OnNavigationCompleted;
        core.WebMessageReceived += OnWebMessage;
        if (app != null)
        {
            NavigateToApp(app);
            return;
        }
        // A new web view (first, or after a browser-process crash): show the
        // page again and replay what it had been told, so a status line or
        // an error page is never lost behind a blank view.
        var view = _hasPending ? _pendingView : _lastView;
        var replay = _history.ToArray();
        _hasPending = false;
        _pendingView = null;
        _ = ShowPageAsync(view);
        foreach (var js in replay) Remember(js);
    }

    void Remember(string js)
    {
        _queue.Enqueue(js);
        _history.Add(js);
        if (_history.Count > HistoryLimit) _history.RemoveAt(0);
    }

    public void Detach()
    {
        if (_core != null)
        {
            try
            {
                _core.NavigationCompleted -= OnNavigationCompleted;
                _core.WebMessageReceived -= OnWebMessage;
            }
            catch (Exception) { }
        }
        _core = null;
        _pageReady = false;
    }

    public async Task ShowPageAsync(PageView? view)
    {
        _queue.Clear();
        _history.Clear();
        _lastView = view;
        ShowingApp = false;
        _pageReady = false;
        ExpectingPage = true;
        var core = _core;
        if (core == null)
        {
            _hasPending = true;
            _pendingView = view;
            return;
        }
        var generation = ++_showGeneration;
        try
        {
            if (_bootScriptId != null) core.RemoveScriptToExecuteOnDocumentCreated(_bootScriptId);
            _bootScriptId = await core.AddScriptToExecuteOnDocumentCreatedAsync(
                PageBridge.BootScript(_messages, AppIdentity.Version, view));
            if (generation != _showGeneration || core != _core) return;
            core.Navigate(PageBridge.StatusPageUrl);
        }
        catch (Exception e)
        {
            _log.Write("page.show_failed", e.GetType().Name);
        }
    }

    /// <summary>Runs a page call once the page is showing; brings the page
    /// back first when the app is showing.</summary>
    public void Run(string js)
    {
        if (ShowingApp) _ = ShowPageAsync(null);
        Remember(js);
        if (_pageReady) Flush();
    }

    public void NavigateToApp(Uri uri)
    {
        _queue.Clear();
        ShowingApp = true;
        ExpectingPage = false;
        _pageReady = false;
        _hasPending = false;
        _showGeneration++;
        _log.Write("app.navigate", uri.AbsoluteUri);
        try { _core?.Navigate(uri.AbsoluteUri); }
        catch (Exception e) { _log.Write("app.navigate_failed", e.GetType().Name); }
    }

    void Flush()
    {
        var core = _core;
        if (core == null) return;
        while (_queue.Count > 0)
        {
            var js = _queue.Dequeue();
            try { _ = core.ExecuteScriptAsync(js); }
            catch (Exception e) { _log.Write("page.script_failed", e.GetType().Name); }
        }
    }

    void OnNavigationCompleted(object? sender, CoreWebView2NavigationCompletedEventArgs e)
    {
        if (ShowingApp || _core == null) return;
        if (e.IsSuccess && _core.Source.StartsWith(PageBridge.Origin + "/", StringComparison.Ordinal))
        {
            ExpectingPage = false;
            _pageReady = true;
            Flush();
        }
    }

    void OnWebMessage(object? sender, CoreWebView2WebMessageReceivedEventArgs e)
    {
        string? message = null;
        try { message = e.TryGetWebMessageAsString(); }
        catch (Exception) { }
        if (e.Source.StartsWith(PageBridge.Origin + "/", StringComparison.Ordinal) && message == PageBridge.RetryMessage)
            RetryRequested?.Invoke(this, EventArgs.Empty);
        else
            _log.Write("message.ignored", e.Source);
    }
}
