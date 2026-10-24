using FoodOptimizer.Core;
using FoodOptimizer.Core.Windows;
using Microsoft.Web.WebView2.Core;

namespace FoodOptimizer.Shell;

/// <summary>The Food Optimizer window: the web view that shows the setup
/// page and then the app, opened at once while the launcher starts.</summary>
sealed class MainForm : Form
{
    static readonly Color Background = Color.FromArgb(0xF7, 0xF6, 0xF2);

    readonly Messages _messages;
    readonly UserPaths _paths;
    readonly InstallLayout _layout;
    readonly ShellSettings _settings;
    readonly ShellLog _log;
    readonly WebViewHost _web = new() { Dock = DockStyle.Fill };
    readonly PageHost _page;
    readonly LaunchController _launch;
    int _recreations;
    bool _sessionEnding;

    const int WM_QUERYENDSESSION = 0x11, WM_ENDSESSION = 0x16;

    public MainForm(Messages messages, UserPaths paths, InstallLayout layout, ShellSettings settings, ShellLog log)
    {
        _messages = messages;
        _paths = paths;
        _layout = layout;
        _settings = settings;
        _log = log;
        Text = messages.Text("window.title");
        BackColor = Background;
        StartPosition = FormStartPosition.Manual;
        AutoScaleMode = AutoScaleMode.Dpi;
        Controls.Add(_web);
        _page = new PageHost(messages, layout, log);
        _launch = new LaunchController(this, _page, layout, paths, settings, log);
        _page.RetryRequested += async (_, _) => await _launch.RetryAsync();
    }

    protected override void OnLoad(EventArgs e)
    {
        base.OnLoad(e);
        var placement = WindowPlacement.Initial(Monitors(), _settings.Window);
        var (minW, minH) = WindowPlacement.MinimumSize(DeviceDpi);
        MinimumSize = new Size(minW, minH);
        var b = placement.Bounds;
        Bounds = new Rectangle(b.X, b.Y, b.Width, b.Height);
        if (placement.Maximized) WindowState = FormWindowState.Maximized;
    }

    static IReadOnlyList<MonitorInfo> Monitors() =>
        Screen.AllScreens.Select(s =>
        {
            var wa = s.WorkingArea;
            return new MonitorInfo(s.DeviceName, new PxRect(wa.X, wa.Y, wa.Width, wa.Height), DpiOf(s), s.Primary);
        }).ToList();

    static int DpiOf(Screen s)
    {
        var center = new NativeMethods.POINT { X = s.Bounds.X + s.Bounds.Width / 2, Y = s.Bounds.Y + s.Bounds.Height / 2 };
        var monitor = NativeMethods.MonitorFromPoint(center, NativeMethods.MONITOR_DEFAULTTONEAREST);
        return NativeMethods.GetDpiForMonitor(monitor, NativeMethods.MDT_EFFECTIVE_DPI, out var x, out _) == 0 ? (int)x : 96;
    }

    protected override async void OnShown(EventArgs e)
    {
        base.OnShown(e);
        _launch.Start();
        await CreateWebViewAsync();
    }

    async Task CreateWebViewAsync()
    {
        try
        {
            var env = await WebEnvironment.CreateAsync(_paths);
            _log.Write("webview.runtime", env.BrowserVersionString);
            await _web.InitializeAsync(env, _settings.Zoom);
            AttachCore();
        }
        catch (Exception ex)
        {
            _log.Write("webview.failed", ex.GetType().Name);
        }
    }

    /// <summary>Wires a (new) web view: the page, crash recovery and the
    /// load-failure check.</summary>
    void AttachCore()
    {
        var core = _web.Core!;
        core.ProcessFailed += OnProcessFailed;
        core.NavigationCompleted += (_, e) =>
        {
            if (!e.IsSuccess && e.WebErrorStatus != CoreWebView2WebErrorStatus.OperationCanceled
                && _page.ShowingApp && _launch.Loaded)
            {
                _log.Write("app.load_failed", e.WebErrorStatus.ToString());
                _launch.OnLoadFailed();
            }
        };
        _page.Attach(core, _launch.Loaded ? _launch.AppUri : null);
    }

    void OnProcessFailed(object? sender, CoreWebView2ProcessFailedEventArgs e)
    {
        switch (e.ProcessFailedKind)
        {
            case CoreWebView2ProcessFailedKind.RenderProcessExited:
                _log.Write("webview.renderer_exited", e.Reason.ToString());
                try { _web.Core?.Reload(); } catch (Exception) { }
                break;
            case CoreWebView2ProcessFailedKind.BrowserProcessExited:
                _log.Write("webview.browser_exited", e.Reason.ToString());
                BeginInvoke(async () => await RecreateAsync());
                break;
            default:
                _log.Write("webview.process_failed", e.ProcessFailedKind + " " + e.Reason);
                break;
        }
    }

    async Task RecreateAsync()
    {
        if (IsDisposed || _sessionEnding) return;
        if (_recreations >= 3)
        {
            _log.Write("webview.gave_up");
            ShellDialogs.Info(this, _messages.Text("error.stopped.title"), _messages.Text("error.stopped.body"));
            Close();
            return;
        }
        _recreations++;
        _page.Detach();
        try
        {
            await _web.RecreateAsync(() => WebEnvironment.CreateAsync(_paths), _settings.Zoom);
            AttachCore();
            _log.Write("webview.recreated", _recreations.ToString(System.Globalization.CultureInfo.InvariantCulture));
        }
        catch (Exception ex)
        {
            _log.Write("webview.recreate_failed", ex.GetType().Name);
        }
    }

    /// <summary>Logoff and shutdown: quit at once, and show no error page.</summary>
    protected override void WndProc(ref Message m)
    {
        if (m.Msg == WM_QUERYENDSESSION)
        {
            _log.Write("session.ending");
            _launch.BeginShutdown();
            m.Result = 1;
            return;
        }
        if (m.Msg == WM_ENDSESSION)
        {
            if (m.WParam != IntPtr.Zero)
            {
                _sessionEnding = true;
                _launch.Shutdown(fast: true);
                Close();
            }
            m.Result = IntPtr.Zero;
            return;
        }
        base.WndProc(ref m);
    }

    /// <summary>The shell crashed: stop the launch at once.</summary>
    public void EmergencyShutdown()
    {
        try { _launch.Shutdown(fast: true); } catch (Exception) { }
    }

    protected override void OnFormClosing(FormClosingEventArgs e)
    {
        base.OnFormClosing(e);
        if (e.Cancel) return;
        var screen = Screen.FromControl(this).DeviceName;
        var bounds = WindowState == FormWindowState.Normal ? Bounds : RestoreBounds;
        _settings.Window = WindowPlacement.ToSaved(screen,
            new PxRect(bounds.X, bounds.Y, bounds.Width, bounds.Height), WindowState == FormWindowState.Maximized);
        _settings.Save(_paths.SettingsPath);
        Hide();
        _launch.Shutdown(fast: _sessionEnding || e.CloseReason == CloseReason.WindowsShutDown);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) _launch.Dispose();
        base.Dispose(disposing);
    }
}
