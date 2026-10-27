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
    readonly KeyRouter _keys;
    readonly WebRules _rules;
    readonly Downloads _downloads;
    int _recreations;
    bool _sessionEnding;
    bool _closing;
    // A logoff asked about (WM_QUERYENDSESSION) and not yet over: only its
    // cancellation may undo the shutdown it began, never a user's close.
    bool _logoffQueried;

    const int WM_QUERYENDSESSION = 0x11, WM_ENDSESSION = 0x16;

    public MainForm(Messages messages, UserPaths paths, InstallLayout layout, ShellSettings settings, ShellLog log)
    {
        _messages = messages;
        _paths = paths;
        _layout = layout;
        _settings = settings;
        _log = log;
        Text = messages.Text("window.title");
        using (var icon = typeof(MainForm).Assembly.GetManifestResourceStream("icon.ico"))
            if (icon != null) Icon = new Icon(icon);
        BackColor = Background;
        StartPosition = FormStartPosition.Manual;
        AutoScaleMode = AutoScaleMode.Dpi;
        _keys = KeyRouter.FromMessages(messages);
        // A command runs after the click returns, so a dialog it opens never
        // blocks the menu (or UI Automation's Invoke).
        var menu = MenuBuilder.Build(messages, id => BeginInvoke(() => Execute(id)));
        Controls.Add(_web);
        Controls.Add(menu);
        MainMenuStrip = menu;
        _page = new PageHost(messages, layout, log);
        _launch = new LaunchController(this, _page, layout, paths, settings, log);
        _page.RetryRequested += async (_, _) => await _launch.RetryAsync();
        _rules = new WebRules(this, _launch, _page, log);
        _downloads = new Downloads(this, _launch, messages, log);
    }

    /// <summary>This window's single-instance claim; released first on close
    /// so reopening the app while this one shuts down opens a new window.</summary>
    [System.ComponentModel.DesignerSerializationVisibility(System.ComponentModel.DesignerSerializationVisibility.Hidden)]
    public SingleInstance? Instance { get; set; }

    /// <summary>A second launch asked for this window (any thread).</summary>
    public void BringBack()
    {
        if (IsDisposed || !IsHandleCreated) return;
        try
        {
            BeginInvoke(() =>
            {
                // SW_RESTORE brings back the state before minimizing (maximized stays maximized).
                if (WindowState == FormWindowState.Minimized) NativeMethods.ShowWindow(Handle, NativeMethods.SW_RESTORE);
                Show();
                Activate();
                NativeMethods.SetForegroundWindow(Handle);
            });
        }
        catch (InvalidOperationException) { }
    }

    /// <summary>The window carries the shortcut's taskbar ID itself, before
    /// it is shown.</summary>
    protected override void OnHandleCreated(EventArgs e)
    {
        base.OnHandleCreated(e);
        if (!WinShell.SetWindowAppUserModelId(Handle, AppIdentity.AppUserModelId))
            _log.Write("appid.window_failed");
    }

    protected override void OnDpiChanged(DpiChangedEventArgs e)
    {
        base.OnDpiChanged(e);
        var (minW, minH) = WindowPlacement.MinimumSize(e.DeviceDpiNew);
        MinimumSize = new Size(minW, minH);
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
        if (!IsDisposed) PinProjectsFolder();
    }

    /// <summary>D2: makes the projects folder and, the first time for this
    /// user (both install types, no admin), pins it to Quick access, here
    /// on the UI thread (Shell COM needs an STA thread). Recorded in
    /// settings.json, so a user who unpins it later keeps it unpinned and
    /// the uninstaller unpins only what the app pinned.</summary>
    void PinProjectsFolder()
    {
        try
        {
            Directory.CreateDirectory(_paths.ProjectsDir);
        }
        catch (Exception ex)
        {
            _log.Write("projects.create_failed", ex.GetType().Name);
            return;
        }
        if (!QuickAccessRule.ShouldPin(_settings) || !QuickAccess.Pin(_paths.ProjectsDir, _log)) return;
        _settings.QuickAccessPinned = true;
        if (!_settings.Save(_paths.SettingsPath)) _log.Write("settings.save_failed");
    }

    /// <summary>Creates the web view, trying again for a few seconds (a
    /// window that just closed may still hold the WebView2 folder). If it
    /// never starts, says so and closes rather than leave a blank window.
    /// A WebView2 folder that cannot be made fails here, before WebView2
    /// starts: WebView2's own box for it waits for a click, and the shell
    /// with it.</summary>
    async Task CreateWebViewAsync()
    {
        const int attempts = 3;
        for (var attempt = 1; ; attempt++)
        {
            try
            {
                Directory.CreateDirectory(_paths.WebViewData);
                var env = await WebEnvironment.CreateAsync(_paths);
                _log.Write("webview.runtime", env.BrowserVersionString);
                await _web.InitializeAsync(env, _settings.Zoom);
                _log.Write("zoom.applied", LaunchController.Invariant(_settings.Zoom));
                AttachCore();
                return;
            }
            catch (Exception ex)
            {
                _log.Write("webview.failed", ex.GetType().Name);
                _web.CloseController();
                if (attempt >= attempts || IsDisposed) break;
                await Task.Delay(1000);
            }
        }
        if (IsDisposed) return;
        ShellDialogs.Info(this, _messages.Text("error.stopped.title"), _messages.Text("error.shell_crashed.body"));
        Close();
    }

    /// <summary>Wires a (new) web view: the page, crash recovery and the
    /// load-failure check.</summary>
    void AttachCore()
    {
        var core = _web.Core!;
        var controller = _web.Controller!;
        controller.AcceleratorKeyPressed += OnAcceleratorKey;
        controller.ZoomFactorChanged += OnZoomChanged;
        core.ProcessFailed += OnProcessFailed;
        _rules.Attach(core, _web.Environment!);
        _downloads.Attach(core);
        core.NavigationCompleted += (_, e) =>
        {
            if (!e.IsSuccess && e.WebErrorStatus != CoreWebView2WebErrorStatus.OperationCanceled
                && _page.ShowingApp && _launch.Loaded)
            {
                _log.Write("app.load_failed", e.WebErrorStatus.ToString());
                _launch.OnLoadFailed();
            }
        };
        _page.Attach(core, _launch.Loaded ? _launch.AppUri : null, _launch.Base);
    }

    static bool Down(int vk) => NativeMethods.GetKeyState(vk) < 0;

    static char? LayoutChar(int vk)
    {
        var c = NativeMethods.MapVirtualKeyW((uint)vk, NativeMethods.MAPVK_VK_TO_CHAR) & 0x7FFF;
        return c == 0 ? null : (char)c;
    }

    /// <summary>Our shortcuts while the web content has the focus.</summary>
    void OnAcceleratorKey(object? sender, CoreWebView2AcceleratorKeyPressedEventArgs e)
    {
        if (e.KeyEventKind != CoreWebView2KeyEventKind.KeyDown) return;
        var vk = (int)e.VirtualKey;
        var id = _keys.Route(vk, LayoutChar(vk), Down(NativeMethods.VK_CONTROL), Down(NativeMethods.VK_SHIFT), Down(NativeMethods.VK_MENU));
        if (id == null) return;
        e.Handled = true;
        BeginInvoke(() => Execute(id));
    }

    /// <summary>Our shortcuts while the menu bar or the form has the focus.</summary>
    protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
    {
        var vk = (int)(keyData & Keys.KeyCode);
        var id = _keys.Route(vk, LayoutChar(vk), (keyData & Keys.Control) != 0, (keyData & Keys.Shift) != 0, (keyData & Keys.Alt) != 0);
        if (id != null)
        {
            Execute(id);
            return true;
        }
        return base.ProcessCmdKey(ref msg, keyData);
    }

    /// <summary>Keeps Ctrl+wheel zoom on our steps and range, and saves it.</summary>
    void OnZoomChanged(object? sender, object e)
    {
        if (_web.Controller is not { } c) return;
        var (reapply, save) = Zoom.OnReported(c.ZoomFactor, _settings.Zoom);
        if (reapply is { } r) c.ZoomFactor = r;
        if (save is { } z) SaveZoom(z);
    }

    void SaveZoom(double z)
    {
        if (Zoom.Same(z, _settings.Zoom)) return;
        _settings.Zoom = z;
        _settings.Save(_paths.SettingsPath);
        _log.Write("zoom.applied", LaunchController.Invariant(z));
    }

    /// <summary>A zoom command: set it, and save it whether or not the
    /// web view reports the change.</summary>
    void SetZoom(double z)
    {
        if (_web.Controller is not { } c) return;
        c.ZoomFactor = z;
        SaveZoom(Zoom.Clamp(c.ZoomFactor));
    }

    /// <summary>Runs a menu command. One that fails says so, naming the
    /// command, in our words; its reason goes to shell.log. The app goes on.</summary>
    void Execute(string id)
    {
        _log.Write("command.run", id);
        Exception? failure;
        try
        {
            failure = Run(id);
        }
        catch (Exception ex)
        {
            failure = ex;
        }
        if (failure == null) return;
        _log.Write("command.failed", id + " " + CommandFailure.Logged(failure));
        var command = _messages.Text(id).Replace("&", "", StringComparison.Ordinal);
        ShellDialogs.Info(this, _messages.Format("error.command_failed.title", ("command", command)),
            _messages.Text("error.command_failed.body"));
    }

    /// <summary>The command itself: null when it worked, else why it did not.</summary>
    Exception? Run(string id)
    {
        var zoom = _web.Controller?.ZoomFactor ?? _settings.Zoom;
        switch (id)
        {
            case MenuIds.OpenProjects:
                Directory.CreateDirectory(_paths.ProjectsDir);
                return WinShell.Open(_paths.ProjectsDir);
            case MenuIds.Quit:
                Close();
                break;
            case MenuIds.Bigger:
                SetZoom(Zoom.Bigger(zoom));
                break;
            case MenuIds.Smaller:
                SetZoom(Zoom.Smaller(zoom));
                break;
            case MenuIds.Actual:
                SetZoom(Zoom.Default);
                break;
            case MenuIds.StartHere:
                if (File.Exists(_layout.StartHere)) return WinShell.Open(_layout.StartHere);
                ShellDialogs.Info(this, _messages.Text("error.start_here_missing.title"), _messages.Text("error.start_here_missing.body"));
                break;
            case MenuIds.GetHelp:
                ShellDialogs.Info(this, _messages.Text("help.get_help.title"), _messages.Text("help.get_help.body"));
                break;
            case MenuIds.ShowLog:
                if (File.Exists(_paths.LauncherLog)) WinShell.RevealInExplorer(_paths.LauncherLog);
                else ShellDialogs.Info(this, _messages.Text("help.no_log.title"), _messages.Text("help.no_log.body"));
                break;
            case MenuIds.About:
                ShellDialogs.Info(this, _messages.Text(MenuIds.About),
                    _messages.Format("about.body", ("version", AppIdentity.Version)));
                break;
        }
        return null;
    }

    void OnProcessFailed(object? sender, CoreWebView2ProcessFailedEventArgs e)
    {
        switch (e.ProcessFailedKind)
        {
            case CoreWebView2ProcessFailedKind.RenderProcessExited:
                _log.Write("webview.renderer_exited", e.Reason.ToString());
                // The app reloads; our own page is shown again with what it
                // had been told (a reload would lose an error it showed).
                if (_page.ShowingApp)
                {
                    try { _web.Core?.Reload(); } catch (Exception) { }
                }
                else _page.ShowAgain();
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

    /// <summary>Logoff and shutdown: quit at once, and show no error page.
    /// A logoff that is cancelled (WM_ENDSESSION with wParam 0) changes
    /// nothing; it never undoes a close the user began.</summary>
    protected override void WndProc(ref Message m)
    {
        if (m.Msg == WM_QUERYENDSESSION)
        {
            _log.Write("session.ending");
            _logoffQueried = !_closing;
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
            else if (_logoffQueried)
            {
                _logoffQueried = false;
                _log.Write("session.end_cancelled");
                _launch.CancelShutdown();
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

    /// <summary>Saves the window's place, then lets go of the single-instance
    /// claim: a window opened while this one shuts down reads these settings.</summary>
    protected override void OnFormClosing(FormClosingEventArgs e)
    {
        base.OnFormClosing(e);
        if (e.Cancel) return;
        _closing = true;
        _logoffQueried = false;
        var screen = Screen.FromControl(this).DeviceName;
        var bounds = WindowState == FormWindowState.Normal ? Bounds : RestoreBounds;
        _settings.Window = WindowPlacement.ToSaved(screen,
            new PxRect(bounds.X, bounds.Y, bounds.Width, bounds.Height), WindowState == FormWindowState.Maximized);
        _log.Write(_settings.Save(_paths.SettingsPath) ? "settings.saved" : "settings.save_failed");
        if (Instance is { } instance)
        {
            instance.Release();
            _log.Write("instance.released");
        }
        Hide();
        _launch.Shutdown(fast: _sessionEnding || e.CloseReason == CloseReason.WindowsShutDown);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) _launch.Dispose();
        base.Dispose(disposing);
    }
}
