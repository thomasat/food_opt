using FoodOptimizer.Core;
using FoodOptimizer.Core.Windows;

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
            _page.Attach(_web.Core!, _launch.AppUri);
        }
        catch (Exception ex)
        {
            _log.Write("webview.failed", ex.GetType().Name);
        }
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
        _launch.Shutdown(fast: e.CloseReason == CloseReason.WindowsShutDown);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) _launch.Dispose();
        base.Dispose(disposing);
    }
}
