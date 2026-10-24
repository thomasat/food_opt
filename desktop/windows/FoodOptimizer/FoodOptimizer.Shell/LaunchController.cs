using System.Globalization;
using FoodOptimizer.Core;
using FoodOptimizer.Core.Windows;

namespace FoodOptimizer.Shell;

/// <summary>Runs one launch at a time: starts the launcher, polls its files,
/// checks health, and turns the launch machine's actions into page calls
/// and navigations. Lives on the UI thread.</summary>
sealed class LaunchController : IDisposable
{
    readonly Control _ui;
    readonly PageHost _page;
    readonly InstallLayout _layout;
    readonly UserPaths _paths;
    readonly ShellSettings _settings;
    readonly ShellLog _log;
    readonly HealthProbe _probe = new();
    readonly System.Windows.Forms.Timer _poll = new() { Interval = 500 };
    readonly System.Windows.Forms.Timer _watch = new() { Interval = 10_000 };
    LaunchMachine? _machine;
    LauncherSession? _session;
    Uri? _watchUri;
    bool _watchInFlight;
    bool _disposed;

    public LaunchController(Control ui, PageHost page, InstallLayout layout, UserPaths paths,
                            ShellSettings settings, ShellLog log)
    {
        _ui = ui;
        _page = page;
        _layout = layout;
        _paths = paths;
        _settings = settings;
        _log = log;
        _poll.Tick += (_, _) => Poll();
        _watch.Tick += async (_, _) => await WatchAsync();
    }

    public int? Port => _machine?.Port;
    public bool Loaded => _machine?.Phase == LaunchPhase.Loaded;

    /// <summary>The app URL once loaded, for a recreated web view.</summary>
    public Uri? AppUri { get; private set; }

    bool FirstOpening => _settings.LastOpenedVersion != AppIdentity.Version;

    public void Start()
    {
        var machine = new LaunchMachine(DateTime.UtcNow, new PageView(FirstOpening));
        _machine = machine;
        AppUri = null;
        Execute(machine.Begin());

        var missing = _layout.MissingForLaunch();
        if (missing.Count > 0)
        {
            _log.Write("launch.missing", string.Join(";", missing));
            Execute(machine.OnLauncherStartFailed());
            return;
        }
        try
        {
            var session = LauncherSession.Start(_layout.Python, LauncherCommand.Arguments(_layout, _paths), _layout.Root, _log);
            _session = session;
            session.Exited += (_, code) => OnUi(() =>
            {
                if (_session == session && _machine == machine) Execute(machine.OnLauncherExited(code));
            });
        }
        catch (LauncherStartException)
        {
            Execute(machine.OnLauncherStartFailed());
            return;
        }
        _poll.Start();
    }

    void OnUi(Action a)
    {
        if (_disposed || _ui.IsDisposed) return;
        try { _ui.BeginInvoke(a); }
        catch (InvalidOperationException) { }
    }

    void Poll()
    {
        if (_machine is not { } m) return;
        var dir = _paths.StateDir;
        var snapshot = new PollSnapshot(
            StateFiles.ReadShared(Path.Combine(dir, StateFiles.Status)),
            StateFiles.ReadShared(Path.Combine(dir, StateFiles.Port)),
            StateFiles.ReadShared(Path.Combine(dir, StateFiles.Base)));
        Execute(m.OnTick(snapshot));
    }

    public void Execute(IReadOnlyList<ShellAction> actions)
    {
        foreach (var a in actions)
        {
            switch (a)
            {
                case PageStatusLine s:
                    _page.Run(PageBridge.StatusLine(s.Raw));
                    break;
                case PageServerStarted:
                    _page.Run(PageBridge.ServerStarted());
                    break;
                case PageShowStalled:
                    _page.Run(PageBridge.ShowStalled());
                    break;
                case PageShowError e:
                    _log.Write("page.error", PageBridge.KindName(e.Kind));
                    _page.Run(PageBridge.ShowError(e.Kind, e.DetailId));
                    break;
                case PageShowSteps v:
                    _ = _page.ShowPageAsync(v.View);
                    break;
                case CheckHealth c:
                    _ = CheckAsync(c);
                    break;
                case Navigate n:
                    AppUri = n.Uri;
                    _page.NavigateToApp(n.Uri);
                    break;
                case StartHealthWatch w:
                    _watchUri = w.Uri;
                    _watch.Start();
                    break;
                case StopHealthWatch:
                    _watch.Stop();
                    break;
                case StopPolling:
                    _poll.Stop();
                    break;
                case RecordSuccessfulOpen:
                    _settings.LastOpenedVersion = AppIdentity.Version;
                    _settings.Save(_paths.SettingsPath);
                    break;
            }
        }
    }

    async Task CheckAsync(CheckHealth c)
    {
        var machine = _machine;
        var ok = await _probe.IsHealthyAsync(c.Uri);
        if (_disposed || machine != _machine || machine == null) return;
        Execute(machine.OnHealth(ok, c.Port, c.Base));
    }

    async Task WatchAsync()
    {
        if (_watchInFlight || _watchUri is not { } uri || _machine is not { } machine) return;
        _watchInFlight = true;
        try
        {
            var ok = await _probe.IsHealthyAsync(uri);
            if (_disposed || machine != _machine) return;
            Execute(machine.OnWatch(ok));
        }
        finally
        {
            _watchInFlight = false;
        }
    }

    /// <summary>Stops the launch: no more pages, the lifeline closed, a
    /// short wait for a clean shutdown (skipped at logoff), then the job.</summary>
    public void Shutdown(bool fast)
    {
        _machine?.BeginShutdown();
        _poll.Stop();
        _watch.Stop();
        var session = _session;
        if (session == null) return;
        session.CloseLifeline();
        if (!fast)
        {
            try { session.WaitForExitAsync(TimeSpan.FromSeconds(3)).GetAwaiter().GetResult(); }
            catch (Exception) { }
        }
        session.KillAll();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _poll.Dispose();
        _watch.Dispose();
        _session?.Dispose();
        _probe.Dispose();
    }

    internal static string Invariant(double d) => d.ToString("0.##", CultureInfo.InvariantCulture);
}
