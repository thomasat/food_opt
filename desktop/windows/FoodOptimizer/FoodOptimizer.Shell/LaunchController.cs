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
    bool _retrying;
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
        Microsoft.Win32.SystemEvents.PowerModeChanged += OnPowerModeChanged;
    }

    void OnPowerModeChanged(object? sender, Microsoft.Win32.PowerModeChangedEventArgs e)
    {
        if (e.Mode == Microsoft.Win32.PowerModes.Resume) OnUi(() => _machine?.OnResume());
    }

    public int? Port => _machine?.Port;
    public bool Loaded => _machine?.Phase == LaunchPhase.Loaded;

    /// <summary>This launch's base path, once its server is found; shell.log
    /// never holds it (M3).</summary>
    public string? Base => _machine?.Base;

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
        if (!StartSession(machine)) return;
        _poll.Start();
    }

    /// <summary>Starts a launcher for <paramref name="machine"/>; false when it could not start.</summary>
    bool StartSession(LaunchMachine machine)
    {
        try
        {
            // The exit is handled on the UI thread, after this method has
            // set _session (an exit before Start returns is still reported).
            LauncherSession? session = null;
            session = LauncherSession.Start(_layout.Python, LauncherCommand.Arguments(_layout, _paths), _layout.Root, _log,
                code => OnUi(() =>
                {
                    if (_session == session && _machine == machine) Execute(machine.OnLauncherExited(code));
                }));
            var old = _session;
            _session = session;
            old?.Dispose();
            return true;
        }
        catch (LauncherStartException)
        {
            Execute(machine.OnLauncherStartFailed());
            return false;
        }
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
                    _page.NavigateToApp(n.Uri, Base);
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
                case StartLauncher:
                    if (_machine is { } m) StartSession(m);
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

    /// <summary>Try again: ends the old launch (its whole job, so its server
    /// lets go of the port and the log), clears its status line and starts
    /// a new one. A second click while this runs does nothing.</summary>
    public async Task RetryAsync()
    {
        if (_retrying || _disposed) return;
        _retrying = true;
        try
        {
            _log.Write("launch.retry");
            _poll.Stop();
            _watch.Stop();
            _watchUri = null;
            _machine = null;
            var old = _session;
            _session = null;
            if (old != null)
            {
                old.KillAll();
                await old.WaitForExitAsync(TimeSpan.FromSeconds(5));
                old.Dispose();
            }
            foreach (var name in new[] { StateFiles.Status, StateFiles.StatusTmp })
            {
                try { File.Delete(Path.Combine(_paths.StateDir, name)); }
                catch (Exception) { }
            }
            if (!_disposed) Start();
        }
        finally
        {
            _retrying = false;
        }
    }

    /// <summary>The app's page failed to load after it had loaded once.</summary>
    public void OnLoadFailed()
    {
        if (_machine is { } m) Execute(m.OnLoadFailed());
    }

    /// <summary>Logoff is coming: from now on no page may show.</summary>
    public void BeginShutdown() => _machine?.BeginShutdown();

    /// <summary>The logoff was cancelled: the launch goes on as before, and a
    /// launcher exit that came meanwhile is acted on now.</summary>
    public void CancelShutdown()
    {
        if (_machine is { } m) Execute(m.CancelShutdown());
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
        Microsoft.Win32.SystemEvents.PowerModeChanged -= OnPowerModeChanged;
        _poll.Dispose();
        _watch.Dispose();
        _session?.Dispose();
        _probe.Dispose();
    }

    internal static string Invariant(double d) => d.ToString("0.##", CultureInfo.InvariantCulture);
}
