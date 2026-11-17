namespace FoodOptimizer.Core;

/// <summary>One launch, from the steps page to the loaded app or an error
/// page: the Swift window's poll loop (FoodOptimizerApp.swift poll,
/// readStatusFile, startHealthWatch, launcherEnded) as a pure state machine.
/// Every input is a method; every output is a list of actions. Ticks are the
/// 0.5 s poll ticks.</summary>
public sealed class LaunchMachine
{
    public const int StalledTick = 720, HealthGiveUpTicks = 360, DeferTicks = 1800, WatchStrikeLimit = 3;

    /// <summary>After an exit 0, the launcher is started again this many
    /// ticks later (5 s).</summary>
    public const int RelaunchTicks = 10;

    const string AnotherCopy = "error.could_not_start.another_copy";

    readonly DateTime _launchStartedUtc;
    readonly PageView _view;
    int? _armedAt;
    bool _timedOut, _stalledShown, _checkInFlight;
    string? _lastStatus;
    int? _deferDeadline;
    int? _relaunchAt;
    int _strikes;
    LaunchPhase _phaseBeforeShutdown;
    int? _exitDuringShutdown;
    bool _stalledMissed;

    public LaunchMachine(DateTime launchStartedUtc, PageView view)
    {
        _launchStartedUtc = launchStartedUtc;
        _view = view;
    }

    public LaunchPhase Phase { get; private set; } = LaunchPhase.Waiting;
    public int Ticks { get; private set; }
    public int? Port { get; private set; }
    public string? Base { get; private set; }

    static readonly IReadOnlyList<ShellAction> None = [];

    public IReadOnlyList<ShellAction> Begin() => [new PageShowSteps(_view)];

    public IReadOnlyList<ShellAction> OnTick(PollSnapshot s)
    {
        Ticks++;
        if (Phase == LaunchPhase.ShuttingDown && Ticks == StalledTick) _stalledMissed = true;
        if (Phase != LaunchPhase.Waiting) return None;
        var actions = new List<ShellAction>();

        if (_armedAt == null && s.Status is { } status && Fresh(status))
        {
            var line = status.Text.Trim();
            if (line.Length > 0 && line != _lastStatus)
            {
                _lastStatus = line;
                actions.Add(new PageStatusLine(line));
            }
        }

        if ((Ticks == StalledTick || _stalledMissed) && _armedAt == null && !_timedOut)
        {
            _stalledMissed = false;
            _stalledShown = true;
            actions.Add(new PageShowStalled());
        }

        if (_deferDeadline is { } deadline && Ticks >= deadline)
        {
            Phase = LaunchPhase.Failed;
            actions.Add(new PageShowError(PageError.CouldNotStart, AnotherCopy));
            actions.Add(new StopPolling());
            return actions;
        }

        // Another launcher held the lock. With one window per user that is a
        // launch whose window has closed, so ours tries again until it can
        // take over - unless a server from that launch is up meanwhile.
        if (_relaunchAt is { } at && Ticks >= at && _armedAt == null)
        {
            _relaunchAt = null;
            actions.Add(new StartLauncher());
        }

        var port = s.Port is { } p && Fresh(p) ? StateFiles.ParsePort(p.Text) : null;
        var @base = s.Base is { } b && Fresh(b) ? StateFiles.ParseBase(b.Text) : null;
        if (port == null || @base == null)
        {
            _armedAt = null;
            Port = null;
            Base = null;
            return actions;
        }

        if (_armedAt == null)
        {
            _armedAt = Ticks;
            _timedOut = false;
            if (!_stalledShown) actions.Add(new PageServerStarted());
        }
        else if (!_timedOut && Ticks - _armedAt.Value >= HealthGiveUpTicks)
        {
            _timedOut = true;
            actions.Add(new PageShowError(PageError.CouldNotStart, null));
        }

        Port = port.Port;
        Base = @base;
        if (!_checkInFlight)
        {
            _checkInFlight = true;
            actions.Add(new CheckHealth(Endpoints.Health(port.Port, @base), port.Port, @base));
        }
        return actions;
    }

    bool Fresh(FileRead f) => StateFiles.IsFresh(f.ModifiedUtc, _launchStartedUtc);

    public IReadOnlyList<ShellAction> OnHealth(bool healthy, int port, string @base)
    {
        _checkInFlight = false;
        if (Phase != LaunchPhase.Waiting) return None;
        if (!healthy || port != Port || @base != Base) return None;
        Phase = LaunchPhase.Loaded;
        return
        [
            new StopPolling(),
            new Navigate(Endpoints.App(port, @base)),
            new StartHealthWatch(Endpoints.Health(port, @base)),
            new RecordSuccessfulOpen(),
        ];
    }

    public IReadOnlyList<ShellAction> OnLauncherExited(int code)
    {
        if (Phase == LaunchPhase.ShuttingDown) _exitDuringShutdown = code;
        if (Phase != LaunchPhase.Waiting) return None;
        if (code == 0)
        {
            // Another launch holds the lock: wait for it, within reason, and
            // start ours again shortly in case it is a launch that is ending.
            _deferDeadline ??= Ticks + DeferTicks;
            _relaunchAt = Ticks + RelaunchTicks;
            return None;
        }
        return Fail(code == 2 ? PageError.Unsupported : PageError.CouldNotStart);
    }

    public IReadOnlyList<ShellAction> OnLauncherStartFailed() =>
        Phase == LaunchPhase.Waiting ? Fail(PageError.MissingPart) : None;

    IReadOnlyList<ShellAction> Fail(PageError kind)
    {
        Phase = LaunchPhase.Failed;
        return [new PageShowError(kind, null), new StopPolling()];
    }

    public IReadOnlyList<ShellAction> OnWatch(bool healthy)
    {
        if (Phase != LaunchPhase.Loaded) return None;
        if (healthy)
        {
            _strikes = 0;
            return None;
        }
        return ++_strikes >= WatchStrikeLimit ? Stopped() : None;
    }

    public IReadOnlyList<ShellAction> OnLoadFailed() =>
        Phase == LaunchPhase.Loaded ? Stopped() : None;

    IReadOnlyList<ShellAction> Stopped()
    {
        Phase = LaunchPhase.Failed;
        return [new StopHealthWatch(), new PageShowError(PageError.Stopped, null)];
    }

    /// <summary>Back from sleep: the missed checks say nothing about the server.</summary>
    public void OnResume() => _strikes = 0;

    /// <summary>Quitting or logging off: no page may show from now on.</summary>
    public void BeginShutdown()
    {
        if (Phase == LaunchPhase.ShuttingDown) return;
        _phaseBeforeShutdown = Phase;
        _exitDuringShutdown = null;
        Phase = LaunchPhase.ShuttingDown;
    }

    /// <summary>A logoff that was cancelled: back to the phase it replaced.
    /// A launcher exit that arrived meanwhile is applied now; its actions are
    /// returned.</summary>
    public IReadOnlyList<ShellAction> CancelShutdown()
    {
        if (Phase != LaunchPhase.ShuttingDown) return None;
        Phase = _phaseBeforeShutdown;
        if (_exitDuringShutdown is not { } code) return None;
        _exitDuringShutdown = null;
        return OnLauncherExited(code);
    }
}
