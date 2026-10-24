using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

public class LaunchMachineTests
{
    static readonly DateTime T0 = new(2026, 10, 24, 12, 0, 0, DateTimeKind.Utc);
    static readonly PageView View = new(false);
    static readonly Uri Health = new("http://127.0.0.1:18501/abc/_stcore/health");
    static readonly Uri App = new("http://127.0.0.1:18501/abc/");

    static FileRead Fresh(string text) => new(text, T0);
    static FileRead Stale(string text) => new(text, T0.AddSeconds(-5));

    static readonly PollSnapshot Nothing = new(null, null, null);
    static readonly PollSnapshot Server = new(null, Fresh("18501 4242\n"), Fresh("abc\n"));

    static LaunchMachine New() => new(T0, View);

    static void TickTo(LaunchMachine m, int tick, PollSnapshot s)
    {
        while (m.Ticks < tick) m.OnTick(s);
    }

    [Fact]
    public void Begin_shows_the_steps()
    {
        Assert.Equal([new PageShowSteps(View)], New().Begin());
        Assert.Equal(LaunchPhase.Waiting, New().Phase);
    }

    [Fact]
    public void A_fresh_status_line_goes_to_the_page_once()
    {
        var m = New();
        var s = new PollSnapshot(Fresh("Loading the app's components|\n"), null, null);
        Assert.Equal([new PageStatusLine("Loading the app's components|")], m.OnTick(s));
        Assert.Empty(m.OnTick(s));
        Assert.Equal([new PageStatusLine("Starting the app|")],
            m.OnTick(new PollSnapshot(Fresh("Starting the app|"), null, null)));
    }

    [Fact]
    public void A_status_line_from_before_this_launch_is_ignored()
    {
        var m = New();
        Assert.Empty(m.OnTick(new PollSnapshot(Stale("Starting the app|\n"), null, null)));
        Assert.Empty(m.OnTick(new PollSnapshot(Fresh("  \n"), null, null)));
    }

    [Fact]
    public void A_port_without_its_base_does_not_arm()
    {
        var m = New();
        Assert.Empty(m.OnTick(new PollSnapshot(null, Fresh("18501 4242\n"), null)));
        Assert.Empty(m.OnTick(new PollSnapshot(null, Fresh("18501 4242\n"), Fresh("a/b"))));
        Assert.Null(m.Port);
    }

    [Fact]
    public void Port_and_base_arm_the_open_step_and_check_health()
    {
        var m = New();
        Assert.Equal([new PageServerStarted(), new CheckHealth(Health, 18501, "abc")], m.OnTick(Server));
        Assert.Equal(18501, m.Port);
        Assert.Equal("abc", m.Base);
    }

    [Fact]
    public void A_stale_port_file_is_not_this_launch()
    {
        var m = New();
        Assert.Empty(m.OnTick(new PollSnapshot(null, Stale("18501 4242\n"), Fresh("abc"))));
        Assert.Empty(m.OnTick(new PollSnapshot(null, Fresh("18501 4242\n"), Stale("abc"))));
        Assert.Null(m.Port);
    }

    [Fact]
    public void Only_one_health_check_at_a_time()
    {
        var m = New();
        m.OnTick(Server);
        Assert.Empty(m.OnTick(Server));
        Assert.Empty(m.OnHealth(false, 18501, "abc"));
        Assert.Equal([new CheckHealth(Health, 18501, "abc")], m.OnTick(Server));
    }

    [Fact]
    public void Status_lines_stop_once_armed()
    {
        var m = New();
        m.OnTick(Server);
        Assert.Empty(m.OnTick(new PollSnapshot(Fresh("Starting the app|"), Server.Port, Server.Base)));
    }

    [Fact]
    public void Healthy_loads_the_app()
    {
        var m = New();
        m.OnTick(Server);
        Assert.Equal(
            [new StopPolling(), new Navigate(App), new StartHealthWatch(Health), new RecordSuccessfulOpen()],
            m.OnHealth(true, 18501, "abc"));
        Assert.Equal(LaunchPhase.Loaded, m.Phase);
        Assert.Empty(m.OnHealth(true, 18501, "abc"));
    }

    [Fact]
    public void A_health_answer_for_another_port_is_ignored()
    {
        var m = New();
        m.OnTick(Server);
        Assert.Empty(m.OnHealth(true, 18502, "abc"));
        Assert.Empty(m.OnHealth(true, 18501, "xyz"));
        Assert.Equal(LaunchPhase.Waiting, m.Phase);
    }

    [Fact]
    public void Give_up_page_after_180_s_armed_and_a_late_success_still_loads()
    {
        var m = New();
        var errors = 0;
        for (var t = 1; t <= 360; t++)
        {
            var a = m.OnTick(Server);
            if (t > 1) m.OnHealth(false, 18501, "abc");
            errors += a.Count(x => x is PageShowError);
        }
        Assert.Equal(0, errors);
        var at361 = m.OnTick(Server);
        Assert.Single(at361.OfType<PageShowError>());
        Assert.Contains(new PageShowError(PageError.CouldNotStart, null), at361);
        m.OnHealth(false, 18501, "abc");
        Assert.Empty(m.OnTick(Server).OfType<PageShowError>());
        Assert.Contains(new Navigate(App), m.OnHealth(true, 18501, "abc"));
    }

    [Fact]
    public void Losing_the_port_file_restarts_the_give_up_clock()
    {
        var m = New();
        TickTo(m, 200, Server);
        m.OnHealth(false, 18501, "abc");
        Assert.Empty(m.OnTick(Nothing));   // tick 201: disarmed
        Assert.Null(m.Port);
        Assert.Contains(new PageServerStarted(), m.OnTick(Server));   // re-armed at 202
        var errors = 0;
        while (m.Ticks < 561)
        {
            errors += m.OnTick(Server).Count(x => x is PageShowError);
            m.OnHealth(false, 18501, "abc");
        }
        Assert.Equal(0, errors);
        Assert.Contains(new PageShowError(PageError.CouldNotStart, null), m.OnTick(Server));
    }

    [Fact]
    public void Stalled_page_at_six_minutes_only_when_not_armed()
    {
        var m = New();
        TickTo(m, 719, Nothing);
        Assert.Equal([new PageShowStalled()], m.OnTick(Nothing));

        var armed = New();
        TickTo(armed, 719, Server);
        Assert.DoesNotContain(new PageShowStalled(), armed.OnTick(Server));
    }

    [Fact]
    public void After_the_stalled_page_server_started_is_not_sent_but_status_lines_are()
    {
        var m = New();
        TickTo(m, 720, Nothing);
        Assert.Equal([new PageStatusLine("Loading the app's components|")],
            m.OnTick(new PollSnapshot(Fresh("Loading the app's components|"), null, null)));
        Assert.Equal([new CheckHealth(Health, 18501, "abc")], m.OnTick(Server));
    }

    [Fact]
    public void Exit_0_waits_fifteen_minutes_then_names_the_other_copy()
    {
        var m = New();
        TickTo(m, 10, Nothing);
        Assert.Empty(m.OnLauncherExited(0));
        var errors = 0;
        while (m.Ticks < 1809) errors += m.OnTick(Nothing).Count(x => x is PageShowError);
        Assert.Equal(0, errors);
        Assert.Equal(
            [new PageShowError(PageError.CouldNotStart, "error.could_not_start.another_copy"), new StopPolling()],
            m.OnTick(Nothing));
        Assert.Equal(LaunchPhase.Failed, m.Phase);
    }

    /// Exit 0 means another launcher holds the lock. With one window per
    /// user that is a launch whose window has just closed, so the launcher
    /// is started again (every 5 s, until the 15-minute deadline) and takes
    /// over once the old one has gone.
    [Fact]
    public void Exit_0_before_load_starts_the_launcher_again_after_five_seconds()
    {
        var m = New();
        TickTo(m, 10, Nothing);
        Assert.Empty(m.OnLauncherExited(0));
        for (var t = 11; t < 10 + LaunchMachine.RelaunchTicks; t++)
            Assert.DoesNotContain(new StartLauncher(), m.OnTick(Nothing));
        Assert.Equal([new StartLauncher()], m.OnTick(Nothing));
        Assert.DoesNotContain(new StartLauncher(), m.OnTick(Nothing));
        Assert.Equal(10, LaunchMachine.RelaunchTicks);

        TickTo(m, 30, Nothing);
        m.OnLauncherExited(0);
        TickTo(m, 39, Nothing);
        Assert.Contains(new StartLauncher(), m.OnTick(Nothing));
    }

    [Fact]
    public void Exit_0_keeps_the_first_deadline()
    {
        var m = New();
        TickTo(m, 10, Nothing);
        m.OnLauncherExited(0);
        TickTo(m, 1000, Nothing);
        m.OnLauncherExited(0);
        TickTo(m, 1809, Nothing);
        Assert.Equal(
            [new PageShowError(PageError.CouldNotStart, "error.could_not_start.another_copy"), new StopPolling()],
            m.OnTick(Nothing));
    }

    [Fact]
    public void No_relaunch_while_a_server_is_armed()
    {
        var m = New();
        m.OnLauncherExited(0);
        for (var t = 0; t < 2 * LaunchMachine.RelaunchTicks; t++)
        {
            Assert.DoesNotContain(new StartLauncher(), m.OnTick(Server));
            m.OnHealth(false, 18501, "abc");
        }
    }

    [Fact]
    public void Exit_0_then_a_healthy_server_loads()
    {
        var m = New();
        m.OnLauncherExited(0);
        m.OnTick(Server);
        Assert.Contains(new Navigate(App), m.OnHealth(true, 18501, "abc"));
    }

    [Theory]
    [InlineData(1, PageError.CouldNotStart)]
    [InlineData(2, PageError.Unsupported)]
    [InlineData(3, PageError.CouldNotStart)]
    [InlineData(4, PageError.CouldNotStart)]
    [InlineData(143, PageError.CouldNotStart)]
    [InlineData(-1, PageError.CouldNotStart)]
    public void Exit_codes_map_to_pages(int code, PageError kind)
    {
        var m = New();
        m.OnTick(Nothing);
        Assert.Equal([new PageShowError(kind, null), new StopPolling()], m.OnLauncherExited(code));
        Assert.Equal(LaunchPhase.Failed, m.Phase);
    }

    [Fact]
    public void Exit_after_load_is_ignored()
    {
        var m = New();
        m.OnTick(Server);
        m.OnHealth(true, 18501, "abc");
        Assert.Empty(m.OnLauncherExited(1));
        Assert.Equal(LaunchPhase.Loaded, m.Phase);
    }

    [Fact]
    public void Start_failure_is_the_missing_part_page()
    {
        var m = New();
        Assert.Equal([new PageShowError(PageError.MissingPart, null), new StopPolling()], m.OnLauncherStartFailed());
        Assert.Equal(LaunchPhase.Failed, m.Phase);
    }

    static LaunchMachine Loaded()
    {
        var m = New();
        m.OnTick(Server);
        m.OnHealth(true, 18501, "abc");
        return m;
    }

    [Fact]
    public void Three_failed_watches_show_the_stopped_page()
    {
        var m = Loaded();
        Assert.Empty(m.OnWatch(false));
        Assert.Empty(m.OnWatch(false));
        Assert.Empty(m.OnWatch(true));
        Assert.Empty(m.OnWatch(false));
        Assert.Empty(m.OnWatch(false));
        Assert.Equal([new StopHealthWatch(), new PageShowError(PageError.Stopped, null)], m.OnWatch(false));
        Assert.Empty(m.OnWatch(false));
        Assert.Equal(LaunchPhase.Failed, m.Phase);
    }

    [Fact]
    public void Resume_resets_the_strikes()
    {
        var m = Loaded();
        m.OnWatch(false);
        m.OnWatch(false);
        m.OnResume();
        Assert.Empty(m.OnWatch(false));
        Assert.Empty(m.OnWatch(false));
    }

    [Fact]
    public void A_load_failure_after_load_shows_stopped()
    {
        Assert.Empty(New().OnLoadFailed());
        var m = Loaded();
        Assert.Equal([new StopHealthWatch(), new PageShowError(PageError.Stopped, null)], m.OnLoadFailed());
        Assert.Equal(LaunchPhase.Failed, m.Phase);
    }

    [Fact]
    public void Shutdown_suppresses_every_page()
    {
        var m = New();
        m.OnTick(Server);
        m.BeginShutdown();
        Assert.Equal(LaunchPhase.ShuttingDown, m.Phase);
        Assert.Empty(m.OnTick(Server));
        Assert.Empty(m.OnHealth(true, 18501, "abc"));
        Assert.Empty(m.OnLauncherExited(1));
        Assert.Empty(m.OnLauncherStartFailed());
        Assert.Empty(m.OnWatch(false));
        Assert.Empty(m.OnLoadFailed());
        var loaded = Loaded();
        loaded.BeginShutdown();
        loaded.OnWatch(false);
        loaded.OnWatch(false);
        Assert.Empty(loaded.OnWatch(false));
        Assert.Empty(loaded.OnLoadFailed());
    }

    [Fact]
    public void A_failed_machine_ignores_ticks()
    {
        var m = New();
        m.OnLauncherExited(1);
        Assert.Empty(m.OnTick(Server));
        Assert.Empty(m.OnTick(new PollSnapshot(Fresh("Starting the app|"), null, null)));
    }
}
