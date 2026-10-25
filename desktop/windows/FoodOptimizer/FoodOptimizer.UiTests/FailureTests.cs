using System.Diagnostics;
using System.Management;
using System.Runtime.InteropServices;
using System.Windows.Automation;

namespace FoodOptimizer.UiTests;

public class FailureTests
{
    static readonly TimeSpan Thirty = TimeSpan.FromSeconds(30);

    [DllImport("user32.dll")]
    static extern IntPtr SendMessageW(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam);

    [Fact]
    public void Unsupported_machine_page()
    {
        var run = ShellRun.Fresh() with { Env = new Dictionary<string, string> { ["FOODOPT_TEST_ARCH"] = "x86" } };
        using var app = ShellApp.Launch(run);
        app.WaitFor(ControlType.Text, "This PC is not supported", Thirty);
        app.WaitFor(ControlType.Text, "Needs 64-bit Windows 10 (22H2) or Windows 11.", Thirty);
        Assert.False(IsShown(app, "Try again"));
    }

    /// <summary>The page keeps its hidden Try again button in the tree.</summary>
    static bool IsShown(ShellApp app, string button)
    {
        var b = Uia.Find(app.Window, ControlType.Button, button);
        return b != null && !b.Current.IsOffscreen;
    }

    [Fact]
    public void Could_not_start_then_try_again_recovers()
    {
        var run = ShellRun.Fresh();
        var blocker = Path.Combine(run.StateDir, "server.port");
        Directory.CreateDirectory(blocker);
        using var app = ShellApp.Launch(run);
        app.WaitFor(ControlType.Text, "The app could not start", ShellApp.Load);
        var retry = app.WaitFor(ControlType.Button, "Try again", Thirty);
        Directory.Delete(blocker);
        Uia.Invoke(retry);
        app.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
    }

    [Fact]
    public void Missing_python_is_the_missing_part_page()
    {
        var run = ShellRun.Fresh();
        var python = Path.Combine(run.Stage, "python", "python.exe");
        var moved = python + ".moved";
        File.Move(python, moved);
        try
        {
            using var app = ShellApp.Launch(run);
            app.WaitFor(ControlType.Text, "Part of the app is missing. Reinstall Food Optimizer.", Thirty);
            Assert.False(IsShown(app, "Try again"));
            Assert.Empty(app.StagePythons());
        }
        finally
        {
            File.Move(moved, python);
        }
    }

    static (int Pid, string CommandLine)[] Serves(ShellApp app) =>
        app.StagePythons().Where(p => p.CommandLine.Contains("child.py") && p.CommandLine.Contains(" serve ")).ToArray();

    static (int Pid, string CommandLine)[] Launchers(ShellApp app) =>
        app.StagePythons().Where(p => p.CommandLine.Contains("-m launcher")).ToArray();

    [Fact]
    public void A_dead_server_shows_stopped_and_try_again_leaves_one_server()
    {
        using var app = ShellApp.Launch(ShellRun.Fresh());
        app.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
        var serve = Assert.Single(Serves(app));
        Process.GetProcessById(serve.Pid).Kill();
        app.WaitFor(ControlType.Text, "Food Optimizer stopped unexpectedly", TimeSpan.FromSeconds(45));
        var retry = app.WaitFor(ControlType.Button, "Try again", Thirty);
        Uia.Invoke(retry);
        Thread.Sleep(100);
        try { Uia.Invoke(retry); } catch (Exception) { }
        app.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
        Thread.Sleep(5000);
        Assert.Single(Serves(app));
        Assert.Single(Launchers(app));
    }

    static IReadOnlyList<(int Pid, string CommandLine)> WebViews(ShellApp app)
    {
        var data = Path.Combine(app.StateDir, "WebView2");
        var found = new List<(int, string)>();
        using var searcher = new ManagementObjectSearcher(
            "SELECT ProcessId, CommandLine FROM Win32_Process WHERE Name = 'msedgewebview2.exe'");
        foreach (ManagementObject o in searcher.Get())
        {
            using (o)
            {
                var cmd = o["CommandLine"] as string ?? "";
                if (cmd.Contains(data, StringComparison.OrdinalIgnoreCase))
                    found.Add((Convert.ToInt32(o["ProcessId"]), cmd));
            }
        }
        return found;
    }

    [Fact]
    public void A_renderer_crash_reloads()
    {
        using var app = ShellApp.Launch(ShellRun.Fresh());
        app.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
        foreach (var r in WebViews(app).Where(p => p.CommandLine.Contains("--type=renderer")))
            Process.GetProcessById(r.Pid).Kill();
        Assert.True(Uia.WaitUntil(() => app.Log().Any(l => l.Contains("webview.renderer_exited")), Thirty));
        app.WaitFor(ControlType.Edit, "Type here", Thirty);
    }

    [Fact]
    public void A_browser_process_exit_recreates_the_view()
    {
        using var app = ShellApp.Launch(ShellRun.Fresh());
        app.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
        var browser = Assert.Single(WebViews(app), p => !p.CommandLine.Contains("--type="));
        Process.GetProcessById(browser.Pid).Kill();
        Assert.True(Uia.WaitUntil(() => app.Log().Any(l => l.Contains("webview.recreated 1")), TimeSpan.FromSeconds(60)));
        app.WaitFor(ControlType.Edit, "Type here", TimeSpan.FromSeconds(60));
    }

    /// A launch lock still held by a launcher that is going away (its window
    /// just closed) must not leave the new window waiting 15 minutes. Needs
    /// phase 2's launcher to exit 0 with --stdin-lifeline (fixed in ed7489a;
    /// it used to die in finalisation, "_enter_buffered_busy"). The relaunch
    /// on exit 0 is unit-tested in LaunchMachineTests.
    [Fact]
    public void A_lock_held_by_a_launch_that_is_ending_is_taken_over()
    {
        var run = ShellRun.Fresh();
        using var holder = Process.Start(new ProcessStartInfo("ping.exe", "-n 9 127.0.0.1")
        {
            UseShellExecute = false,
            CreateNoWindow = true,
        })!;
        try
        {
            var lockDir = Path.Combine(run.StateDir, "launch.lock");
            Directory.CreateDirectory(lockDir);
            File.WriteAllText(Path.Combine(lockDir, "pid"), holder.Id.ToString());
            using var app = ShellApp.Launch(run);
            app.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
            Assert.Contains(app.Log(), l => l.EndsWith("launcher.exit 0"));
        }
        finally
        {
            try { holder.Kill(); } catch (Exception) { }
        }
    }

    [Fact]
    public void A_browser_process_exit_on_an_error_page_shows_it_again()
    {
        var run = ShellRun.Fresh() with { Env = new Dictionary<string, string> { ["FOODOPT_TEST_ARCH"] = "x86" } };
        using var app = ShellApp.Launch(run);
        app.WaitFor(ControlType.Text, "This PC is not supported", Thirty);
        var browser = Assert.Single(WebViews(app), p => !p.CommandLine.Contains("--type="));
        Process.GetProcessById(browser.Pid).Kill();
        Assert.True(Uia.WaitUntil(() => app.Log().Any(l => l.Contains("webview.recreated 1")), TimeSpan.FromSeconds(60)));
        app.WaitFor(ControlType.Text, "This PC is not supported", Thirty);
        app.WaitFor(ControlType.Text, "Needs 64-bit Windows 10 (22H2) or Windows 11.", Thirty);
    }

    /// A renderer crash on an error page shows the page again, not a blank
    /// view or the steps it began with.
    [Fact]
    public void A_renderer_crash_on_an_error_page_shows_it_again()
    {
        var run = ShellRun.Fresh() with { Env = new Dictionary<string, string> { ["FOODOPT_TEST_ARCH"] = "x86" } };
        using var app = ShellApp.Launch(run);
        app.WaitFor(ControlType.Text, "This PC is not supported", Thirty);
        var renderers = WebViews(app).Where(p => p.CommandLine.Contains("--type=renderer")).ToList();
        Assert.NotEmpty(renderers);
        foreach (var r in renderers) Process.GetProcessById(r.Pid).Kill();
        Assert.True(Uia.WaitUntil(() => app.Log().Any(l => l.Contains("webview.renderer_exited")), Thirty));
        app.WaitFor(ControlType.Text, "This PC is not supported", Thirty);
        app.WaitFor(ControlType.Text, "Needs 64-bit Windows 10 (22H2) or Windows 11.", Thirty);
    }

    /// A logoff another app vetoed: the window must still show a server
    /// that dies afterwards, as if the logoff never began.
    [Fact]
    public void A_cancelled_logoff_keeps_the_window_working()
    {
        using var app = ShellApp.Launch(ShellRun.Fresh());
        app.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
        var hwnd = app.Hwnd;
        SendMessageW(hwnd, 0x11, IntPtr.Zero, IntPtr.Zero);
        SendMessageW(hwnd, 0x16, IntPtr.Zero, IntPtr.Zero);
        var serve = Assert.Single(Serves(app));
        Process.GetProcessById(serve.Pid).Kill();
        app.WaitFor(ControlType.Text, "Food Optimizer stopped unexpectedly", TimeSpan.FromSeconds(45));
        Assert.Contains(app.Log(), l => l.Contains("session.end_cancelled"));
    }

    [Fact]
    public async Task Session_end_quits_quietly_and_the_next_launch_loads()
    {
        using (var app = ShellApp.Launch(ShellRun.Fresh()))
        {
            app.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
            var hwnd = app.Hwnd;
            // A logoff another app vetoes first, then the real one.
            SendMessageW(hwnd, 0x11, IntPtr.Zero, IntPtr.Zero);
            SendMessageW(hwnd, 0x16, IntPtr.Zero, IntPtr.Zero);
            SendMessageW(hwnd, 0x11, IntPtr.Zero, IntPtr.Zero);
            SendMessageW(hwnd, 0x16, new IntPtr(1), IntPtr.Zero);
            Assert.True(await app.WaitForExitAsync(TimeSpan.FromSeconds(3)), "the shell is still running");
            Assert.True(Uia.WaitUntil(() => app.StagePythons().Count == 0, TimeSpan.FromSeconds(3)));
            var log = app.Log();
            var ending = log.ToList().FindIndex(l => l.Contains("session.ending"));
            Assert.True(ending >= 0, string.Join("\n", log));
            Assert.DoesNotContain(log.Skip(ending), l => l.Contains("page.error"));
        }
        using var next = ShellApp.Launch(ShellRun.Same());
        next.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
    }
}
