using System.Text.RegularExpressions;
using System.Windows.Automation;
using FoodOptimizer.Core;

namespace FoodOptimizer.UiTests;

public class LaunchTests
{
    static readonly TimeSpan Fifteen = TimeSpan.FromSeconds(15);

    [Fact]
    public void Window_opens_at_once_with_the_steps_page()
    {
        using var app = ShellApp.Launch(ShellRun.Fresh());
        Uia.Record("window_visible_seconds", app.WindowSeconds.ToString("0.00"));
        Assert.True(app.WindowSeconds <= 5, $"window after {app.WindowSeconds:0.0} s");
        app.WaitFor(ControlType.Text, "Loading the app's components", Fifteen);
        app.WaitFor(ControlType.Text, "Opening your projects", Fifteen);
        app.WaitFor(ControlType.Text, "Setting up Food Optimizer", Fifteen);
        app.WaitFor(ControlType.Text, "First opening takes longer. Keep this window open.", Fifteen);
    }

    [Fact]
    public void The_app_loads_from_the_base_path()
    {
        using var app = ShellApp.Launch(ShellRun.Fresh());
        app.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
        var @base = ShellApp.ReadShared(Path.Combine(app.StateDir, "server.base"));
        Assert.Matches(new Regex(@"^[A-Za-z0-9_-]{32}\r?\n$"), @base);
        var port = ShellApp.ReadShared(Path.Combine(app.StateDir, "server.port")).Split(' ')[0].Trim();
        // M3: shell.log never holds the base path, the address's secret.
        var log = string.Join("\n", app.Log());
        Assert.Contains($"app.navigate http://127.0.0.1:{port}/…/", log);
        Assert.DoesNotContain(@base.Trim(), log);
        Assert.Equal(AppIdentity.Version, (string?)app.Settings()["lastOpenedVersion"]);
    }

    [Fact]
    public async Task A_second_opening_says_starting()
    {
        using (var first = ShellApp.Launch(ShellRun.Fresh()))
        {
            first.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
            first.Close();
            Assert.True(await first.WaitForExitAsync(TimeSpan.FromSeconds(10)));
        }
        using var second = ShellApp.Launch(ShellRun.Same());
        second.WaitFor(ControlType.Text, "Starting Food Optimizer", Fifteen);
        Assert.False(second.Exists(ControlType.Text, "Setting up Food Optimizer"));
    }

    [Fact]
    public async Task No_console_window_ever_appears()
    {
        var watch = new ConsoleWatch();
        watch.Start();
        using (var app = ShellApp.Launch(ShellRun.Fresh()))
        {
            app.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
            app.Close();
            Assert.True(await app.WaitForExitAsync(TimeSpan.FromSeconds(10)));
        }
        Assert.Empty(watch.Stop());
    }

    [Fact]
    public async Task Closing_the_window_stops_everything()
    {
        using var app = ShellApp.Launch(ShellRun.Fresh());
        app.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
        app.Close();
        Assert.True(await app.WaitForExitAsync(TimeSpan.FromSeconds(5)), "the shell is still running");
        Assert.True(Uia.WaitUntil(() => app.StagePythons().Count == 0, TimeSpan.FromSeconds(5)),
            string.Join("; ", app.StagePythons().Select(p => p.CommandLine)));
        Assert.False(File.Exists(Path.Combine(app.StateDir, "server.port")));
        Assert.False(File.Exists(Path.Combine(app.StateDir, "server.base")));
        Assert.Contains("the window closed - shut down", app.LauncherLog());
    }
}
