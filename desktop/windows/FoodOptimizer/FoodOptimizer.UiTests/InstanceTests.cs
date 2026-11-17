using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Windows.Automation;
using FoodOptimizer.Core;

namespace FoodOptimizer.UiTests;

public class InstanceTests
{
    [DllImport("user32.dll")] static extern IntPtr GetForegroundWindow();

    static int Shells(string stage) =>
        Process.GetProcessesByName("FoodOptimizer").Count(p =>
        {
            try { return p.MainModule?.FileName?.StartsWith(stage, StringComparison.OrdinalIgnoreCase) == true; }
            catch (Exception) { return false; }
        });

    [Fact]
    public async Task A_second_launch_brings_the_first_back()
    {
        var run = ShellRun.Fresh();
        using var first = ShellApp.Launch(run);
        var pattern = (WindowPattern)first.Window.GetCurrentPattern(WindowPattern.Pattern);
        // A small screen opens the window maximized; this test is about a
        // normal one (a maximized one is the next test).
        pattern.SetWindowVisualState(WindowVisualState.Normal);
        Assert.True(Uia.WaitUntil(() => pattern.Current.WindowVisualState == WindowVisualState.Normal, TimeSpan.FromSeconds(5)));
        pattern.SetWindowVisualState(WindowVisualState.Minimized);
        Thread.Sleep(800);
        using var second = ShellApp.StartExe(run);
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        await second.WaitForExitAsync(cts.Token);
        Assert.Equal(0, second.ExitCode);
        Assert.True(Uia.WaitUntil(() => pattern.Current.WindowVisualState == WindowVisualState.Normal
                                        && GetForegroundWindow() == first.Hwnd, TimeSpan.FromSeconds(5)),
            $"state {pattern.Current.WindowVisualState}, foreground {GetForegroundWindow()} vs {first.Hwnd}");
        Assert.Equal(1, Shells(run.Stage));
    }

    [Fact]
    public async Task A_second_launch_restores_a_maximized_window_maximized()
    {
        var run = ShellRun.Fresh();
        using var first = ShellApp.Launch(run);
        var pattern = (WindowPattern)first.Window.GetCurrentPattern(WindowPattern.Pattern);
        pattern.SetWindowVisualState(WindowVisualState.Maximized);
        Assert.True(Uia.WaitUntil(() => pattern.Current.WindowVisualState == WindowVisualState.Maximized, TimeSpan.FromSeconds(5)));
        pattern.SetWindowVisualState(WindowVisualState.Minimized);
        Assert.True(Uia.WaitUntil(() => pattern.Current.WindowVisualState == WindowVisualState.Minimized, TimeSpan.FromSeconds(5)));
        using var second = ShellApp.StartExe(run);
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        await second.WaitForExitAsync(cts.Token);
        Assert.Equal(0, second.ExitCode);
        Assert.True(Uia.WaitUntil(() => pattern.Current.WindowVisualState == WindowVisualState.Maximized, TimeSpan.FromSeconds(10)),
            $"state {pattern.Current.WindowVisualState}");
    }

    [Fact]
    public async Task Reopening_while_closing_opens_a_new_window()
    {
        var run = ShellRun.Fresh();
        using (var first = ShellApp.Launch(run))
        {
            first.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
            first.Close();
            using var next = ShellApp.Launch(ShellRun.Same());
            next.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
            Assert.True(await first.WaitForExitAsync(TimeSpan.FromSeconds(10)));
            Assert.Equal(1, Shells(run.Stage));
        }
    }

    /// Before the server is up the old launcher still holds the launch lock;
    /// the new window must wait for it, not take its exit 0 as "another copy".
    [Fact]
    public async Task Reopening_during_loading_opens_a_new_window()
    {
        var run = ShellRun.Fresh();
        using (var first = ShellApp.Launch(run))
        {
            // The launcher holds the lock once it is loading the components.
            Assert.True(Uia.WaitUntil(() => first.LauncherLog().Contains("Loading the app's components"), TimeSpan.FromSeconds(30)));
            first.Close();
            using var next = ShellApp.Launch(ShellRun.Same());
            next.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
            Assert.True(await first.WaitForExitAsync(TimeSpan.FromSeconds(10)));
            Assert.Equal(1, Shells(run.Stage));
        }
    }

    /// A window opened while this one closes reads the settings this one
    /// saved: the claim is let go only after they are written.
    [Fact]
    public async Task Settings_are_saved_before_a_new_window_can_open()
    {
        using var app = ShellApp.Launch(ShellRun.Fresh());
        app.Close();
        Assert.True(await app.WaitForExitAsync(TimeSpan.FromSeconds(10)), "the shell is still running");
        var log = app.Log().ToList();
        var saved = log.FindIndex(l => l.Contains(" settings.saved"));
        var released = log.FindIndex(l => l.Contains(" instance.released"));
        Assert.True(saved >= 0, string.Join("\n", log));
        Assert.True(released > saved, string.Join("\n", log));
    }

    [Fact]
    public async Task The_installer_mutex_is_held_while_running()
    {
        var app = ShellApp.Launch(ShellRun.Fresh());
        try
        {
            Assert.True(Uia.WaitUntil(() => Mutex.TryOpenExisting(AppIdentity.RunningPerUser, out var m) && Dispose(m), TimeSpan.FromSeconds(5)));
            app.Close();
            Assert.True(await app.WaitForExitAsync(TimeSpan.FromSeconds(10)));
            Assert.True(Uia.WaitUntil(() => !Mutex.TryOpenExisting(AppIdentity.RunningPerUser, out _), TimeSpan.FromSeconds(5)));
        }
        finally
        {
            app.Dispose();
        }
    }

    static bool Dispose(Mutex m)
    {
        m.Dispose();
        return true;
    }
}
