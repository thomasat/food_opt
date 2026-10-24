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

    [Fact]
    public async Task The_installer_mutex_is_held_while_running()
    {
        var app = ShellApp.Launch(ShellRun.Fresh());
        try
        {
            Assert.True(Uia.WaitUntil(() => Mutex.TryOpenExisting(AppIdentity.InstallerMutex, out var m) && Dispose(m), TimeSpan.FromSeconds(5)));
            app.Close();
            Assert.True(await app.WaitForExitAsync(TimeSpan.FromSeconds(10)));
            Assert.True(Uia.WaitUntil(() => !Mutex.TryOpenExisting(AppIdentity.InstallerMutex, out _), TimeSpan.FromSeconds(5)));
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
