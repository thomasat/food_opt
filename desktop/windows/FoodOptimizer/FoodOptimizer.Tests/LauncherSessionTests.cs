using System.Diagnostics;
using System.Runtime.Versioning;
using FoodOptimizer.Core;
using FoodOptimizer.Core.Windows;

namespace FoodOptimizer.Tests;

[SupportedOSPlatform("windows")]
public sealed class LauncherSessionTests : IDisposable
{
    readonly string _dir = Directory.CreateTempSubdirectory("foodopt-session-").FullName;
    ShellLog Log => new(Path.Combine(_dir, StateFiles.ShellLog));

    public void Dispose()
    {
        try { Directory.Delete(_dir, true); } catch (IOException) { }
    }

    static string System32(string exe) => Path.Combine(Environment.SystemDirectory, exe);

    [Fact]
    public async Task Lifeline_holds_the_launcher_until_stdin_closes()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        using var s = LauncherSession.Start(System32("findstr.exe"), ["x"], _dir, Log);
        var exited = new TaskCompletionSource<int>();
        s.Exited += (_, code) => exited.TrySetResult(code);
        Assert.True(s.ProcessId > 0);
        Assert.True(DateTime.UtcNow - s.StartedUtc < TimeSpan.FromSeconds(10));
        Assert.False(await s.WaitForExitAsync(TimeSpan.FromMilliseconds(500)));
        s.CloseLifeline();
        s.CloseLifeline();
        Assert.True(await s.WaitForExitAsync(TimeSpan.FromSeconds(3)));
        Assert.True(await Task.WhenAny(exited.Task, Task.Delay(3000)) == exited.Task);
        var log = File.ReadAllText(Path.Combine(_dir, StateFiles.ShellLog));
        Assert.Contains("launcher.start " + s.ProcessId, log);
        Assert.Contains("launcher.exit ", log);
    }

    [Fact]
    public async Task KillAll_stops_a_process_that_ignores_stdin()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        using var s = LauncherSession.Start(System32("PING.EXE"), ["-n", "60", "127.0.0.1"], _dir, Log);
        s.CloseLifeline();
        Assert.False(await s.WaitForExitAsync(TimeSpan.FromMilliseconds(500)));
        s.KillAll();
        Assert.True(await s.WaitForExitAsync(TimeSpan.FromSeconds(3)));
        Assert.False(JobObjectTests.Alive(s.ProcessId));
    }

    [Fact]
    public void A_missing_python_is_a_start_exception()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        var e = Assert.Throws<LauncherStartException>(() =>
            LauncherSession.Start(Path.Combine(_dir, "python", "python.exe"), ["-I"], _dir, Log));
        Assert.False(string.IsNullOrEmpty(e.Message));
    }
}
