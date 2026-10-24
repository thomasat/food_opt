using System.Runtime.Versioning;
using FoodOptimizer.Core.Windows;

namespace FoodOptimizer.Tests;

public class SingleInstanceTests
{
    [Fact]
    [SupportedOSPlatform("windows")]
    public void Second_instance_signals_the_first()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        var id = Guid.NewGuid().ToString("N");
        var mutex = @"Local\FoodOptimizer.Test." + id;
        var evt = @"Local\FoodOptimizer.TestActivate." + id;
        var activated = new ManualResetEventSlim();

        var first = SingleInstance.TryAcquire(mutex, evt, () => activated.Set());
        Assert.NotNull(first);
        Assert.Null(SingleInstance.TryAcquire(mutex, evt, () => { }));
        Assert.True(SingleInstance.SignalFirst(evt));
        Assert.True(activated.Wait(TimeSpan.FromSeconds(2)));

        first.Release();
        first.Release();
        Assert.False(SingleInstance.SignalFirst(evt));
        var again = SingleInstance.TryAcquire(mutex, evt, () => { });
        Assert.NotNull(again);
        again.Dispose();
    }

    [Fact]
    [SupportedOSPlatform("windows")]
    public void The_installer_mutex_is_held_until_disposed()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        var name = @"Local\FoodOptimizer.TestRunning." + Guid.NewGuid().ToString("N");
        var log = new FoodOptimizer.Core.ShellLog(Path.Combine(Path.GetTempPath(), "foodopt-mutex-test.log"));
        var held = InstallerMutex.Hold(name, log);
        Assert.NotNull(held);
        Assert.True(Mutex.TryOpenExisting(name, out var other));
        other.Dispose();
        held.Dispose();
        Assert.False(Mutex.TryOpenExisting(name, out _));
    }
}
