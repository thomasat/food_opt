using System.Runtime.Versioning;
using FoodOptimizer.Core.Windows;

namespace FoodOptimizer.Tests;

[SupportedOSPlatform("windows")]
public class StaThreadTests
{
    [Fact]
    public async Task A_pool_thread_runs_the_action_on_an_sta_thread_and_waits()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        ApartmentState? seen = null;
        await Task.Run(() => StaThread.Run(() => seen = Thread.CurrentThread.GetApartmentState()));
        Assert.Equal(ApartmentState.STA, seen);
    }

    [Fact]
    public void An_sta_thread_runs_the_action_inline()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        int? ran = null, caller = null;
        var t = new Thread(() =>
        {
            caller = Environment.CurrentManagedThreadId;
            StaThread.Run(() => ran = Environment.CurrentManagedThreadId);
        });
        t.SetApartmentState(ApartmentState.STA);
        t.Start();
        t.Join();
        Assert.NotNull(ran);
        Assert.Equal(caller, ran);
    }

    [Fact]
    public async Task An_exception_in_the_action_reaches_the_caller()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            Task.Run(() => StaThread.Run(() => throw new InvalidOperationException("x"))));
    }
}
