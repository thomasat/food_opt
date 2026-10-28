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

    /// The Quick access pin (Shell COM) runs here, so the window's own
    /// thread never waits on Explorer (gate: a Close sent while the pin ran
    /// was never handled).
    [Fact]
    public async Task RunAsync_runs_on_a_background_sta_thread_of_its_own()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        var caller = Environment.CurrentManagedThreadId;
        var (id, apartment, background) = await StaThread.RunAsync(() =>
            (Environment.CurrentManagedThreadId, Thread.CurrentThread.GetApartmentState(), Thread.CurrentThread.IsBackground));
        Assert.NotEqual(caller, id);
        Assert.Equal(ApartmentState.STA, apartment);
        Assert.True(background);
    }

    [Fact]
    public async Task RunAsync_never_blocks_the_caller()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        using var go = new ManualResetEventSlim();
        var task = StaThread.RunAsync(() => go.Wait(TimeSpan.FromSeconds(30)));
        Assert.False(task.IsCompleted);
        go.Set();
        Assert.True(await task);
    }

    [Fact]
    public async Task RunAsync_carries_an_exception_to_the_task()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            StaThread.RunAsync<bool>(() => throw new InvalidOperationException("x")));
    }
}
