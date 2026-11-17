using System.Runtime.ExceptionServices;
using System.Runtime.Versioning;

namespace FoodOptimizer.Core.Windows;

/// <summary>Runs code that needs a single-threaded apartment (a dialog)
/// from any thread.</summary>
[SupportedOSPlatform("windows")]
public static class StaThread
{
    /// <summary>Runs <paramref name="action"/> here when this thread is STA,
    /// else on a new STA thread, and waits for it. An exception it throws
    /// is thrown here.</summary>
    public static void Run(Action action)
    {
        if (Thread.CurrentThread.GetApartmentState() == ApartmentState.STA)
        {
            action();
            return;
        }
        ExceptionDispatchInfo? failure = null;
        var thread = new Thread(() =>
        {
            try { action(); }
            catch (Exception e) { failure = ExceptionDispatchInfo.Capture(e); }
        });
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        thread.Join();
        failure?.Throw();
    }

    /// <summary>Runs <paramref name="f"/> on a new STA thread of its own,
    /// always (never the caller's), and returns at once: the task ends with
    /// its result or its exception. A background thread, so a window that
    /// closes meanwhile never waits for it; its continuation never runs on
    /// that thread.</summary>
    public static Task<T> RunAsync<T>(Func<T> f)
    {
        var done = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
        var thread = new Thread(() =>
        {
            try { done.SetResult(f()); }
            catch (Exception e) { done.SetException(e); }
        }) { IsBackground = true };
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        return done.Task;
    }
}
