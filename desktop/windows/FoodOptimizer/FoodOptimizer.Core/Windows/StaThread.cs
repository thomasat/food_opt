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
}
