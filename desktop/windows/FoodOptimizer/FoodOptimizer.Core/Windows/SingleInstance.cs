using System.Runtime.Versioning;

namespace FoodOptimizer.Core.Windows;

/// <summary>One Food Optimizer window per user session. The first instance
/// owns a Local mutex and waits on an activation event; a second launch
/// lets it take the foreground, sets the event and exits.</summary>
[SupportedOSPlatform("windows")]
public sealed class SingleInstance : IDisposable
{
    Mutex? _mutex;
    EventWaitHandle? _event;
    RegisteredWaitHandle? _wait;
    readonly object _lock = new();

    SingleInstance(Mutex mutex, EventWaitHandle evt, Action onActivate)
    {
        _mutex = mutex;
        _event = evt;
        _wait = ThreadPool.RegisterWaitForSingleObject(evt, (_, _) => onActivate(), null, Timeout.Infinite, executeOnlyOnce: false);
    }

    /// <summary>This instance's claim, or null when another instance holds it.</summary>
    public static SingleInstance? TryAcquire(string mutexName, string eventName, Action onActivate)
    {
        var mutex = new Mutex(initiallyOwned: true, mutexName, out var createdNew);
        if (!createdNew)
        {
            mutex.Dispose();
            return null;
        }
        var evt = new EventWaitHandle(false, EventResetMode.AutoReset, eventName);
        return new SingleInstance(mutex, evt, onActivate);
    }

    /// <summary>Asks the first instance to come to the front; false when
    /// there is none.</summary>
    public static bool SignalFirst(string eventName)
    {
        if (!EventWaitHandle.TryOpenExisting(eventName, out var evt)) return false;
        using (evt)
        {
            NativeMethods.AllowSetForegroundWindow(NativeMethods.ASFW_ANY);
            return evt.Set();
        }
    }

    /// <summary>Gives the claim up, so a new launch opens its own window.</summary>
    public void Release()
    {
        lock (_lock)
        {
            _wait?.Unregister(null);
            _wait = null;
            _event?.Dispose();
            _event = null;
            if (_mutex != null)
            {
                try { _mutex.ReleaseMutex(); }
                catch (ApplicationException) { }
                _mutex.Dispose();
                _mutex = null;
            }
        }
    }

    public void Dispose() => Release();
}

/// <summary>The Global mutex the installer checks to see the app running.</summary>
[SupportedOSPlatform("windows")]
public static class InstallerMutex
{
    public static Mutex? Hold(string name, ShellLog log)
    {
        try
        {
            return new Mutex(initiallyOwned: false, name);
        }
        catch (Exception e)
        {
            log.Write("mutex.global_failed", e.GetType().Name);
            return null;
        }
    }
}
