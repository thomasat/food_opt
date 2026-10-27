using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Security.Principal;
using Microsoft.Win32.SafeHandles;

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

/// <summary>The running-app name the installer checks (AppIdentity's
/// RunningPerUser or RunningAllUsers). A Local name is made as any mutex.
/// A Global name gets a DACL that lets every signed-in user wait on it
/// (SYNCHRONIZE, which is all an installer's check or another shell
/// needs) and nobody else do more than SYSTEM, Administrators and its
/// maker; when another user's app made it first, it is opened for waiting
/// only. Null, logged, when it can't be held.</summary>
[SupportedOSPlatform("windows")]
public static class InstallerMutex
{
    const uint Synchronize = 0x00100000;
    const int ErrorAccessDenied = 5, ErrorAlreadyExists = 183;

    /// <summary>The Global name's security descriptor, in SDDL.</summary>
    public static string GlobalSddl(string userSid) =>
        "D:P(A;;GA;;;SY)(A;;GA;;;BA)(A;;GA;;;" + userSid + ")(A;;0x00100000;;;AU)";

    /// <summary>Holds the name; logs "mutex.held &lt;name&gt; created" (this
    /// app made it) or "... opened" (another app had it already).</summary>
    public static WaitHandle? Hold(string name, ShellLog log)
    {
        var global = name.StartsWith(@"Global\", StringComparison.OrdinalIgnoreCase);
        try
        {
            bool created;
            WaitHandle held = global
                ? HoldGlobal(name, out created)
                : new Mutex(initiallyOwned: false, name, out created);
            log.Write("mutex.held", name + (created ? " created" : " opened"));
            return held;
        }
        catch (Exception e)
        {
            log.Write(global ? "mutex.global_failed" : "mutex.failed",
                e is Win32Exception w ? "win32_" + w.NativeErrorCode : e.GetType().Name);
            return null;
        }
    }

    static WaitHandle HoldGlobal(string name, out bool created)
    {
        string user;
        using (var identity = WindowsIdentity.GetCurrent())
            user = identity.User?.Value ?? throw new InvalidOperationException("token.no_user");
        if (!NativeMethods.ConvertStringSecurityDescriptorToSecurityDescriptorW(GlobalSddl(user), 1, out var sd, IntPtr.Zero))
            throw new Win32Exception(Marshal.GetLastPInvokeError());
        try
        {
            var attributes = new NativeMethods.SECURITY_ATTRIBUTES
            {
                Length = (uint)Marshal.SizeOf<NativeMethods.SECURITY_ATTRIBUTES>(),
                SecurityDescriptor = sd,
            };
            var handle = NativeMethods.CreateMutexW(attributes, false, name);
            var error = Marshal.GetLastPInvokeError();
            if (!handle.IsInvalid)
            {
                created = error != ErrorAlreadyExists;
                return new Held(handle);
            }
            handle.Dispose();
            if (error != ErrorAccessDenied) throw new Win32Exception(error);
            // Another user's all-users app made it: waiting is all we may, and need.
            handle = NativeMethods.OpenMutexW(Synchronize, false, name);
            if (handle.IsInvalid)
            {
                error = Marshal.GetLastPInvokeError();
                handle.Dispose();
                throw new Win32Exception(error);
            }
            created = false;
            return new Held(handle);
        }
        finally
        {
            NativeMethods.LocalFree(sd);
        }
    }

    /// <summary>A kernel handle kept open until disposed.</summary>
    sealed class Held : WaitHandle
    {
        public Held(SafeWaitHandle handle) => SafeWaitHandle = handle;
    }
}
