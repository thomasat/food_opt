using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace FoodOptimizer.Core.Windows;

/// <summary>A Job Object for one launch: the launcher and every process it
/// starts. Disposing it kills them all, so a closed window or a Try again
/// never leaves a server holding the port.</summary>
[SupportedOSPlatform("windows")]
public sealed class JobObject : IDisposable
{
    readonly SafeJobHandle _handle;

    public JobObject()
    {
        _handle = NativeMethods.CreateJobObjectW(IntPtr.Zero, null);
        if (_handle.IsInvalid) throw new Win32Exception(Marshal.GetLastPInvokeError());
        var info = new NativeMethods.JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        {
            BasicLimitInformation = new NativeMethods.JOBOBJECT_BASIC_LIMIT_INFORMATION
            {
                LimitFlags = NativeMethods.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                             | NativeMethods.JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION,
            },
        };
        if (!NativeMethods.SetInformationJobObject(_handle, NativeMethods.JobObjectExtendedLimitInformation,
                ref info, (uint)Marshal.SizeOf<NativeMethods.JOBOBJECT_EXTENDED_LIMIT_INFORMATION>()))
        {
            var error = Marshal.GetLastPInvokeError();
            _handle.Dispose();
            throw new Win32Exception(error);
        }
    }

    public void Assign(Process p)
    {
        if (!NativeMethods.AssignProcessToJobObject(_handle, p.SafeHandle))
            throw new Win32Exception(Marshal.GetLastPInvokeError());
    }

    public void Dispose() => _handle.Dispose();
}
