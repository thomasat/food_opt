using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Security.AccessControl;
using System.Security.Principal;
using FoodOptimizer.Core;
using FoodOptimizer.Core.Windows;

namespace FoodOptimizer.Tests;

public class SingleInstanceTests
{
    const int Synchronize = 0x00100000;
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
        var logPath = TempLog();
        var held = InstallerMutex.Hold(name, new ShellLog(logPath));
        Assert.NotNull(held);
        Assert.Contains(" mutex.held " + name + "\n", File.ReadAllText(logPath));
        Assert.True(Mutex.TryOpenExisting(name, out var other));
        other.Dispose();
        held.Dispose();
        Assert.False(Mutex.TryOpenExisting(name, out _));
        File.Delete(logPath);
    }

    /// The all-users copy's name: every signed-in user's shell may open it
    /// (SYNCHRONIZE, all it needs to keep it alive and to be seen), and
    /// nobody but SYSTEM, Administrators and its creator may do more.
    [Fact]
    [SupportedOSPlatform("windows")]
    public void A_global_mutex_lets_every_signed_in_user_see_it()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        var name = @"Global\FoodOptimizer.TestRunning." + Guid.NewGuid().ToString("N");
        var logPath = TempLog();
        var held = InstallerMutex.Hold(name, new ShellLog(logPath));
        Assert.NotNull(held);
        try
        {
            Assert.Contains(" mutex.held " + name + "\n", File.ReadAllText(logPath));
            var aces = Dacl(held).OfType<CommonAce>().ToList();
            var au = new SecurityIdentifier(WellKnownSidType.AuthenticatedUserSid, null);
            var forAu = aces.Where(a => a.SecurityIdentifier == au).ToList();
            Assert.Contains(forAu, a => a.AceQualifier == AceQualifier.AccessAllowed && a.AccessMask == Synchronize);
            Assert.All(forAu, a => Assert.True(a.AceQualifier != AceQualifier.AccessAllowed || a.AccessMask == Synchronize,
                $"Authenticated Users are allowed 0x{a.AccessMask:X}"));
            var allowed = aces.Where(a => a.AceQualifier == AceQualifier.AccessAllowed).Select(a => a.SecurityIdentifier).ToHashSet();
            var me = WindowsIdentity.GetCurrent().User!;
            Assert.Equal(new HashSet<SecurityIdentifier>
            {
                new(WellKnownSidType.LocalSystemSid, null),
                new(WellKnownSidType.BuiltinAdministratorsSid, null),
                me,
                au,
            }, allowed);
            Assert.True(Mutex.TryOpenExisting(name, out var other));
            other.Dispose();
        }
        finally
        {
            held.Dispose();
            File.Delete(logPath);
        }
        Assert.False(Mutex.TryOpenExisting(name, out _));
    }

    static string TempLog() => Path.Combine(Path.GetTempPath(), "foodopt-mutex-test-" + Guid.NewGuid().ToString("N") + ".log");

    /// The handle's DACL as the kernel holds it (GetSecurityInfo; the
    /// managed GetAccessControl needs a package the app does not ship).
    [SupportedOSPlatform("windows")]
    static RawAcl Dacl(WaitHandle handle)
    {
        const int SeKernelObject = 6, DaclSecurityInformation = 4;
        var rc = GetSecurityInfo(handle.SafeWaitHandle.DangerousGetHandle(), SeKernelObject, DaclSecurityInformation,
            out _, out _, out _, out _, out var sd);
        Assert.Equal(0, rc);
        try
        {
            var bytes = new byte[GetSecurityDescriptorLength(sd)];
            Marshal.Copy(sd, bytes, 0, bytes.Length);
            return new RawSecurityDescriptor(bytes, 0).DiscretionaryAcl!;
        }
        finally
        {
            LocalFree(sd);
        }
    }

    [DllImport("advapi32.dll")]
    static extern int GetSecurityInfo(IntPtr handle, int objectType, int securityInfo,
        out IntPtr owner, out IntPtr group, out IntPtr dacl, out IntPtr sacl, out IntPtr securityDescriptor);

    [DllImport("advapi32.dll")]
    static extern int GetSecurityDescriptorLength(IntPtr securityDescriptor);

    [DllImport("kernel32.dll")]
    static extern IntPtr LocalFree(IntPtr memory);
}
