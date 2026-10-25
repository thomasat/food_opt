using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Windows.Automation;
using FoodOptimizer.Core;

namespace FoodOptimizer.UiTests;

/// <summary>The app as the installer leaves it: Try again on a real server
/// (spec Level 2: Try again leaves exactly one server) and the taskbar
/// identity the Start menu shortcut and the window share.</summary>
[Trait("Category", "Installed")]
public class InstalledAppTests
{
    static (int Pid, string CommandLine)[] Serves(string dir) =>
        ShellApp.StagePythonsOf(dir).Where(p => p.CommandLine.Contains("child.py") && p.CommandLine.Contains(" serve ")).ToArray();

    static (int Pid, string CommandLine)[] Launchers(string dir) =>
        ShellApp.StagePythonsOf(dir).Where(p => p.CommandLine.Contains("-m launcher")).ToArray();

    [Fact]
    public void Try_again_on_the_installed_app_leaves_one_server()
    {
        var run = InstallerRun.Fresh("retry");
        try
        {
            InstallerRun.SilentInstall(run);
            var title = InstallerRun.AppTitle(run.Stage);
            using var app = ShellApp.Launch(run);
            Assert.True(Uia.WaitUntil(() => InstallerRun.ShowsDocument(app.Window, title), TimeSpan.FromSeconds(300), 1000));
            var serve = Assert.Single(Serves(run.Stage));
            Process.GetProcessById(serve.Pid).Kill();
            app.WaitFor(ControlType.Text, "Food Optimizer stopped unexpectedly", TimeSpan.FromSeconds(45));
            var retry = app.WaitFor(ControlType.Button, "Try again", TimeSpan.FromSeconds(30));
            Uia.Invoke(retry);
            Thread.Sleep(100);
            try { Uia.Invoke(retry); } catch (Exception) { }
            Assert.True(Uia.WaitUntil(() => InstallerRun.ShowsDocument(app.Window, title), TimeSpan.FromSeconds(180), 1000));
            Thread.Sleep(5000);
            Assert.Single(Serves(run.Stage));
            Assert.Single(Launchers(run.Stage));
        }
        finally
        {
            InstallerRun.SilentUninstall(run.Stage, InstallerRun.Env(run));
        }
    }

    [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IPropertyStore
    {
        void GetCount(out uint count);
        void GetAt(uint index, out PropertyKey key);
        void GetValue(ref PropertyKey key, out PropVariant value);
        void SetValue(ref PropertyKey key, ref PropVariant value);
        void Commit();
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct PropertyKey
    {
        public Guid FormatId;
        public uint PropertyId;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct PropVariant
    {
        public ushort Type;
        public ushort Reserved1, Reserved2, Reserved3;
        public IntPtr Value;
        public IntPtr Value2;
    }

    [DllImport("shell32.dll")]
    static extern int SHGetPropertyStoreForWindow(IntPtr hwnd, ref Guid riid, out IPropertyStore store);

    [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
    static extern int SHGetPropertyStoreFromParsingName(string path, IntPtr bindContext, int flags, ref Guid riid, out IPropertyStore store);

    [DllImport("ole32.dll")]
    static extern int PropVariantClear(ref PropVariant value);

    static readonly Guid PropertyStoreId = new("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99");

    /// <summary>PKEY_AppUserModel_ID from a property store ("" when unset).</summary>
    static string AppUserModelId(IPropertyStore store)
    {
        var key = new PropertyKey { FormatId = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), PropertyId = 5 };
        store.GetValue(ref key, out var value);
        try
        {
            return value.Type == 31 ? Marshal.PtrToStringUni(value.Value) ?? "" : "";     // VT_LPWSTR
        }
        finally { PropVariantClear(ref value); }
    }

    [Fact]
    public void Window_and_shortcut_share_the_app_id()
    {
        var run = InstallerRun.Fresh("appid");
        try
        {
            InstallerRun.SilentInstall(run);
            var lnk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs), "Food Optimizer.lnk");
            var riid = PropertyStoreId;
            Assert.Equal(0, SHGetPropertyStoreFromParsingName(lnk, IntPtr.Zero, 0, ref riid, out var lnkStore));
            Assert.Equal(AppIdentity.AppUserModelId, AppUserModelId(lnkStore));
            // The shell sets the ID for its whole process
            // (SetCurrentProcessExplicitAppUserModelID), which the taskbar
            // uses for every window that sets none of its own; a window may
            // carry only that same ID, never another.
            using var app = ShellApp.Launch(run);
            riid = PropertyStoreId;
            Assert.Equal(0, SHGetPropertyStoreForWindow(app.Hwnd, ref riid, out var windowStore));
            Assert.Contains(AppUserModelId(windowStore), new[] { "", AppIdentity.AppUserModelId });
        }
        finally
        {
            InstallerRun.SilentUninstall(run.Stage, InstallerRun.Env(run));
        }
    }
}
