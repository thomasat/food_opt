using System.Diagnostics;
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

    [Fact]
    public void Window_and_shortcut_share_the_app_id()
    {
        var run = InstallerRun.Fresh("appid");
        try
        {
            InstallerRun.SilentInstall(run);
            var lnk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs), "Food Optimizer.lnk");
            Assert.Equal(AppIdentity.AppUserModelId, AppIdProbe.OfShortcut(lnk));
            // The window carries the shortcut's ID itself, not only through
            // the process-wide one, so the taskbar pins and groups it with
            // the Start menu entry.
            using var app = ShellApp.Launch(run);
            Assert.Equal(AppIdentity.AppUserModelId, AppIdProbe.OfWindow(app.Hwnd));
        }
        finally
        {
            InstallerRun.SilentUninstall(run.Stage, InstallerRun.Env(run));
        }
    }
}
