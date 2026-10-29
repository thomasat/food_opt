using FoodOptimizer.Core;
using FoodOptimizer.Core.Windows;

namespace FoodOptimizer.Shell;

static class Program
{
    [STAThread]
    static int Main()
    {
        var paths = UserPaths.Resolve(Environment.GetEnvironmentVariable,
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile));
        Directory.CreateDirectory(paths.StateDir);
        var log = new ShellLog(paths.ShellLogPath);
        var layout = InstallLayout.FromExeDirectory(Path.TrimEndingDirectorySeparator(AppContext.BaseDirectory));
        // D1: a per-user copy on a PC with the IT copy opens the IT copy and
        // quits, before it holds any name: the IT copy then runs this copy's
        // own uninstaller. The IT copy never hands off.
        if (ItCopyHandoff.Run(layout, log)) return 0;

        WinShell.SetAppUserModelId(AppIdentity.AppUserModelId);
        ApplicationConfiguration.Initialize();
        Application.SetUnhandledExceptionMode(UnhandledExceptionMode.CatchException);
        var messages = Messages.LoadEmbedded();
        ShellDialogs.Messages = messages;

        // One window per user: a second launch brings the first one back.
        MainForm? window = null;
        var instance = SingleInstance.TryAcquire(AppIdentity.SingleInstanceMutex, AppIdentity.ActivateEvent,
            () => window?.BringBack());
        if (instance == null)
        {
            SingleInstance.SignalFirst(AppIdentity.ActivateEvent);
            return 0;
        }

        ShellLog.Rotate(paths.ShellLogPath);
        log.Write("shell.start", AppIdentity.Version);
        // The name the installer and uninstaller of this copy's scope check.
        using var installerMutex = InstallerMutex.Hold(layout.RunningMutex, log);

        if (!WebViewRuntimePresent(log))
        {
            ShellDialogs.Info(null, messages.Text("webview2.missing.title"), messages.Text("webview2.missing.body"));
            return 0;
        }
        if (layout.MissingForPage() is { Count: > 0 } missing)
        {
            log.Write("page.missing", string.Join(";", missing));
            ShellDialogs.Info(null, messages.Text("error.missing_part.title"), messages.Text("error.missing_part.body"));
            return 0;
        }

        var settings = ShellSettings.Load(paths.SettingsPath);
        // D1: an all-users copy hands this user's per-user copy, if any, to
        // that copy's own uninstaller (started here, not waited for).
        PerUserCleanup.Run(layout, log);

        var crashed = 0;
        void Crash(Exception? ex)
        {
            if (Interlocked.Exchange(ref crashed, 1) == 1) return;
            log.Write("shell.crash", ex?.GetType().FullName ?? "unknown");
            window?.EmergencyShutdown();
            // An unhandled exception may come on a pool (MTA) thread; the
            // dialog needs an STA thread.
            try { StaThread.Run(() => ShellDialogs.Info(null, messages.Text("error.stopped.title"), messages.Text("error.shell_crashed.body"))); }
            catch (Exception) { }
            Environment.Exit(1);
        }
        Application.ThreadException += (_, e) => Crash(e.Exception);
        AppDomain.CurrentDomain.UnhandledException += (_, e) => Crash(e.ExceptionObject as Exception);

        using (window = new MainForm(messages, paths, layout, settings, log) { Instance = instance })
            Application.Run(window);
        instance.Release();
        log.Write("shell.quit");
        return 0;
    }

    /// <summary>The WebView2 runtime is installed (the shell can't show anything without it).</summary>
    static bool WebViewRuntimePresent(ShellLog log)
    {
        try
        {
            var version = Microsoft.Web.WebView2.Core.CoreWebView2Environment.GetAvailableBrowserVersionString();
            if (!string.IsNullOrEmpty(version)) return true;
            log.Write("webview.missing");
        }
        catch (Exception e)
        {
            log.Write("webview.missing", e.GetType().Name);
        }
        return false;
    }
}
