using FoodOptimizer.Core;
using FoodOptimizer.Core.Windows;

namespace FoodOptimizer.Shell;

static class Program
{
    [STAThread]
    static int Main()
    {
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

        var paths = UserPaths.Resolve(Environment.GetEnvironmentVariable,
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile));
        Directory.CreateDirectory(paths.StateDir);
        ShellLog.Rotate(paths.ShellLogPath);
        var log = new ShellLog(paths.ShellLogPath);
        log.Write("shell.start", AppIdentity.Version);
        using var installerMutex = InstallerMutex.Hold(AppIdentity.InstallerMutex, log);

        if (!WebViewRuntimePresent(log))
        {
            ShellDialogs.Info(null, messages.Text("webview2.missing.title"), messages.Text("webview2.missing.body"));
            return 0;
        }
        var layout = InstallLayout.FromExeDirectory(Path.TrimEndingDirectorySeparator(AppContext.BaseDirectory));
        if (layout.MissingForPage() is { Count: > 0 } missing)
        {
            log.Write("page.missing", string.Join(";", missing));
            ShellDialogs.Info(null, messages.Text("error.missing_part.title"), messages.Text("error.missing_part.body"));
            return 0;
        }

        var settings = ShellSettings.Load(paths.SettingsPath);

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
