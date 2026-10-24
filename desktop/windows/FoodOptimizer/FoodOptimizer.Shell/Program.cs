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

        var paths = UserPaths.Resolve(Environment.GetEnvironmentVariable,
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile));
        Directory.CreateDirectory(paths.StateDir);
        ShellLog.Rotate(paths.ShellLogPath);
        var log = new ShellLog(paths.ShellLogPath);
        log.Write("shell.start", AppIdentity.Version);

        var settings = ShellSettings.Load(paths.SettingsPath);
        var layout = InstallLayout.FromExeDirectory(Path.TrimEndingDirectorySeparator(AppContext.BaseDirectory));

        MainForm? window = null;
        var crashed = 0;
        void Crash(Exception? ex)
        {
            if (Interlocked.Exchange(ref crashed, 1) == 1) return;
            log.Write("shell.crash", ex?.GetType().FullName ?? "unknown");
            window?.EmergencyShutdown();
            try { ShellDialogs.Info(null, messages.Text("error.stopped.title"), messages.Text("error.shell_crashed.body")); }
            catch (Exception) { }
            Environment.Exit(1);
        }
        Application.ThreadException += (_, e) => Crash(e.Exception);
        AppDomain.CurrentDomain.UnhandledException += (_, e) => Crash(e.ExceptionObject as Exception);

        using (window = new MainForm(messages, paths, layout, settings, log))
            Application.Run(window);
        log.Write("shell.quit");
        return 0;
    }
}
