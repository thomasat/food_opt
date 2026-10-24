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
        var messages = Messages.LoadEmbedded();

        var paths = UserPaths.Resolve(Environment.GetEnvironmentVariable,
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile));
        Directory.CreateDirectory(paths.StateDir);
        ShellLog.Rotate(paths.ShellLogPath);
        var log = new ShellLog(paths.ShellLogPath);
        log.Write("shell.start", AppIdentity.Version);

        var settings = ShellSettings.Load(paths.SettingsPath);
        var layout = InstallLayout.FromExeDirectory(Path.TrimEndingDirectorySeparator(AppContext.BaseDirectory));

        using (var form = new MainForm(messages, paths, layout, settings, log))
            Application.Run(form);
        log.Write("shell.quit");
        return 0;
    }
}
