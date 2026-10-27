namespace FoodOptimizer.Core;

/// <summary>The per-user folders: state (%LOCALAPPDATA%\FoodOptimizer) and
/// projects (%USERPROFILE%\FoodOptimizer). FOODOPT_STATE_DIR and
/// FOODOPT_PROJECTS_DIR override them for tests.</summary>
public sealed record UserPaths(string StateDir, string ProjectsDir)
{
    const string Folder = "FoodOptimizer";

    public static UserPaths Resolve(Func<string, string?> env, string localAppData, string userProfile)
    {
        var state = env("FOODOPT_STATE_DIR");
        var projects = env("FOODOPT_PROJECTS_DIR");
        return new UserPaths(
            string.IsNullOrEmpty(state) ? Path.Combine(localAppData, Folder) : state,
            string.IsNullOrEmpty(projects) ? Path.Combine(userProfile, Folder) : projects);
    }

    public string LauncherLog => Path.Combine(StateDir, StateFiles.LauncherLog);
    public string ShellLogPath => Path.Combine(StateDir, StateFiles.ShellLog);
    public string SettingsPath => Path.Combine(StateDir, StateFiles.Settings);
    public string WebViewData => Path.Combine(StateDir, StateFiles.WebViewData);
}

/// <summary>The install folder: the exe's folder, holding the bundled
/// Python, the app and the shared launcher and page.</summary>
public sealed record InstallLayout(string Root)
{
    public static InstallLayout FromExeDirectory(string dir) => new(dir);

    public string Python => Path.Combine(Root, "python", "python.exe");
    public string App => Path.Combine(Root, "app", "app.py");
    public string LauncherMain => Path.Combine(Root, "shared", "launcher", "__main__.py");
    public string PagesDir => Path.Combine(Root, "shared", "pages");
    public string StatusPage => Path.Combine(PagesDir, "status.html");
    public string StartHere => Path.Combine(Root, "Start Here.txt");

    /// <summary>Installed for all users: the all-users installer writes the
    /// file "all-users" into its folder (a per-user one never does).</summary>
    public bool AllUsers => File.Exists(Path.Combine(Root, "all-users"));

    /// <summary>The name this copy's running app holds: the one the
    /// installer and uninstaller of its scope check.</summary>
    public string RunningMutex => AllUsers ? AppIdentity.RunningAllUsers : AppIdentity.RunningPerUser;

    /// <summary>The page files that are missing (the window can show no page without them).</summary>
    public IReadOnlyList<string> MissingForPage() =>
        new[] { "status.html", "status.css", "status.js" }
            .Select(f => Path.Combine(PagesDir, f)).Where(p => !File.Exists(p)).ToList();

    /// <summary>The launch parts that are missing.</summary>
    public IReadOnlyList<string> MissingForLaunch() =>
        new[] { Python, LauncherMain, App }.Where(p => !File.Exists(p)).ToList();
}

/// <summary>The launcher's command line: the phase 2 contract.</summary>
public static class LauncherCommand
{
    public static IReadOnlyList<string> Arguments(InstallLayout l, UserPaths p) =>
    [
        "-I", "-X", "utf8", "-B", "-m", "launcher",
        "--state-dir", p.StateDir,
        "--projects-dir", p.ProjectsDir,
        "--app", l.App,
        "--stdin-lifeline",
    ];
}
