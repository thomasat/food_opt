using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

public sealed class InstallLayoutTests : IDisposable
{
    readonly string _dir = Directory.CreateTempSubdirectory("foodopt-layout-").FullName;

    public void Dispose() => Directory.Delete(_dir, true);

    [Fact]
    public void Arguments_are_the_phase_2_contract()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows paths");
        var layout = InstallLayout.FromExeDirectory(@"C:\Program Files\Food Optimizer");
        var paths = new UserPaths(@"C:\Users\José Müller\AppData\Local\FoodOptimizer", @"C:\Users\José Müller\FoodOptimizer");
        Assert.Equal(@"C:\Program Files\Food Optimizer\python\python.exe", layout.Python);
        Assert.Equal(
            [
                "-I", "-X", "utf8", "-B", "-m", "launcher",
                "--state-dir", @"C:\Users\José Müller\AppData\Local\FoodOptimizer",
                "--projects-dir", @"C:\Users\José Müller\FoodOptimizer",
                "--app", @"C:\Program Files\Food Optimizer\app\app.py",
                "--stdin-lifeline",
            ],
            LauncherCommand.Arguments(layout, paths));
        Assert.Equal(@"C:\Program Files\Food Optimizer\shared\pages", layout.PagesDir);
        Assert.Equal(@"C:\Program Files\Food Optimizer\Start Here.txt", layout.StartHere);
        Assert.Equal(@"C:\Users\José Müller\AppData\Local\FoodOptimizer\settings.json", paths.SettingsPath);
        Assert.Equal(@"C:\Users\José Müller\AppData\Local\FoodOptimizer\WebView2", paths.WebViewData);
        Assert.Equal(@"C:\Users\José Müller\AppData\Local\FoodOptimizer\launcher.log", paths.LauncherLog);
        Assert.Equal(@"C:\Users\José Müller\AppData\Local\FoodOptimizer\shell.log", paths.ShellLogPath);
    }

    [Fact]
    public void Resolve_uses_the_overrides_when_set()
    {
        var env = new Dictionary<string, string?> { ["FOODOPT_STATE_DIR"] = "/s", ["FOODOPT_PROJECTS_DIR"] = "/p" };
        Assert.Equal(new UserPaths("/s", "/p"), UserPaths.Resolve(n => env.GetValueOrDefault(n), "/l", "/u"));

        var empty = new Dictionary<string, string?> { ["FOODOPT_STATE_DIR"] = "" };
        Assert.Equal(new UserPaths(Path.Combine("/l", "FoodOptimizer"), Path.Combine("/u", "FoodOptimizer")),
            UserPaths.Resolve(n => empty.GetValueOrDefault(n), "/l", "/u"));
    }

    [Fact]
    public void Missing_parts_are_listed()
    {
        var l = InstallLayout.FromExeDirectory(_dir);
        Assert.Equal(3, l.MissingForPage().Count);
        Assert.Equal([l.Python, l.LauncherMain, l.App], l.MissingForLaunch());

        Directory.CreateDirectory(l.PagesDir);
        foreach (var f in new[] { "status.html", "status.css" }) File.WriteAllText(Path.Combine(l.PagesDir, f), "");
        Assert.Equal([Path.Combine(l.PagesDir, "status.js")], l.MissingForPage());
        Assert.Equal(Path.Combine(l.PagesDir, "status.html"), l.StatusPage);

        foreach (var f in new[] { l.Python, l.LauncherMain, l.App })
        {
            Directory.CreateDirectory(Path.GetDirectoryName(f)!);
            File.WriteAllText(f, "");
        }
        Assert.Empty(l.MissingForLaunch());
    }
}
