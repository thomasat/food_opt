using System.Diagnostics;
using System.Windows.Automation;

namespace FoodOptimizer.UiTests;

/// <summary>The real Food Optimizer app in the shell (stage_layout.ps1 -App real).</summary>
[Trait("Category", "RealApp")]
public class RealAppTests
{
    static string AppTitle(string stage)
    {
        var info = new ProcessStartInfo(Path.Combine(stage, "python", "python.exe"))
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            CreateNoWindow = true,
        };
        foreach (var a in new[] { "-I", "-c", "import sys, wording; sys.stdout.write(wording.APP_TITLE)" })
            info.ArgumentList.Add(a);
        using var p = Process.Start(info)!;
        var title = p.StandardOutput.ReadToEnd();
        p.WaitForExit();
        Assert.Equal(0, p.ExitCode);
        return title;
    }

    [Fact]
    public async Task Food_optimizer_opens_its_first_screen()
    {
        var title = AppTitle(Uia.Stage);
        Assert.False(string.IsNullOrEmpty(title));
        using var app = ShellApp.Launch(ShellRun.Fresh());
        Assert.True(Uia.WaitUntil(() => Uia.All(app.Window, ControlType.Document).Any(d => Uia.NameOf(d) == title),
            TimeSpan.FromSeconds(180), 1000), "no page titled " + title);
        Uia.Record("real_app_title", title);

        app.Menu("Help", "Get Help");
        var d = app.WaitForDialog(TimeSpan.FromSeconds(10));
        Uia.WaitFor(d, ControlType.Text, "Food Optimizer support", TimeSpan.FromSeconds(3));
        Uia.Invoke(Uia.WaitFor(d, ControlType.Button, "OK", TimeSpan.FromSeconds(3)));

        app.Close();
        Assert.True(await app.WaitForExitAsync(TimeSpan.FromSeconds(10)));
        Assert.True(Uia.WaitUntil(() => app.StagePythons().Count == 0, TimeSpan.FromSeconds(5)),
            string.Join("; ", app.StagePythons().Select(p => p.CommandLine)));
    }
}
