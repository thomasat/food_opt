using System.Windows.Automation;

namespace FoodOptimizer.UiTests;

public class StartupTests
{
    static readonly TimeSpan Ten = TimeSpan.FromSeconds(10);

    static AutomationElement WaitForStartupDialog(ShellApp app)
    {
        AutomationElement? d = null;
        Assert.True(Uia.WaitUntil(() => (d = Uia.TopWindows(app.Process.Id)
            .FirstOrDefault(w => Uia.SafeClass(w) == "#32770" && Uia.NameOf(w) == "Food Optimizer")) != null, TimeSpan.FromSeconds(20)),
            "no Food Optimizer dialog");
        return d!;
    }

    [Fact]
    public async Task Missing_webview2_runtime_says_ask_it()
    {
        var empty = Directory.CreateTempSubdirectory("foodopt-no-webview-").FullName;
        var run = ShellRun.Fresh() with { Env = new Dictionary<string, string> { ["WEBVIEW2_BROWSER_EXECUTABLE_FOLDER"] = empty } };
        using var app = ShellApp.Launch(run, waitForWindow: false);
        var d = WaitForStartupDialog(app);
        Uia.WaitFor(d, ControlType.Text, "Food Optimizer can't open", Ten);
        Uia.WaitFor(d, ControlType.Text, "Ask IT to install Microsoft Edge WebView2 Runtime.", Ten);
        Uia.Invoke(Uia.WaitFor(d, ControlType.Button, "OK", Ten));
        Assert.True(await app.WaitForExitAsync(TimeSpan.FromSeconds(5)));
        Assert.DoesNotContain(app.Log(), l => l.Contains("launcher.start"));
    }

    [Fact]
    public async Task Missing_page_files_say_reinstall()
    {
        var run = ShellRun.Fresh();
        var js = Path.Combine(run.Stage, "shared", "pages", "status.js");
        File.Move(js, js + ".moved");
        try
        {
            using var app = ShellApp.Launch(run, waitForWindow: false);
            var d = WaitForStartupDialog(app);
            Uia.WaitFor(d, ControlType.Text, "Part of the app is missing. Reinstall Food Optimizer.", Ten);
            Uia.Invoke(Uia.WaitFor(d, ControlType.Button, "OK", Ten));
            Assert.True(await app.WaitForExitAsync(TimeSpan.FromSeconds(5)));
            Assert.DoesNotContain(app.Log(), l => l.Contains("launcher.start"));
        }
        finally
        {
            File.Move(js + ".moved", js);
        }
    }
}
