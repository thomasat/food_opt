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

    /// A WebView2 folder that cannot be made (a file in its place) is
    /// refused before WebView2 starts on it: WebView2's own "couldn't create
    /// the data directory" box, which waited for a click while the shell
    /// waited for WebView2, never comes up, and the stopped dialog does.
    [Fact]
    public async Task A_web_view_that_cannot_start_says_so()
    {
        var run = ShellRun.Fresh();
        Directory.CreateDirectory(run.StateDir);
        var folder = Path.Combine(run.StateDir, "WebView2");
        File.WriteAllText(folder, "not a folder");
        using var app = ShellApp.Launch(run);
        var browsers = new HashSet<int>();
        AutomationElement? d = null;
        Assert.True(Uia.WaitUntil(() =>
        {
            browsers.UnionWith(ShellApp.WebViewBrowsersUsing(folder));
            return (d = app.Dialog("Food Optimizer")) != null;
        }, TimeSpan.FromSeconds(30), 100), "no Food Optimizer dialog within 30 s:\n" + string.Join("\n", app.Log()));
        Uia.WaitFor(d!, ControlType.Text, "Food Optimizer stopped unexpectedly", Ten);
        Uia.WaitFor(d!, ControlType.Text, "Your projects are saved. Open Food Optimizer again.", Ten);
        var log = app.Log();
        Assert.True(browsers.Count == 0, "WebView2 started on the file: pids " + string.Join(",", browsers));
        Assert.DoesNotContain(log, l => l.Contains(" webview.runtime ", StringComparison.Ordinal));
        Assert.Equal(3, log.Count(l => l.EndsWith(" webview.failed IOException", StringComparison.Ordinal)));
        Uia.Invoke(Uia.WaitFor(d!, ControlType.Button, "OK", Ten));
        Assert.True(await app.WaitForExitAsync(TimeSpan.FromSeconds(10)));
        Assert.True(Uia.WaitUntil(() => app.StagePythons().Count == 0, TimeSpan.FromSeconds(5)));
        Assert.True(File.Exists(folder));
    }
}
