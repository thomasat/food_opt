using System.Runtime.CompilerServices;
using System.Windows.Automation;
using WinForms = System.Windows.Forms;

namespace FoodOptimizer.UiTests;

public class NavigationTests
{
    static readonly TimeSpan Three = TimeSpan.FromSeconds(3);
    static readonly TimeSpan Five = TimeSpan.FromSeconds(5);
    static readonly TimeSpan Ten = TimeSpan.FromSeconds(10);

    static ShellApp Loaded([CallerMemberName] string test = "")
    {
        var app = ShellApp.Launch(ShellRun.Fresh(test));
        app.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
        return app;
    }

    static bool Logged(ShellApp app, string line, TimeSpan timeout) =>
        Uia.WaitUntil(() => app.Log().Any(l => l.Contains(line)), timeout);

    /// <summary>The handles of every top-level window now.</summary>
    static HashSet<int> TopHandles() =>
        AutomationElement.RootElement.FindAll(TreeScope.Children, Condition.TrueCondition)
            .Cast<AutomationElement>().Select(Handle).ToHashSet();

    static int Handle(AutomationElement w)
    {
        try { return w.Current.NativeWindowHandle; }
        catch (Exception) { return 0; }
    }

    static bool IsProbePage(AutomationElement w)
    {
        var name = Uia.NameOf(w);
        return name.Contains("Example Domain") || name.Contains("foodopt-") || name.Contains("example.com");
    }

    /// <summary>Closes the browser windows the link opened (new since
    /// <paramref name="before"/>), never one that was open already. A
    /// browser that starts cold can show its window seconds after the
    /// test is done; left open, it takes the foreground (and the keys) from
    /// the next test, so it is waited for first.</summary>
    static void CloseBrowserWindows(HashSet<int> before)
    {
        AutomationElement[] Opened() =>
            AutomationElement.RootElement.FindAll(TreeScope.Children, Condition.TrueCondition)
                .Cast<AutomationElement>().Where(w => !before.Contains(Handle(w)) && IsProbePage(w)).ToArray();
        // In a finally: a failure here must never hide the test's own.
        try
        {
            Uia.WaitUntil(() => Opened().Length > 0, TimeSpan.FromSeconds(20));
            Thread.Sleep(500);
            foreach (var w in Opened())
            {
                try { ((WindowPattern)w.GetCurrentPattern(WindowPattern.Pattern)).Close(); } catch (Exception) { }
            }
            Uia.WaitUntil(() => Opened().Length == 0, Five);
        }
        catch (Exception) { }
    }

    static string Describe(IEnumerable<AutomationElement> windows) =>
        string.Join("; ", windows.Select(w => $"{Uia.SafeClass(w)} '{Uia.NameOf(w)}'"));

    /// <summary>The app still shows its page, in its one window: the link
    /// opened no window of the app's own. Waited for, so a window that
    /// shows only for a moment while the browser takes the foreground is
    /// not counted; one that stays fails, naming what it is.</summary>
    static void AssertStillOnTheApp(ShellApp app)
    {
        Assert.True(Uia.WaitUntil(() => app.Exists(ControlType.Edit, "Type here"), Three));
        IReadOnlyList<AutomationElement> windows = [];
        Assert.True(Uia.WaitUntil(() => (windows = Uia.TopWindows(app.Process.Id)).Count == 1, Three),
            "the app's top-level windows: " + Describe(windows));
    }

    [Fact]
    public void A_blank_target_link_opens_in_the_browser()
    {
        using var app = Loaded();
        var before = TopHandles();
        try
        {
            Uia.Invoke(app.WaitFor(ControlType.Hyperlink, "Blank link", Ten));
            Assert.True(Logged(app, "open.external https://example.com/foodopt-blank", Five), string.Join("\n", app.Log()));
            AssertStillOnTheApp(app);
        }
        finally { CloseBrowserWindows(before); }
    }

    [Fact]
    public void A_same_window_external_link_opens_in_the_browser()
    {
        using var app = Loaded();
        var before = TopHandles();
        try
        {
            Uia.Invoke(app.WaitFor(ControlType.Hyperlink, "Same link", Ten));
            Assert.True(Logged(app, "open.external https://example.com/foodopt-same", Five), string.Join("\n", app.Log()));
            AssertStillOnTheApp(app);
        }
        finally { CloseBrowserWindows(before); }
    }

    [Fact]
    public void Window_open_goes_to_the_browser()
    {
        using var app = Loaded();
        var before = TopHandles();
        try
        {
            Uia.Invoke(app.WaitFor(ControlType.Button, "Open window", Ten));
            Assert.True(Logged(app, "open.external https://example.com/foodopt-open", Five), string.Join("\n", app.Log()));
            AssertStillOnTheApp(app);
        }
        finally { CloseBrowserWindows(before); }
    }

    [Fact]
    public void A_file_link_is_blocked()
    {
        using var app = Loaded();
        var before = app.Log().Count;
        Uia.Invoke(app.WaitFor(ControlType.Hyperlink, "File link", Ten));
        Thread.Sleep(3000);
        // Chromium itself refuses a file: link on an http page before any
        // NavigationStarting; when one does start, the shell blocks it.
        var after = app.Log().Skip(before).ToList();
        Assert.True(after.Any(l => l.Contains("nav.blocked file:///C:/Windows/win.ini")) || !after.Any(l => l.Contains("navigate") || l.Contains("open.external")),
            string.Join("\n", after));
        AssertStillOnTheApp(app);
    }

    [Fact]
    public void Outside_resources_are_blocked()
    {
        using var app = Loaded();
        Assert.True(Logged(app, "request.blocked https://example.com/foodopt-probe.png", Ten), string.Join("\n", app.Log()));
    }

    [Fact]
    public void Back_button_does_not_leave_the_app()
    {
        using var app = Loaded();
        var before = app.Log().Count;
        app.MouseBack();
        Thread.Sleep(3000);
        Assert.True(app.Exists(ControlType.Edit, "Type here"));
        var after = app.Log().Skip(before).ToList();
        Assert.True(after.Any(l => l.Contains("nav.history_blocked")) || !after.Any(l => l.Contains("nav.")),
            string.Join("\n", after));
        Assert.DoesNotContain(after, l => l.Contains("app.navigate"));
    }

    [Fact]
    public void Alert_confirm_and_prompt_use_food_optimizer_dialogs()
    {
        using var app = Loaded();
        Uia.Invoke(app.WaitFor(ControlType.Button, "Alert", Ten));
        var alert = app.WaitForDialog(Ten);
        Uia.WaitFor(alert, ControlType.Text, "probe alert", Three);
        Uia.Invoke(Uia.WaitFor(alert, ControlType.Button, "OK", Three));
        Assert.True(Uia.WaitUntil(() => app.Dialog("Food Optimizer") == null, Five));

        Uia.Invoke(app.WaitFor(ControlType.Button, "Confirm", Ten));
        var confirm = app.WaitForDialog(Ten);
        Uia.WaitFor(confirm, ControlType.Button, "OK", Three);
        Uia.Invoke(Uia.WaitFor(confirm, ControlType.Button, "Cancel", Three));
        app.WaitFor(ControlType.Text, "confirm: false", Ten);

        Uia.Invoke(app.WaitFor(ControlType.Button, "Prompt", Ten));
        var prompt = app.WaitForDialog(Ten);
        var box = Uia.WaitFor(prompt, ControlType.Edit, "", Three);
        box.SetFocus();
        WinForms.SendKeys.SendWait("abc");
        Thread.Sleep(300);
        Uia.Invoke(Uia.WaitFor(prompt, ControlType.Button, "OK", Three));
        app.WaitFor(ControlType.Text, "prompt: abc", Ten);
    }

    /// WebView2's context menu is drawn by the browser process and is not
    /// in the UI Automation tree, so the shell logs what it kept.
    [Fact]
    public void Context_menu_has_only_editing_items()
    {
        using var app = Loaded();
        app.RightClick(app.WaitFor(ControlType.Edit, "Type here", Ten));
        string? kept = null;
        Assert.True(Uia.WaitUntil(() => (kept = app.Log().LastOrDefault(l => l.Contains("context_menu."))) != null, Five),
            string.Join("\n", app.Log()));
        var line = kept!;
        Assert.Contains("context_menu.kept ", line);
        var names = line[(line.IndexOf("context_menu.kept ") + 18)..].Split(',');
        Assert.Contains("paste", names);
        Assert.All(names, n => Assert.Contains(n, new[] { "cut", "copy", "paste", "selectAll" }));
        WinForms.SendKeys.SendWait("{ESC}");
        Thread.Sleep(500);

        var before = app.Log().Count;
        app.RightClickWindowAt(0.9, 0.9);
        Assert.True(Uia.WaitUntil(() => app.Log().Skip(before).Any(l => l.Contains("context_menu.suppressed")), Five),
            string.Join("\n", app.Log()));
    }

    [Fact]
    public void A_retry_message_from_the_app_page_is_ignored()
    {
        using var app = Loaded();
        var serve = app.StagePythons().Single(p => p.CommandLine.Contains(" serve ")).Pid;
        Uia.Invoke(app.WaitFor(ControlType.Button, "Post retry", Ten));
        Assert.True(Logged(app, "message.ignored", Five), string.Join("\n", app.Log()));
        Thread.Sleep(5000);
        Assert.Equal(serve, app.StagePythons().Single(p => p.CommandLine.Contains(" serve ")).Pid);
        Assert.True(app.Exists(ControlType.Edit, "Type here"));
    }
}
