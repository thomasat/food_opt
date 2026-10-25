using System.Globalization;
using System.Windows.Automation;
using FoodOptimizer.Core;
using WinForms = System.Windows.Forms;

namespace FoodOptimizer.UiTests;

public class KeysAndMenusTests
{
    static readonly TimeSpan Three = TimeSpan.FromSeconds(3);
    static readonly TimeSpan Ten = TimeSpan.FromSeconds(10);

    static (ShellApp App, AutomationElement Edit) Loaded(ShellRun run)
    {
        var app = ShellApp.Launch(run);
        var edit = app.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
        return (app, edit);
    }

    static void TypeInto(ShellApp app, AutomationElement edit, string keys)
    {
        app.Activate();
        edit.SetFocus();
        Thread.Sleep(300);
        WinForms.SendKeys.SendWait(keys);
        Thread.Sleep(400);
    }

    static void AssertZoom(ShellApp app, double expected) =>
        Assert.True(Uia.WaitUntil(() => app.SavedZoom() is { } z && Math.Abs(z - expected) < 0.001, TimeSpan.FromSeconds(2)),
            $"saved zoom {app.SavedZoom()} != {expected}");

    [Fact]
    public void Editing_keys_work_in_a_text_box()
    {
        var (app, edit) = Loaded(ShellRun.Fresh());
        using var _ = app;
        TypeInto(app, edit, "hello spike");
        WinForms.SendKeys.SendWait("^a");
        WinForms.SendKeys.SendWait("^c");
        Thread.Sleep(400);
        Assert.Equal("hello spike", ShellApp.Clipboard());
        WinForms.SendKeys.SendWait("{END}");
        WinForms.SendKeys.SendWait("^v");
        Thread.Sleep(400);
        Assert.Equal("hello spikehello spike", Uia.Value(edit));
        WinForms.SendKeys.SendWait("^z");
        Thread.Sleep(400);
        Assert.Equal("hello spike", Uia.Value(edit));
    }

    [Fact]
    public void Browser_keys_do_nothing()
    {
        var (app, edit) = Loaded(ShellRun.Fresh());
        using var _ = app;
        TypeInto(app, edit, "keep");
        WinForms.SendKeys.SendWait("{F5}");
        Thread.Sleep(3000);
        Assert.Equal("keep", Uia.Value(app.WaitFor(ControlType.Edit, "Type here", Three)));
        WinForms.SendKeys.SendWait("^p");
        Assert.False(Uia.WaitUntil(() => Uia.TopWindow("Print") != null, Three));
        WinForms.SendKeys.SendWait("{F12}");
        WinForms.SendKeys.SendWait("^+i");
        Assert.False(Uia.WaitUntil(() => Uia.TopWindow("DevTools") != null, Three));
    }

    [Fact]
    public void Zoom_keys_step_and_persist()
    {
        var (app, edit) = Loaded(ShellRun.Fresh());
        using var _ = app;
        TypeInto(app, edit, "^=");
        WinForms.SendKeys.SendWait("^=");
        AssertZoom(app, 1.2);
        WinForms.SendKeys.SendWait("^-");
        AssertZoom(app, 1.1);
        WinForms.SendKeys.SendWait("^{ADD}");
        AssertZoom(app, 1.2);
        WinForms.SendKeys.SendWait("^{SUBTRACT}");
        AssertZoom(app, 1.1);
        WinForms.SendKeys.SendWait("^0");
        AssertZoom(app, 1.0);
    }

    [Fact]
    public void Zoom_stops_at_two_and_a_half()
    {
        var (app, edit) = Loaded(ShellRun.Fresh());
        using var _ = app;
        TypeInto(app, edit, "");
        for (var i = 0; i < 25; i++)
        {
            WinForms.SendKeys.SendWait("^=");
            Thread.Sleep(100);
        }
        AssertZoom(app, 2.5);
    }

    [Fact]
    public async Task Ctrl_wheel_zoom_is_saved_and_used_next_time()
    {
        double saved;
        using (var app = Loaded(ShellRun.Fresh()).App)
        {
            app.CtrlWheel(2);
            Assert.True(Uia.WaitUntil(() => app.SavedZoom() > 1.0, TimeSpan.FromSeconds(3)), $"zoom {app.SavedZoom()}");
            Thread.Sleep(500);
            saved = app.SavedZoom()!.Value;
            app.Close();
            Assert.True(await app.WaitForExitAsync(Ten));
        }
        using var next = ShellApp.Launch(ShellRun.Same());
        var want = "zoom.applied " + saved.ToString("0.##", CultureInfo.InvariantCulture);
        Assert.True(Uia.WaitUntil(() => next.Log().Any(l => l.EndsWith(want)), Ten), string.Join("\n", next.Log()));
    }

    [Fact]
    public void Menus_are_the_windows_layout()
    {
        using var app = ShellApp.Launch(ShellRun.Fresh());
        Assert.Equal(["File", "View", "Help"], app.MenuBarItems());
        Assert.Equal(["Open Projects Folder", "Exit"], app.OpenMenu("File").Select(Uia.NameOf));
        app.CloseMenus();
        var view = app.OpenMenu("View");
        Assert.Equal(["Bigger Text", "Smaller Text", "Actual Size"], view.Select(Uia.NameOf));
        Assert.Equal("Ctrl+=", view[0].Current.AcceleratorKey);
        app.CloseMenus();
        Assert.Equal(["Start Here", "Get Help", "Show Log File", "About Food Optimizer"], app.OpenMenu("Help").Select(Uia.NameOf));
        app.CloseMenus();
        Assert.Equal("Ctrl+O", app.OpenMenu("File")[0].Current.AcceleratorKey);
        app.CloseMenus();
    }

    [Fact]
    public void View_menu_zooms()
    {
        using var app = Loaded(ShellRun.Fresh()).App;
        app.Menu("View", "Bigger Text");
        AssertZoom(app, 1.1);
    }

    static void CloseDialogWithOk(AutomationElement dialog)
    {
        Uia.Invoke(Uia.WaitFor(dialog, ControlType.Button, "OK", Three));
    }

    [Fact]
    public void Get_help_names_the_address()
    {
        using var app = ShellApp.Launch(ShellRun.Fresh());
        app.Menu("Help", "Get Help");
        var d = app.WaitForDialog(Ten);
        Uia.WaitFor(d, ControlType.Text, "Food Optimizer support", Three);
        Uia.WaitFor(d, ControlType.Text, "sohum.patnaik@fsi.org", Three);
        CloseDialogWithOk(d);
        Assert.True(Uia.WaitUntil(() => app.Dialog("Food Optimizer") == null, Three));
    }

    [Fact]
    public void About_shows_the_version()
    {
        using var app = ShellApp.Launch(ShellRun.Fresh());
        app.Menu("Help", "About Food Optimizer");
        var d = app.WaitForDialog(Ten);
        Uia.WaitFor(d, ControlType.Text, "Food Optimizer " + AppIdentity.Version, Three);
        CloseDialogWithOk(d);
    }

    [Fact]
    public void Show_log_file_without_a_log_says_so()
    {
        var run = ShellRun.Fresh() with { Env = new Dictionary<string, string> { ["FOODOPT_TEST_ARCH"] = "x86" } };
        using var app = ShellApp.Launch(run);
        app.WaitFor(ControlType.Text, "This PC is not supported", TimeSpan.FromSeconds(30));
        Assert.False(File.Exists(Path.Combine(app.StateDir, "launcher.log")));
        app.Menu("Help", "Show Log File");
        var d = app.WaitForDialog(Ten);
        Uia.WaitFor(d, ControlType.Text, "There is no log file yet", Three);
        CloseDialogWithOk(d);
    }

    static void CloseWindow(AutomationElement w)
    {
        try { ((WindowPattern)w.GetCurrentPattern(WindowPattern.Pattern)).Close(); } catch (Exception) { }
    }

    [Fact]
    public void Show_log_file_selects_the_log()
    {
        using var app = Loaded(ShellRun.Fresh()).App;
        app.Menu("Help", "Show Log File");
        AutomationElement? w = null;
        var leaf = Path.GetFileName(app.StateDir);
        Assert.True(Uia.WaitUntil(() => (w = Uia.TopWindow(leaf, "CabinetWClass")) != null, Ten));
        CloseWindow(w!);
    }

    [Fact]
    public void Open_projects_folder_opens_it()
    {
        using var app = ShellApp.Launch(ShellRun.Fresh());
        app.Menu("File", "Open Projects Folder");
        AutomationElement? w = null;
        Assert.True(Uia.WaitUntil(() => (w = Uia.TopWindow("FoodOptimizer", "CabinetWClass")) != null, Ten));
        Assert.True(Directory.Exists(app.ProjectsDir));
        CloseWindow(w!);
    }

    /// A command that fails says so, and the app keeps working.
    [Fact]
    public void Open_projects_folder_when_the_path_is_a_file_says_so()
    {
        var run = ShellRun.Fresh();
        Directory.CreateDirectory(run.TempRoot);
        var file = Path.Combine(run.TempRoot, "projects.txt");
        File.WriteAllText(file, "not a folder");
        using var app = ShellApp.Launch(run with { Env = new Dictionary<string, string> { ["FOODOPT_PROJECTS_DIR"] = file } });
        app.Menu("File", "Open Projects Folder");
        var d = app.WaitForDialog(Ten);
        Uia.WaitFor(d, ControlType.Text, "Open Projects Folder did not work", Three);
        CloseDialogWithOk(d);
        Assert.True(Uia.WaitUntil(() => app.Dialog("Food Optimizer") == null, Three));
        Assert.True(File.Exists(file));
        app.Menu("Help", "About Food Optimizer");
        var about = app.WaitForDialog(Ten);
        Uia.WaitFor(about, ControlType.Text, "Food Optimizer " + AppIdentity.Version, Three);
        CloseDialogWithOk(about);
    }

    [Fact]
    public void Start_here_opens_the_text_file()
    {
        using var app = ShellApp.Launch(ShellRun.Fresh());
        app.Menu("Help", "Start Here");
        AutomationElement? w = null;
        Assert.True(Uia.WaitUntil(() => (w = Uia.TopWindow("Start Here")) != null, Ten));
        CloseWindow(w!);
    }

    [Fact]
    public async Task Exit_quits()
    {
        using var app = Loaded(ShellRun.Fresh()).App;
        app.Menu("File", "Exit");
        Assert.True(await app.WaitForExitAsync(TimeSpan.FromSeconds(5)));
        Assert.True(Uia.WaitUntil(() => app.StagePythons().Count == 0, TimeSpan.FromSeconds(5)));
    }
}
